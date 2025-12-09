from scalesim.scale_sim import scalesim
from models.sparse_info import SparseInfo
from arch.accelerator import Accelerator
from models.models import ModelConfig

from utils.utils import split_into_chunks, set_csv_column, set_file_row

import torch
import os

class SimulatorComp:
    """
    A class to handle the simulation of computation cycles for a given model configuration and sparse information.
    """

    def __init__(self, accelerator: Accelerator):
        self.accelerator = accelerator
        self.compression_ratio_list = []

    def run_attn_focus(self, mask_zero, mask_similar, group_idx, layer_config):
        num_vector_q = int((mask_zero.numel() - (torch.sum(mask_zero) + torch.sum(mask_similar))) / (mask_zero.shape[-1] * mask_zero.shape[0]))
        # num_vector_k = int((mask_zero.numel() - torch.sum(mask_zero)) / (mask_zero.shape[-1] * mask_zero.shape[0]))
        num_vector_k = int(mask_zero.shape[1] - torch.sum(torch.all(mask_zero, dim=(0,-1)), dim=-1))
        num_vector_v = num_vector_k

        if self.accelerator.SEC_only:
            num_vector_q = num_vector_k

        compression_ratio = num_vector_k / num_vector_q
        self.compression_ratio_list.append(compression_ratio)
        # print("compression ratio: ", compression_ratio)

        num_ops = num_vector_q * num_vector_k * layer_config['dim_per_head'] * layer_config['num_heads'] // 2
        dense_ops = layer_config['seq_len'] * layer_config['seq_len'] * layer_config['dim_per_head'] * layer_config['num_heads'] // 2

        m_tile_size = self.accelerator.buffer_config["m_in_size"]
        n_tile_size = self.accelerator.systolic_config["array_width"]
        k_tile_size = self.accelerator.systolic_config["array_height"]

        num_partition_q = (num_vector_q + m_tile_size - 1) // m_tile_size
        chunk_size_q = num_vector_q // num_partition_q

        qk_cycles = self.call_scalesim(chunk_size_q, n_tile_size, k_tile_size, verbose=False) * num_partition_q

        N_repeat = (num_vector_k + n_tile_size - 1) // n_tile_size
        K_repeat = (layer_config['dim_per_head'] + k_tile_size - 1) // k_tile_size

        qk_cycles *= N_repeat * K_repeat

        num_partition_k = (num_vector_k + m_tile_size - 1) // m_tile_size
        chunk_size_k = num_vector_k // num_partition_k

        sv_cycles = self.call_scalesim(chunk_size_k, n_tile_size, k_tile_size, verbose=False) * num_partition_k

        K_repeat = (num_vector_v + n_tile_size - 1) // n_tile_size
        N_repeat = (layer_config['dim_per_head'] + k_tile_size - 1) // k_tile_size

        sv_cycles *= N_repeat * K_repeat

        is_causal = True
        compute_cycles = (qk_cycles + sv_cycles) / 2 if is_causal else (qk_cycles + sv_cycles)
        compute_cycles *= layer_config['num_heads']

        num_ops += num_vector_k * num_vector_v * layer_config['dim_per_head'] * layer_config['num_heads'] // 2
        dense_ops += layer_config['seq_len'] * layer_config['seq_len'] * layer_config['dim_per_head'] * layer_config['num_heads'] // 2

        return compute_cycles, num_ops, dense_ops

    def run_linear_focus(self, mask_zero, mask_similar, group_idx, layer_config):

        no_overlap = not (mask_zero & mask_similar).any()
        assert no_overlap

        assert mask_zero.shape[0] == 1
        M_size = layer_config['seq_len']
        N_size = layer_config['out_features']
        K_size = layer_config['in_features']

        m_tile_size = self.accelerator.buffer_config["m_in_size"]
        n_tile_size = self.accelerator.systolic_config["array_width"]
        k_tile_size = self.accelerator.systolic_config["array_height"]

        num_vector = int((mask_zero.numel() - (torch.sum(mask_zero) + torch.sum(mask_similar))) / mask_zero.shape[-1])
        # num_tokens = int((mask_zero.numel() - torch.sum(mask_zero)) // (mask_zero.shape[-1]))
        num_tokens = int(mask_zero.shape[1] - torch.sum(torch.all(mask_zero, dim=-1), dim=-1))

        if self.accelerator.SEC_only:
            num_vector = num_tokens
        
        assert num_tokens >= num_vector
        if num_vector == 0:
            return 0, 0, M_size * N_size * K_size

        compression_ratio = num_tokens / num_vector
        self.compression_ratio_list.append(compression_ratio)
        # print("compression ratio: ", compression_ratio)

        num_partition = (num_tokens + m_tile_size - 1) // m_tile_size
        chunk_size = num_vector // num_partition
        # if chunk_size < 32:
        #     compute_cycles = 32 * num_partition
        # else:
        compute_cycles = self.call_scalesim(chunk_size, n_tile_size, k_tile_size, verbose=False) * num_partition

        N_repeat = (N_size + n_tile_size - 1) // n_tile_size
        K_repeat = (K_size + k_tile_size - 1) // k_tile_size

        compute_cycles *= N_repeat * K_repeat
        num_ops = num_vector * N_size * K_size
        dense_ops = M_size * N_size * K_size

        return compute_cycles, num_ops, dense_ops

    def run_linear_scatter_focus(self, mask_zero, mask_similar, group_idx, layer_config):
        num_tokens = int((mask_zero.numel() - torch.sum(mask_zero)) // (mask_zero.shape[-1]))

        num_repeat_k = (layer_config['in_features'] + self.accelerator.systolic_config["array_height"] - 1) // self.accelerator.systolic_config["array_height"]
        num_repeat_n = (layer_config['out_features'] + self.accelerator.systolic_config["array_width"] - 1) // self.accelerator.systolic_config["array_width"]

        scatter_cycles = (num_tokens * num_repeat_k * num_repeat_n + self.accelerator.num_scatter_vector - 1) // self.accelerator.num_scatter_vector

        return scatter_cycles

    def run_qk_scatter_focus(self, mask_zero, mask_similar, group_idx, layer_config):
        num_tokens = int((mask_zero.numel() - torch.sum(mask_zero)) // (mask_zero.shape[-1] * mask_zero.shape[0]))

        num_repeat_k = (layer_config['dim_per_head'] * layer_config['num_heads'] + self.accelerator.systolic_config["array_height"] - 1) // self.accelerator.systolic_config["array_height"]
        num_repeat_n = (layer_config['seq_len'] + self.accelerator.systolic_config["array_width"] - 1) // self.accelerator.systolic_config["array_width"]

        scatter_cycles = (num_tokens * num_repeat_k * num_repeat_n + self.accelerator.num_scatter_vector - 1) // self.accelerator.num_scatter_vector

        return scatter_cycles
    
    def get_scatted_ops(self, mask_zero, mask_similar, group_idx, layer_config, tile_size):
        num_tokens = int(mask_zero.shape[1] - torch.sum(torch.all(mask_zero, dim=-1), dim=-1))

        num_repeat_k = (layer_config['in_features'] + tile_size - 1) // tile_size
        N_size = layer_config['out_features']
        num_scatter_ops = num_tokens * N_size * num_repeat_k

        return num_scatter_ops


    def run_gather_linear_focus(self, mask_zero, mask_similar, group_idx, layer_config):
        num_tokens = int(mask_zero.shape[1] - torch.sum(torch.all(mask_zero, dim=-1), dim=-1))
        tile_size_m = self.accelerator.buffer_config["m_in_size"]
        tile_size_n = self.accelerator.buffer_config["n_size"]
        repeat_n = tile_size_n // self.accelerator.systolic_config["array_width"]
        block_size = self.accelerator.block_size

        num_tile_m = (num_tokens + tile_size_m - 1) // tile_size_m
        num_tile_n = (layer_config['out_features'] + tile_size_n - 1) // tile_size_n

        cycles_per_tile = block_size * tile_size_m * repeat_n
        cycles = cycles_per_tile * num_tile_m * num_tile_n

        return cycles, cycles_per_tile

    def run_linear_dense(self, layer_config):
        '''weight stationary systolic array'''

        M_size = layer_config['seq_len']
        N_size = layer_config['out_features']
        K_size = layer_config['in_features']

        m_tile_size = 1024

        M_num_tile = (M_size + m_tile_size - 1) // m_tile_size
        m_input_size = (M_size + M_num_tile - 1) // M_num_tile

        compute_cycles = 0

        n_tile_size = self.accelerator.systolic_config["array_height"]
        k_tile_size = self.accelerator.systolic_config["array_width"]

        compute_cycles += self.call_scalesim(m_input_size, n_tile_size, k_tile_size) * M_num_tile

        N_repeat = (N_size + n_tile_size - 1) // n_tile_size
        K_repeat = (K_size + k_tile_size - 1) // k_tile_size

        compute_cycles *= N_repeat * K_repeat

        return compute_cycles

    def run_attn_dense(self, layer_config):
        M_size = layer_config['seq_len']
        N_size = layer_config['seq_len']
        K_size = layer_config['dim_per_head']

        m_tile_size = 1024
        n_tile_size = self.accelerator.systolic_config["array_height"]
        k_tile_size = self.accelerator.systolic_config["array_width"]
        
        M_num_tile = (M_size + m_tile_size - 1) // m_tile_size
        m_input_size = (M_size + M_num_tile - 1) // M_num_tile

        qk_cycles = self.call_scalesim(m_input_size, n_tile_size, k_tile_size) * M_num_tile

        N_repeat = (N_size + n_tile_size - 1) // n_tile_size
        K_repeat = (K_size + k_tile_size - 1) // k_tile_size

        qk_cycles *= N_repeat * K_repeat * layer_config['num_heads']

        sv_cycles = qk_cycles
        is_causal = True
        compute_cycles = (qk_cycles + sv_cycles) / 2 if is_causal else (qk_cycles + sv_cycles)
        return compute_cycles

    def run_linear_adaptiv(self, layer_config, sparsity):
        preprocess_cycles = (layer_config['seq_len'] * layer_config['in_features'] + 63) // 64
        reduced_seq_len = layer_config['seq_len'] * (1 - sparsity)
        M_size = reduced_seq_len
        N_size = layer_config['out_features']
        K_size = layer_config['in_features']

        n_tile_size = self.accelerator.array_config['MAC_line_num']
        m_tile_size = self.accelerator.array_config['MAC_line_width']
        k_tile_size = self.accelerator.buffer_config['k_size']


        n_repeat_num = (N_size + n_tile_size - 1) // n_tile_size
        m_repeat_num = (M_size + m_tile_size - 1) // m_tile_size
        k_repeat_num = (K_size + k_tile_size - 1) // k_tile_size

        compute_cycles = k_tile_size * n_repeat_num * m_repeat_num * k_repeat_num

        num_ops = M_size * N_size * K_size
        dense_ops = layer_config['seq_len'] * layer_config['in_features'] * layer_config['out_features']

        return compute_cycles + preprocess_cycles, num_ops, dense_ops

    def run_attn_adaptiv(self, layer_config, sparsity):

        qk_config = {
            'seq_len': layer_config['seq_len'],
            'in_features': layer_config['dim_per_head'],
            'out_features': layer_config['seq_len']
        }
        qk_cycles, num_ops, dense_ops = self.run_linear_adaptiv(qk_config, sparsity)
        kv_cycles = qk_cycles

        total_cycles = (qk_cycles + kv_cycles) * layer_config['num_heads']
        num_ops *= 2 * layer_config['num_heads']
        dense_ops *= 2 * layer_config['num_heads']
        is_causal = True
        if is_causal:
            total_cycles = total_cycles // 2
            num_ops = num_ops // 2
            dense_ops = dense_ops // 2

        return total_cycles, num_ops, dense_ops

    def run_linear_cmc(self, layer_config, linear_sparsity, num_frames, num_patches, is_kv=False):
        '''half weight stationary'''
        compute_cycles = 0

        M_size = layer_config['seq_len']
        N_size = layer_config['out_features']
        K_size = layer_config['in_features']

        assert num_frames * num_patches == M_size

        num_partition = (num_frames + self.accelerator.codec_config['frames_per_partition'] - 1) // self.accelerator.codec_config['frames_per_partition']
        num_cmp = (num_frames - num_partition) * num_patches * num_patches * K_size
        preprocess_cycles = (num_cmp + self.accelerator.codec_config['PE_width'] * self.accelerator.codec_config['PE_height'] - 1) // (self.accelerator.codec_config['PE_width'] * self.accelerator.codec_config['PE_height'])
        if is_kv:
            preprocess_cycles = 0

        n_repeat = (N_size + self.accelerator.systolic_config['array_height'] - 1) // self.accelerator.systolic_config['array_height']
        k_repeat = (K_size + self.accelerator.systolic_config['array_width'] - 1) // self.accelerator.systolic_config['array_width']
        m_tile_size = int(M_size * (1 - linear_sparsity) + num_partition - 1) // num_partition
        m_repeat = num_partition

        compute_cycles = self.call_scalesim(m_tile_size, self.accelerator.systolic_config['array_height'], self.accelerator.systolic_config['array_width']) * m_repeat * n_repeat * k_repeat
        # preprocess_latency = (preprocess_cycles + num_partition - 1) // num_partition
        preprocess_latency = 0

        num_ops = M_size * N_size * K_size * (1 - linear_sparsity)
        dense_ops = layer_config['seq_len'] * layer_config['in_features'] * layer_config['out_features']

        return compute_cycles + preprocess_latency, num_ops, dense_ops
    
    def run_attn_cmc(self, layer_config, q_sparsity, s_sparsity, num_frames, num_patches):
        '''half weight stationary'''
        qk_config = {
            'seq_len': layer_config['seq_len'],
            'in_features': layer_config['dim_per_head'],
            'out_features': layer_config['seq_len']
        }
        qk_cycles, num_ops_qk, dense_ops_qk = self.run_linear_cmc(qk_config, q_sparsity, num_frames, num_patches, is_kv=False)

        sv_config = {
            'seq_len': layer_config['seq_len'],
            'in_features': layer_config['seq_len'],
            'out_features': layer_config['dim_per_head']
        }
        sv_cycles, num_ops_sv, dense_ops_sv = self.run_linear_cmc(sv_config, s_sparsity, num_frames, num_patches, is_kv=False)
        compute_cycles = (qk_cycles + sv_cycles) * layer_config['num_heads']
        num_ops_qk *= layer_config['num_heads']
        num_ops_sv *= layer_config['num_heads']
        dense_ops_qk *= layer_config['num_heads']
        dense_ops_sv *= layer_config['num_heads']

        is_causal = True
        if is_causal:
            compute_cycles = compute_cycles // 2
            num_ops_qk = num_ops_qk // 2
            num_ops_sv = num_ops_sv // 2
            dense_ops_qk = dense_ops_qk // 2
            dense_ops_sv = dense_ops_sv // 2

        num_ops = num_ops_qk + num_ops_sv
        dense_ops = dense_ops_qk + dense_ops_sv

        return compute_cycles, num_ops, dense_ops



    def call_scalesim(self, M_size, N_size, K_size, verbose=False):
        """
        Call the scalesim with the given matrix sizes.
        """
        config = os.path.join(os.path.dirname(__file__), "scalesim_cfg/config.cfg")
        topology = os.path.join(os.path.dirname(__file__), "scalesim_cfg/gemm.csv")
        gemm_input = True

        set_csv_column(topology, "M", M_size)
        set_csv_column(topology, "N", N_size)
        set_csv_column(topology, "K", K_size)

        set_file_row(config, "ArrayHeight", str(self.accelerator.systolic_config['array_height']))
        set_file_row(config, "ArrayWidth", str(self.accelerator.systolic_config['array_width']))

        s = scalesim(save_disk_space=True, verbose=verbose,
                    config=config,
                    topology=topology,
                    input_type_gemm=gemm_input)

        top_path = os.path.join(os.path.dirname(__file__), 'scalesim_logs')
        s.run_scale(top_path=top_path)
        compute_cycles = s.get_total_cycles()
        return compute_cycles


if __name__ == "__main__":
    model_type = "llava_vid"
    model_args = "pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,conv_template=qwen_1_5,max_frames_num=64,mm_spatial_pool_mode=average"

    model_config = ModelConfig(model_type, model_args)
    model_config.add_seq_len(11648)

    # sparse_info = SparseInfo("/home/cw541/vlm/lmms-eval/lmms_eval/output/focus_info_2.pth")
    # accelerator = Accelerator("focus")

    # focus_simulator = SimulatorComp(model_config, accelerator)
    # focus_simulator.run_focus(sparse_info)

    accelerator = Accelerator("dense")
    dense_simulator = SimulatorComp(model_config, accelerator)
    dense_simulator.run_dense()
