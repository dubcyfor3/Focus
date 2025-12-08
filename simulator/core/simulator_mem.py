import torch
from arch.accelerator import Accelerator

class FocusData:
    elemment_size = 2  # FP16
    data_type = [
        "input",
        "concentrate_out",
        "wgt",
        "output",
        "layouter",
        "similarity_map",
        "similarity_table",
    ]

    def __init__(self):
        pass  # No need to set them in __init__

class MemCounter:
    def __init__(self, name_space):
        self.sram_read = {name: 0 for name in name_space}
        self.sram_write = {name: 0 for name in name_space}
        self.dram_read = {name: 0 for name in name_space}
        self.dram_write = {name: 0 for name in name_space}

    def reset(self):
        for d in [self.sram_read, self.sram_write, self.dram_read, self.dram_write]:
            for key in d:
                d[key] = 0

    def add(self, other: "MemCounter"):
        if not isinstance(other, MemCounter):
            raise ValueError("Expected MemCounter")
        for name in self.sram_read:
            self.sram_read[name] += other.sram_read[name]
            self.sram_write[name] += other.sram_write[name]
            self.dram_read[name] += other.dram_read[name]
            self.dram_write[name] += other.dram_write[name]

    def __add__(self, other):
        result = MemCounter(self.sram_read.keys())
        result.add(self)
        result.add(other)
        return result

    def __iadd__(self, other):
        self.add(other)
        return self

    def __imul__(self, value: int):
        if not isinstance(value, (int, float)):
            raise ValueError("Only scalar multiplication is supported.")
        for name in self.sram_read:
            self.sram_read[name] *= value
            self.sram_write[name] *= value
            self.dram_read[name] *= value
            self.dram_write[name] *= value
        return self

    def __repr__(self):
        # return (f"SRAM Read: {self.sram_read}\n"
        #         f"SRAM Write: {self.sram_write}\n"
        #         f"DRAM Read: {self.dram_read}\n"
        #         f"DRAM Write: {self.dram_write}")
        return "mem_counter"


class SimulatorMem:
    def __init__ (self, accelerator: Accelerator):
        self.accelerator = accelerator
        self.data = FocusData()

    def run_linear_focus(self, mask_zero, mask_similar, group_idx, layer_config):
        num_cluster = int((mask_zero.numel() - torch.sum(mask_zero) - torch.sum(mask_similar) + mask_zero.shape[-1] - 1) // mask_zero.shape[-1])
        num_tokens = int(mask_zero.shape[1] - torch.sum(torch.all(mask_zero, dim=-1), dim=-1))

        if self.accelerator.SEC_only:
            num_cluster = num_tokens

        K_size = layer_config['in_features']
        N_size = layer_config['out_features']

        array_height = self.accelerator.systolic_config["array_height"]
        array_width = self.accelerator.systolic_config["array_width"]

        num_buffer_repeat_M = (num_tokens + self.accelerator.buffer_config["m_out_size"] - 1) // self.accelerator.buffer_config["m_out_size"]
        num_buffer_repeat_K = (K_size + self.accelerator.buffer_config["k_size"] - 1) // self.accelerator.buffer_config["k_size"]
        num_buffer_repeat_N = (N_size + self.accelerator.buffer_config["n_size"] - 1) // self.accelerator.buffer_config["n_size"]

        num_array_repeat_M = (num_tokens + self.accelerator.buffer_config["m_out_size"] - 1) // self.accelerator.buffer_config["m_out_size"]
        num_array_repeat_K = (K_size + array_height - 1) // array_height
        num_array_repeat_N = (N_size + array_width - 1) // array_width

        num_cycles = (num_cluster * K_size * N_size + array_height * array_width - 1) // (array_height * array_width)

        mem_counter = MemCounter(self.data.data_type)

        mem_counter.sram_read["input"] += num_cycles * array_height * self.data.elemment_size
        mem_counter.sram_write["concentrate_out"] += num_cycles * array_width * self.data.elemment_size

        mem_counter.sram_read["concentrate_out"] += num_tokens * ((K_size + array_width - 1) // array_width) * N_size * self.data.elemment_size
        mem_counter.sram_write["output"] += num_tokens * ((K_size + array_width - 1) // array_width) * N_size * self.data.elemment_size

        mem_counter.sram_read["wgt"] += K_size * N_size * num_array_repeat_M * self.data.elemment_size

        mem_counter.dram_read["input"] += num_cluster * K_size * num_buffer_repeat_N * self.data.elemment_size
        mem_counter.sram_write["input"] += num_cluster * K_size * num_buffer_repeat_N * self.data.elemment_size

        mem_counter.dram_read["wgt"] += K_size * N_size * num_buffer_repeat_M * self.data.elemment_size
        mem_counter.sram_write["wgt"] += K_size * N_size * num_buffer_repeat_M * self.data.elemment_size

        return mem_counter

    def run_detect_linear_focus(self, mask_zero, mask_similar, group_idx, layer_config):
        mem_counter = MemCounter(self.data.data_type)

        num_tokens = int(mask_zero.shape[1] - torch.sum(torch.all(mask_zero, dim=-1), dim=-1))
        num_vectors = int((mask_zero.numel() - (torch.sum(mask_zero) + torch.sum(mask_similar)) + mask_zero.shape[-1] - 1) // mask_zero.shape[-1])

        if self.accelerator.SEC_only:
            num_vectors = num_tokens

        mem_counter.sram_read["output"] += num_tokens * layer_config['out_features'] * self.data.elemment_size \
                                        + num_tokens * ((layer_config['out_features'] + 31) // 32) * self.data.elemment_size
        mem_counter.sram_write["layouter"] += num_tokens * layer_config['out_features'] * self.data.elemment_size \
                                              + num_tokens * ((layer_config['out_features'] + 31) // 32) * self.data.elemment_size

        mem_counter.sram_read["layouter"] += num_tokens * 7 * layer_config['out_features'] * self.data.elemment_size \
                                            + num_tokens * 7 * ((layer_config['out_features'] + 31) // 32) * self.data.elemment_size

        mem_counter.sram_write["similarity_map"] += num_tokens * ((layer_config['out_features'] + 31) // 32) * self.data.elemment_size

        mem_counter.sram_read["similarity_table"] += num_tokens * layer_config['out_features'] * self.data.elemment_size
        mem_counter.sram_write["similarity_table"] += num_tokens * layer_config['out_features'] * self.data.elemment_size

        mem_counter.sram_read["similarity_table"] += num_vectors * layer_config['out_features'] * self.data.elemment_size
        mem_counter.dram_write["input"] += num_vectors * layer_config['out_features'] * self.data.elemment_size

        return mem_counter

    def run_detect_attn_focus(self, mask_zero, mask_similar, group_idx, layer_config):
        qk_config = {
            "seq_len": layer_config['seq_len'],
            "in_features": layer_config['dim_per_head'],
            "out_features": layer_config['seq_len'],
        }
        # assert layer_config['num_heads'] == mask_zero.shape[0], "Number of heads does not match the mask_zero shape."
        mem_counter_attn = MemCounter(self.data.data_type)
        for i in range(mask_zero.shape[0]):
            cur_mask_zero = mask_zero[i]
            cur_mask_similar = mask_similar[i]
            cur_group_idx = group_idx[i]
            mem_counter_qk = self.run_detect_linear_focus(cur_mask_zero, cur_mask_similar, cur_group_idx, qk_config)
            mem_counter_attn += mem_counter_qk

        return mem_counter_attn

    def run_attn_focus(self, mask_zero, mask_similar, group_idx, layer_config):
        is_causal = True
        num_tokens = int(mask_zero.shape[1] - torch.sum(torch.all(mask_zero, dim=(0,-1)), dim=-1))

        qk_config = {
            "seq_len": num_tokens,
            "in_features": layer_config['dim_per_head'],
            "out_features": num_tokens,
        }
        # assert layer_config['num_heads'] == mask_zero.shape[0], "Number of heads does not match the mask_zero shape."
        mem_counter_attn = MemCounter(self.data.data_type)
        for i in range(mask_zero.shape[0]):
            cur_mask_zero = mask_zero[i]
            cur_mask_similar = mask_similar[i]
            cur_group_idx = group_idx[i]
            mem_counter_qk = self.run_linear_focus(cur_mask_zero, cur_mask_similar, cur_group_idx, qk_config)
            mem_counter_attn += mem_counter_qk

        mem_counter_intermediate = MemCounter(self.data.data_type)
        mem_counter_attn += mem_counter_intermediate

        sv_config = {
            "seq_len": num_tokens,
            "in_features": num_tokens,
            "out_features": layer_config['dim_per_head'],
        }
        mem_counter_sv = self.run_linear_focus_no_cluster(sv_config)
        mem_counter_sv *= layer_config['num_heads']
        mem_counter_attn += mem_counter_sv
        if is_causal:
            mem_counter_attn *= 0.5
        return mem_counter_attn

    def run_linear_focus_no_cluster(self, layer_config):
        M_size = layer_config['seq_len']
        N_size = layer_config['out_features']
        K_size = layer_config['in_features']

        array_height = self.accelerator.systolic_config["array_height"]
        array_width = self.accelerator.systolic_config["array_width"]

        num_buffer_repeat_M = (M_size + self.accelerator.buffer_config["m_out_size"] - 1) // self.accelerator.buffer_config["m_out_size"]
        num_buffer_repeat_K = (K_size + self.accelerator.buffer_config["k_size"] - 1) // self.accelerator.buffer_config["k_size"]
        num_buffer_repeat_N = (N_size + self.accelerator.buffer_config["n_size"] - 1) // self.accelerator.buffer_config["n_size"]

        num_array_repeat_M = (M_size + self.accelerator.buffer_config["m_out_size"] - 1) // self.accelerator.buffer_config["m_out_size"]
        num_array_repeat_K = (K_size + array_height - 1) // array_height
        num_array_repeat_N = (N_size + array_width - 1) // array_width

        num_cycles = (M_size * K_size * N_size + array_height * array_width - 1) // (array_height * array_width)

        mem_counter = MemCounter(self.data.data_type)

        mem_counter.sram_read["input"] += num_cycles * array_height * self.data.elemment_size
        mem_counter.sram_write["output"] += num_cycles * array_width * self.data.elemment_size

        mem_counter.sram_read["wgt"] += K_size * N_size * num_array_repeat_M * self.data.elemment_size

        mem_counter.dram_read["input"] += M_size * K_size * num_buffer_repeat_N * self.data.elemment_size
        mem_counter.sram_write["input"] += M_size * K_size * num_buffer_repeat_N * self.data.elemment_size

        mem_counter.dram_read["wgt"] += K_size * N_size * num_buffer_repeat_M * self.data.elemment_size
        mem_counter.sram_write["wgt"] += K_size * N_size * num_buffer_repeat_M * self.data.elemment_size

        mem_counter.dram_write["output"] += M_size * N_size * self.data.elemment_size
        mem_counter.sram_read["output"] += M_size * N_size * self.data.elemment_size

        return mem_counter
    
    def run_linear_dense(self, layer_config):

        '''weight stationary systolic array'''

        M_size = layer_config['seq_len']
        N_size = layer_config['out_features']
        K_size = layer_config['in_features']

        m_tile_size = self.accelerator.buffer_config["m_in_size"]
        k_tile_size = self.accelerator.buffer_config["k_size"]
        n_tile_size = self.accelerator.buffer_config["n_size"]

        M_repeat = (M_size + m_tile_size - 1) // m_tile_size
        K_repeat = (K_size + k_tile_size - 1) // k_tile_size
        N_repeat = (N_size + n_tile_size - 1) // n_tile_size

        array_height = self.accelerator.systolic_config["array_height"]
        array_width = self.accelerator.systolic_config["array_width"]

        # num_buffer_repeat_M = (M_size + self.accelerator.buffer_config["m_out_size"] - 1) // self.accelerator.buffer_config["m_out_size"]
        # num_buffer_repeat_K = (K_size + self.accelerator.buffer_config["k_size"] - 1) // self.accelerator.buffer_config["k_size"]
        # num_buffer_repeat_N = (N_size + self.accelerator.buffer_config["n_size"] - 1) // self.accelerator.buffer_config["n_size"]

        # num_array_repeat_M = (M_size + self.accelerator.buffer_config["m_out_size"] - 1) // self.accelerator.buffer_config["m_out_size"]
        # num_array_repeat_K = (K_size + array_height - 1) // array_height
        # num_array_repeat_N = (N_size + array_width - 1) // array_width

        num_cycles = (M_size * K_size * N_size + array_height * array_width - 1) // (array_height * array_width)

        mem_counter = MemCounter(self.data.data_type)

        mem_counter.sram_read["input"] += num_cycles * array_height * self.data.elemment_size
        mem_counter.sram_write["output"] += num_cycles * array_width * self.data.elemment_size

        mem_counter.sram_read["wgt"] += K_size * N_size * M_repeat * self.data.elemment_size

        mem_counter.dram_read["input"] += M_size * K_size * N_repeat * self.data.elemment_size
        mem_counter.sram_write["input"] += M_size * K_size * N_repeat * self.data.elemment_size

        mem_counter.dram_read["wgt"] += K_size * N_size * M_repeat * self.data.elemment_size
        mem_counter.sram_write["wgt"] += K_size * N_size * M_repeat * self.data.elemment_size

        mem_counter.dram_write["output"] += M_size * N_size * self.data.elemment_size
        mem_counter.sram_read["output"] += M_size * N_size * self.data.elemment_size

        return mem_counter


    def run_attn_dense(self, layer_config):
        is_causal = True
        num_tokens = layer_config['seq_len']
        dim_per_head = layer_config['dim_per_head']
        num_heads = layer_config['num_heads']

        qk_config = {
            "seq_len": num_tokens,
            "in_features": dim_per_head,
            "out_features": num_tokens,
        }

        mem_counter_qk = self.run_linear_dense(qk_config)
        mem_counter_qk *= num_heads

        sv_config = {
            "seq_len": num_tokens,
            "in_features": num_tokens,
            "out_features": dim_per_head,
        }
        mem_counter_sv = self.run_linear_dense(sv_config)
        mem_counter_sv *= num_heads
        if is_causal:
            mem_counter_qk *= 0.5
            mem_counter_sv *= 0.5

        return mem_counter_qk + mem_counter_sv

    def run_linear_adaptiv(self, layer_config, sparsity):
        '''output stationary'''
        reduced_seq_len = layer_config['seq_len'] * (1 - sparsity)

        mem_counter = MemCounter(self.data.data_type)

        # TME stage
        mem_counter.dram_read['input'] += layer_config['seq_len'] * layer_config['in_features'] * self.data.elemment_size
        mem_counter.sram_write['input'] += layer_config['seq_len'] * layer_config['in_features'] * self.data.elemment_size
        mem_counter.sram_read['input'] += 2 * layer_config['seq_len'] * layer_config['in_features'] * self.data.elemment_size
        
        M_size = reduced_seq_len
        N_size = layer_config['out_features']
        K_size = layer_config['in_features']

        m_tile_size = self.accelerator.buffer_config['m_out_size']
        n_tile_size = self.accelerator.buffer_config['n_size']
        k_tile_size = self.accelerator.buffer_config['k_size']

        n_array_size = self.accelerator.array_config['MAC_line_num']
        m_array_size = self.accelerator.array_config['MAC_line_width']

        n_repeat_num = (N_size + n_tile_size - 1) // n_tile_size
        m_repeat_num = (M_size + m_tile_size - 1) // m_tile_size
        k_repeat_num = (K_size + k_tile_size - 1) // k_tile_size

        num_cycles = K_size * n_repeat_num * m_repeat_num

        mem_counter.sram_read['input'] += num_cycles * m_array_size * n_array_size * self.data.elemment_size
        mem_counter.sram_read['wgt'] += num_cycles * m_array_size * n_array_size * self.data.elemment_size
        mem_counter.sram_write['output'] += N_size * M_size * k_repeat_num * self.data.elemment_size

        mem_counter.dram_read['input'] += M_size * K_size * n_repeat_num * self.data.elemment_size
        mem_counter.sram_write['input'] += M_size * K_size * n_repeat_num * self.data.elemment_size

        mem_counter.dram_read['wgt'] += K_size * N_size * m_repeat_num * self.data.elemment_size
        mem_counter.sram_write['wgt'] += K_size * N_size * m_repeat_num * self.data.elemment_size

        mem_counter.dram_write['output'] += N_size * M_size * self.data.elemment_size
        mem_counter.sram_read['output'] += N_size * M_size * self.data.elemment_size

        return mem_counter
    
    def run_attn_adaptiv(self, layer_config, sparsity):
        qk_config = {
            "seq_len": layer_config['seq_len'],
            "in_features": layer_config['dim_per_head'],
            "out_features": layer_config['seq_len'],
        }
        mem_counter_qk = self.run_linear_adaptiv(qk_config, sparsity)   
        mem_counter_qk *= layer_config['num_heads']

        kv_config = {
            "seq_len": layer_config['seq_len'],
            "in_features": layer_config['seq_len'],
            "out_features": layer_config['dim_per_head'],
        }
        mem_counter_kv = self.run_linear_adaptiv(kv_config, sparsity)
        mem_counter_kv *= layer_config['num_heads']

        is_causal = True
        if is_causal:
            mem_counter_qk *= 0.5
            mem_counter_kv *= 0.5

        return mem_counter_qk + mem_counter_kv

    def run_linear_cmc(self, layer_config, linear_sparsity, num_frames, num_patches, is_kv=False):
        '''half weight stationary'''

        mem_counter = MemCounter(self.data.data_type)

        assert num_frames * num_patches == layer_config['seq_len']
        M_size = layer_config['seq_len'] * (1 - linear_sparsity)
        N_size = layer_config['out_features']
        K_size = layer_config['in_features']

        array_height = self.accelerator.systolic_config["array_height"]
        array_width = self.accelerator.systolic_config["array_width"]

        if not is_kv:
            mem_counter.sram_read['input'] += num_frames * num_patches * num_patches * layer_config['in_features'] * self.data.elemment_size
            mem_counter.dram_read['input'] += num_frames * num_patches * layer_config['in_features'] * self.data.elemment_size
            mem_counter.sram_write['input'] += num_frames * num_patches * layer_config['in_features'] * self.data.elemment_size

        n_repeat = (N_size + self.accelerator.buffer_config['n_size'] - 1) // self.accelerator.buffer_config['n_size']
        m_repeat = (M_size + self.accelerator.buffer_config['m_in_size'] - 1) // self.accelerator.buffer_config['m_in_size']
        k_repeat = (K_size + self.accelerator.buffer_config['k_size'] - 1) // self.accelerator.buffer_config['k_size']
        
        num_cycles = (M_size * K_size * N_size + array_height * array_width - 1) // (array_height * array_width)

        mem_counter.sram_read['input'] += num_cycles * array_height * self.data.elemment_size
        mem_counter.sram_read['wgt'] += N_size * K_size * self.data.elemment_size * m_repeat
        mem_counter.sram_write['output'] += num_cycles * array_width * self.data.elemment_size

        mem_counter.dram_read['input'] += M_size * K_size * n_repeat * self.data.elemment_size
        mem_counter.sram_write['input'] += M_size * K_size * n_repeat * self.data.elemment_size
        mem_counter.dram_read['wgt'] += K_size * N_size * m_repeat * self.data.elemment_size
        mem_counter.sram_write['wgt'] += K_size * N_size * m_repeat * self.data.elemment_size
        mem_counter.dram_write['output'] += N_size * M_size * self.data.elemment_size
        mem_counter.sram_read['output'] += N_size * M_size * self.data.elemment_size

        return mem_counter

    def run_attn_cmc(self, layer_config, q_sparsity, s_sparsity, num_frames, num_patches):
        '''half weight stationary'''

        qk_config = {
            'seq_len': layer_config['seq_len'],
            'in_features': layer_config['dim_per_head'],
            'out_features': layer_config['seq_len'],
        }
        mem_counter_qk = self.run_linear_cmc(qk_config, q_sparsity, num_frames, num_patches, is_kv=False)
        mem_counter_qk *= layer_config['num_heads']

        kv_config = {
            'seq_len': layer_config['seq_len'],
            'in_features': layer_config['seq_len'],
            'out_features': layer_config['dim_per_head'],
        }
        mem_counter_kv = self.run_linear_cmc(kv_config, s_sparsity, num_frames, num_patches, is_kv=False)
        mem_counter_kv *= layer_config['num_heads']

        is_causal = True
        if is_causal:
            mem_counter_qk *= 0.5
            mem_counter_kv *= 0.5

        return mem_counter_qk + mem_counter_kv