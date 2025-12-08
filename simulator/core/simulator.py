from models.sparse_info import SparseInfo
from arch.accelerator import Accelerator
from models.models import ModelConfig
from .simulator_comp import SimulatorComp
from .simulator_mem import SimulatorMem, MemCounter, FocusData

from memory.cacti import get_buffer_area_power_energy

from utils.utils import split_into_chunks, set_csv_column

import torch
import pandas as pd

class ActivationCounter:
    def __init__(self):
        self.size_dict = {}

    def add(self, name, size):
        if name not in self.size_dict:
            self.size_dict[name] = 0
        self.size_dict[name] += size

    def __mul__(self, other):
        for name in self.size_dict:
            self.size_dict[name] *= other
        return self

    def __repr__(self):
        # return sum of all values
        return str(int(sum(self.size_dict.values())))

class Simulator:
    """
    Simulator class
    """

    def __init__(self, model_config, accelerator, sparse_info):
        self.model_config = model_config
        self.accelerator = accelerator
        self.sim_compute = SimulatorComp(accelerator)
        self.sim_memory = SimulatorMem(accelerator)
        self.sparse_info = sparse_info

        self.result_ready = False
        self.result_dict = {
            "model": self.model_config.model,
            "dataset": self.model_config.dataset,
            "accelerator": self.accelerator.type,
            "execution_time": 0,
            "total_cycles": 0,
            "total_compute_cycles": 0,
            "total_stall_cycles": 0,
            "total_energy": 0,
            "total_dram_access": 0,
            "dram_bandwidth": 0,
            "dense_ops": 0,
            "num_ops": 0,
        }

    def get_result(self):
        if not self.result_ready:
            raise ValueError("Result is not ready")
        return self.result_dict

    def get_layer_wise_energy(self, result_dict):
        if not self.accelerator.buffer_evaluated:
            self.accelerator.evaluate_buffer()
        assert self.accelerator.buffer_evaluated, "Buffer evaluation not done"
        
        dram_energy = result_dict['dram_access'] * self.accelerator.dram_config['energy_per_byte']
        sram_energy = 0
        for name in result_dict['mem_counter_layer'].sram_read:
            num_bytes_read = result_dict['mem_counter_layer'].sram_read[name]
            num_bytes_write = result_dict['mem_counter_layer'].sram_write[name]
            if num_bytes_read + num_bytes_write > 0:
                assert name in self.accelerator.buffer_dict, f"Buffer {name} not found in buffer_dict"  
                sram_energy += num_bytes_read * self.accelerator.buffer_dict[name]['read_energy_per_byte'] + num_bytes_write * self.accelerator.buffer_dict[name]['write_energy_per_byte']

        for name in self.accelerator.buffer_dict:
            if name not in result_dict['mem_counter_layer'].sram_read and name not in result_dict['mem_counter_layer'].sram_write:
                sram_energy += self.accelerator.buffer_dict[name]['leak_power'] * result_dict['layer_execution_time']
        
        core_energy = result_dict['layer_execution_time'] * self.accelerator.core_power

        result_dict['dram_energy'] = dram_energy
        result_dict['sram_energy'] = sram_energy
        result_dict['core_energy'] = core_energy
        result_dict['total_energy'] = dram_energy + sram_energy + core_energy

        return result_dict
        

    def get_energy_breakdown(self):
        if not self.accelerator.buffer_evaluated:
            self.accelerator.evaluate_buffer_with_compiler()
        assert self.accelerator.buffer_evaluated, "Buffer evaluation not done"
        assert self.result_ready, "Result not ready"

        # unit mJ

        dram_access_total = 0
        for name in self.result_dict["mem_counter_total"].dram_read:
            dram_access_total += self.result_dict["mem_counter_total"].dram_read[name] + self.result_dict["mem_counter_total"].dram_write[name]
        dram_energy = dram_access_total * self.accelerator.dram_config['energy_per_byte']
    
        sram_energy = 0
        for name in self.result_dict["mem_counter_total"].sram_read:
            num_bytes_read = self.result_dict["mem_counter_total"].sram_read[name]
            num_bytes_write = self.result_dict["mem_counter_total"].sram_write[name]

            if num_bytes_read + num_bytes_write > 0:
                assert name in self.accelerator.buffer_dict, f"Buffer {name} not found in buffer_dict"
                sram_energy += num_bytes_read * self.accelerator.buffer_dict[name]['read_energy_per_byte'] + num_bytes_write * self.accelerator.buffer_dict[name]['write_energy_per_byte']

        for name in self.accelerator.buffer_dict:
            sram_energy += self.accelerator.buffer_dict[name]['leak_power'] * self.result_dict["execution_time"]

        core_energy = self.accelerator.core_power * self.result_dict["execution_time"]

        self.result_dict["dram_energy"] = dram_energy
        self.result_dict["sram_energy"] = sram_energy
        self.result_dict["core_energy"] = core_energy
        self.result_dict["total_energy"] = dram_energy + sram_energy + core_energy

        print(f"dram energy: {dram_energy} mJ")
        print(f"sram energy: {sram_energy} mJ")
        print(f"core energy: {core_energy} mJ")
        print(f"total energy: {self.result_dict['total_energy']} mJ")

        return self.result_dict

    def run(self):
        print(f"Running simulator for {self.accelerator.type}...")
        if self.accelerator.type == "focus":
            self.run_focus(self.sparse_info)
        elif self.accelerator.type == "dense":
            self.run_dense()
        elif self.accelerator.type == "adaptiv":
            self.run_adaptiv(self.sparse_info)
        elif self.accelerator.type == "cmc":
            self.run_cmc(self.sparse_info)
        else:
            raise ValueError(f"Unsupported accelerator type: {self.accelerator.type}")

    def run_layer_wise_focus(self, sparse_info, layer_type, block_idx, tile_size):

        layers = self.model_config.layers
        num_blocks = self.model_config.num_blocks

        mask_zero_dict = sparse_info.info_dict['mask_zero']
        mask_similar_dict = sparse_info.info_dict['mask_similar']
        group_idx_dict = sparse_info.info_dict['group_idx']

        name_map = {
            'k_proj': 'q_proj',
            'v_proj': 'q_proj',
            'up_proj': 'gate_proj',
            'attn': 'query',
        }

        result_dict_list = []

        for i in range(block_idx, block_idx + 1):
            for layer_name, layer_config in layers.items():
                if layer_name != layer_type:
                    continue
                if layer_name not in mask_zero_dict:
                    mapped_name = name_map[layer_name]
                else:
                    mapped_name = layer_name
                mask_zero = mask_zero_dict[mapped_name][i]
                mask_similar = mask_similar_dict[mapped_name][i]
                group_idx = group_idx_dict[mapped_name][i]

                mem_counter_layer = MemCounter(FocusData.data_type)

                if layer_name == "attn":
                    compute_cycles, num_ops, dense_ops = self.sim_compute.focus(mask_zero, mask_similar, group_idx, layer_config)
                    scatter_cycles = self.sim_compute.run_qk_scatter_focus(mask_zero, mask_similar, group_idx, layer_config)
                    compute_stall_cycles = max(0, scatter_cycles - compute_cycles)
                    mem_counter_layer = self.sim_memory.focus(mask_zero, mask_similar, group_idx, layer_config)
                    mem_counter_layer = self.sim_memory.run_detect_attn_focus(mask_zero, mask_similar, group_idx, layer_config)
                    scatter_ops = 0
                else:
                    compute_cycles, num_ops, dense_ops = self.sim_compute.run_linear_focus(mask_zero, mask_similar, group_idx, layer_config)
                    scatter_cycles = self.sim_compute.run_linear_scatter_focus(mask_zero, mask_similar, group_idx, layer_config)
                    compute_stall_cycles = max(0, scatter_cycles - compute_cycles)
                    mem_counter_layer = self.sim_memory.run_linear_focus(mask_zero, mask_similar, group_idx, layer_config)
                    mem_counter_layer = self.sim_memory.run_detect_linear_focus(mask_zero, mask_similar, group_idx, layer_config)
                    scatter_ops = self.sim_compute.get_scatted_ops(mask_zero, mask_similar, group_idx, layer_config, tile_size)

                dram_access_layer = 0
                for name in mem_counter_layer.dram_read:
                    dram_access_layer += mem_counter_layer.dram_read[name] + mem_counter_layer.dram_write[name]

                num_ops *= 2 # MAC counted as two ops

                result_dict = {
                    "block_idx": i,
                    "layer_name": layer_name,
                    "layer_cycles": compute_cycles + compute_stall_cycles,
                    "stall_cycles": compute_stall_cycles,
                    "compute_cycles": compute_cycles,
                    "layer_execution_time": (compute_cycles + compute_stall_cycles) / self.accelerator.frequency,
                    "mem_counter_layer": mem_counter_layer,
                    "layer_array_ops": num_ops,
                    "layer_scatter_ops": scatter_ops,
                    "layer_dense_ops": dense_ops,
                    "dram_access": dram_access_layer,
                }

                result_dict = self.get_layer_wise_energy(result_dict)
                result_dict_list.append(result_dict)

        return result_dict_list

    def run_focus(self, sparse_info):

        layers = self.model_config.layers
        num_blocks = self.model_config.num_blocks
        # num_blocks = 1 # For testing purposes, we can set it to 1

        mask_zero_dict = sparse_info.info_dict['mask_zero']
        mask_similar_dict = sparse_info.info_dict['mask_similar']
        group_idx_dict = sparse_info.info_dict['group_idx']

        total_compute_cycles = 0
        total_cycles = 0
        total_stall_cycles = 0
        total_num_ops = 0
        total_dense_ops = 0

        name_map = {
            'k_proj': 'q_proj',
            'v_proj': 'q_proj',
            'up_proj': 'gate_proj',
            'attn': 'query',
        }

        mem_counter_total = MemCounter(FocusData.data_type)

        activation_counter = ActivationCounter()

        for i in range(num_blocks):
            for layer_name, layer_config in layers.items():
                if layer_name not in mask_zero_dict:
                    mapped_name = name_map[layer_name]
                else:
                    mapped_name = layer_name
                mask_zero = mask_zero_dict[mapped_name][i]
                mask_similar = mask_similar_dict[mapped_name][i]
                group_idx = group_idx_dict[mapped_name][i]
                if layer_name == "attn":
                    # compute_cycles = 0
                    compute_cycles, num_ops, dense_ops = self.sim_compute.run_attn_focus(mask_zero, mask_similar, group_idx, layer_config)
                    scatter_cycles = self.sim_compute.run_qk_scatter_focus(mask_zero, mask_similar, group_idx, layer_config)
                    compute_stall_cycles = max(0, scatter_cycles - compute_cycles)
                    mem_counter_total += self.sim_memory.run_attn_focus(mask_zero, mask_similar, group_idx, layer_config)
                    mem_counter_total += self.sim_memory.run_detect_attn_focus(mask_zero, mask_similar, group_idx, layer_config)

                    num_cluster = int((mask_zero.numel() - (torch.sum(mask_zero) + torch.sum(mask_similar))) / (mask_zero.shape[-1] * mask_zero.shape[0]))
                    num_tokens = int(mask_zero.shape[1] - torch.sum(torch.all(mask_zero, dim=(0,-1)), dim=-1))
                    activation_size_0 = num_cluster * layer_config['dim_per_head'] * layer_config['num_heads'] * FocusData.elemment_size
                    activation_size_1 = num_tokens * num_tokens * layer_config['num_heads'] * FocusData.elemment_size
                    activation_counter.add(layer_name, activation_size_0 + activation_size_1)
                    
                else:        
                    # compute_cycles = 0
                    # print(f"Processing linear layer: {layer_name}, compute cycles: {compute_cycles}")
                    compute_cycles, num_ops, dense_ops = self.sim_compute.run_linear_focus(mask_zero, mask_similar, group_idx, layer_config)
                    scatter_cycles = self.sim_compute.run_linear_scatter_focus(mask_zero, mask_similar, group_idx, layer_config)
                    gather_cycles, gather_cycles_per_tile = self.sim_compute.run_gather_linear_focus(mask_zero, mask_similar, group_idx, layer_config)
                    gather_or_scatter_cycles = max(scatter_cycles, gather_cycles)
                    compute_stall_cycles = max(0, gather_or_scatter_cycles - compute_cycles) + gather_cycles_per_tile

                    mem_counter_total += self.sim_memory.run_linear_focus(mask_zero, mask_similar, group_idx, layer_config)
                    mem_counter_total += self.sim_memory.run_detect_linear_focus(mask_zero, mask_similar, group_idx, layer_config)

                    num_cluster = int((mask_zero.numel() - (torch.sum(mask_zero) + torch.sum(mask_similar))) / mask_zero.shape[-1])
                    activation_counter.add(layer_name, num_cluster * layer_config['in_features'] * FocusData.elemment_size)
                
                total_num_ops += num_ops
                total_dense_ops += dense_ops
                total_stall_cycles += compute_stall_cycles
                total_cycles += compute_cycles + compute_stall_cycles
                total_compute_cycles += compute_cycles

                # dram_access_this_layer = 0
                # for name in mem_counter_total.dram_read:
                #     dram_access_this_layer += mem_counter_total.dram_read[name] + mem_counter_total.dram_write[name]
                # print(f"dram access for {layer_name}: {dram_access_this_layer} bytes")

        # sum up dram read and write
        dram_access_total = 0
        for name in mem_counter_total.dram_read:
            dram_access_total += mem_counter_total.dram_read[name] + mem_counter_total.dram_write[name]


        self.result_dict["mem_counter_total"] = mem_counter_total
        self.result_dict["execution_time"] = total_cycles / self.accelerator.frequency
        self.result_dict["total_cycles"] = total_cycles
        self.result_dict["total_compute_cycles"] = total_compute_cycles
        self.result_dict["total_stall_cycles"] = total_stall_cycles
        self.result_dict["total_dram_access"] = dram_access_total
        # Avoid division by zero
        self.result_dict["dram_bandwidth"] = dram_access_total / max(1, total_compute_cycles) if total_compute_cycles > 0 else 0
        self.result_dict["num_ops"] = total_num_ops
        self.result_dict["dense_ops"] = total_dense_ops
        self.result_dict["total_activation"] = activation_counter
        self.result_ready = True

        print(f"Execution time on focus: {self.result_dict['execution_time']} seconds")
        print(f"Total cycles: {self.result_dict['total_cycles']}")
        print(f"Total compute cycles: {self.result_dict['total_compute_cycles']}")
        print(f"Total stall cycles: {self.result_dict['total_stall_cycles']}")
        print(f"Total DRAM access: {self.result_dict['total_dram_access']} bytes")
        print("dram bandwidth: ", self.result_dict['dram_bandwidth'], "bytes/cycle")
        print(f"Total num ops: {self.result_dict['num_ops']}")
        print(f"Total dense ops: {self.result_dict['dense_ops']}")

        compression_ratio_tensor = torch.tensor(self.sim_compute.compression_ratio_list)
        compression_ratio = torch.sum(compression_ratio_tensor) / compression_ratio_tensor.numel()
        self.result_dict["total_compression_ratio"] = compression_ratio

        print("total compression ratio", self.result_dict["total_compression_ratio"])

    def run_adaptiv(self, sparse_info):

        sparsity = sparse_info.info_dict['sparsity']

        layers = self.model_config.layers
        num_blocks = self.model_config.num_blocks

        total_compute_cycles = 0
        mem_counter_total = MemCounter(FocusData.data_type)

        activation_counter = ActivationCounter()

        total_cycles = 0
        total_num_ops = 0
        total_dense_ops = 0

        for layer_name, layer_config in layers.items():
            if layer_name == 'attn':
                compute_cycles, num_ops, dense_ops = self.sim_compute.run_attn_adaptiv(layer_config, sparsity)
                mem_counter = self.sim_memory.run_attn_adaptiv(layer_config, sparsity)
                activation_size_0 = layer_config['seq_len'] * (1 - sparsity) * layer_config['dim_per_head'] * layer_config['num_heads'] * FocusData.elemment_size
                activation_size_1 = layer_config['seq_len'] * layer_config['seq_len'] * (1 - sparsity) * (1 - sparsity) * layer_config['num_heads'] * FocusData.elemment_size
                activation_counter.add(layer_name, activation_size_0 + activation_size_1)

            elif layer_name.endswith('proj'):
                compute_cycles, num_ops, dense_ops = self.sim_compute.run_linear_adaptiv(layer_config, sparsity)
                mem_counter = self.sim_memory.run_linear_adaptiv(layer_config, sparsity)
                activation_counter.add(layer_name, layer_config['in_features'] * layer_config['seq_len'] * (1 - sparsity) * FocusData.elemment_size)
            else:
                raise ValueError(f"Unsupported layer type: {layer_name}")

            total_compute_cycles += compute_cycles
            mem_counter_total += mem_counter
            total_cycles += compute_cycles
            total_num_ops += num_ops
            total_dense_ops += dense_ops

            # dram_access_this_layer = 0
            # for name in mem_counter.dram_read:
            #     dram_access_this_layer += mem_counter.dram_read[name] + mem_counter.dram_write[name]
            # print(f"dram access for {layer_name}: {dram_access_this_layer} bytes")

        total_compute_cycles *= num_blocks
        mem_counter_total *= num_blocks
        activation_counter *= num_blocks
        total_num_ops *= num_blocks
        total_dense_ops *= num_blocks

        dram_access_total = 0
        for name in mem_counter_total.dram_read:
            dram_access_total += mem_counter_total.dram_read[name] + mem_counter_total.dram_write[name]

        execution_time = total_compute_cycles / self.accelerator.frequency
        self.result_dict["mem_counter_total"] = mem_counter_total
        self.result_dict["execution_time"] = execution_time
        self.result_dict["total_cycles"] = total_compute_cycles
        self.result_dict["total_compute_cycles"] = total_compute_cycles
        self.result_dict["total_dram_access"] = dram_access_total
        self.result_dict["dram_bandwidth"] = dram_access_total / total_compute_cycles
        self.result_dict["num_ops"] = total_num_ops
        self.result_dict["dense_ops"] = total_dense_ops
        self.result_dict["total_activation"] = activation_counter
        self.result_ready = True

        print(f"Execution time for adaptiv simulation: {execution_time} seconds")
        print(f"Total compute cycles for dense simulation: {total_compute_cycles}")
        print(f"Total DRAM access for dense simulation: {dram_access_total} bytes")
        print("dram bandwidth: ", dram_access_total / total_compute_cycles, "bytes/cycle")


    
    def run_dense(self):

        layers = self.model_config.layers
        num_blocks = self.model_config.num_blocks
        # num_blocks = 1

        total_compute_cycles = 0
        mem_counter_total = MemCounter(FocusData.data_type)

        activation_counter = ActivationCounter()

        for layer_name, layer_config in layers.items():
            if layer_name == "attn":
                compute_cycles = self.sim_compute.run_attn_dense(layer_config)
                mem_counter = self.sim_memory.run_attn_dense(layer_config)
                mem_counter_total += mem_counter
                activation_size_0 = layer_config['seq_len'] * layer_config['dim_per_head'] * layer_config['num_heads'] * FocusData.elemment_size
                activation_size_1 = layer_config['seq_len'] * layer_config['seq_len'] * layer_config['num_heads'] * FocusData.elemment_size
                activation_counter.add(layer_name, activation_size_0 + activation_size_1)

            elif layer_name.endswith("proj"):
                compute_cycles = self.sim_compute.run_linear_dense(layer_config)
                mem_counter = self.sim_memory.run_linear_dense(layer_config)
                mem_counter_total += mem_counter
                activation_counter.add(layer_name, layer_config['in_features'] * layer_config['seq_len'] * FocusData.elemment_size)
            else:
                raise ValueError(f"Unsupported layer type: {layer_name}")

            # dram_access_this_layer = 0
            # for name in mem_counter.dram_read:
            #     dram_access_this_layer += mem_counter.dram_read[name] + mem_counter.dram_write[name]
            # print(f"dram access for {layer_name}: {dram_access_this_layer} bytes")
            
            total_compute_cycles += compute_cycles


        total_compute_cycles *= num_blocks
        mem_counter_total *= num_blocks
        activation_counter *= num_blocks

        dram_access_total = 0
        for name in mem_counter_total.dram_read:
            dram_access_total += mem_counter_total.dram_read[name] + mem_counter_total.dram_write[name]

        self.result_dict["mem_counter_total"] = mem_counter_total
        self.result_dict["execution_time"] = total_compute_cycles / self.accelerator.frequency
        self.result_dict["total_cycles"] = total_compute_cycles
        self.result_dict["total_compute_cycles"] = total_compute_cycles
        self.result_dict["total_dram_access"] = dram_access_total
        self.result_dict["dram_bandwidth"] = dram_access_total / total_compute_cycles
        self.result_dict["total_activation"] = activation_counter
        self.result_ready = True

        print(f"Execution time for dense simulation: {self.result_dict['execution_time']} seconds")
        print(f"Total compute cycles for dense simulation: {self.result_dict['total_compute_cycles']}")
        print(f"Total DRAM access for dense simulation: {self.result_dict['total_dram_access']} bytes")
        print("dram bandwidth: ", self.result_dict['dram_bandwidth'], "bytes/cycle")

    def run_cmc(self, sparse_info):

        linear_sparsity = sparse_info.info_dict['linear_sparsity']
        q_sparsity = sparse_info.info_dict['q_sparsity']
        s_sparsity = sparse_info.info_dict['s_sparsity']

        layers = self.model_config.layers
        num_blocks = self.model_config.num_blocks

        num_frames = self.model_config.num_frames
        num_patches = self.model_config.num_patches

        total_compute_cycles = 0
        total_num_ops = 0
        total_dense_ops = 0
        mem_counter_total = MemCounter(FocusData.data_type)

        activation_counter = ActivationCounter()

        for layer_name, layer_config in layers.items():
            if layer_name == "attn":
                compute_cycles, num_ops, dense_ops = self.sim_compute.run_attn_cmc(layer_config, q_sparsity, s_sparsity, num_frames, num_patches)
                mem_counter = self.sim_memory.run_attn_cmc(layer_config, q_sparsity, s_sparsity, num_frames, num_patches)
                mem_counter_total += mem_counter
                total_compute_cycles += compute_cycles
                activation_size_0 = layer_config['seq_len'] * layer_config['dim_per_head'] * layer_config['num_heads'] * (1 - q_sparsity) * FocusData.elemment_size
                activation_size_1 = layer_config['seq_len'] * layer_config['seq_len'] * layer_config['num_heads'] * (1 - s_sparsity) * FocusData.elemment_size
                activation_counter.add(layer_name, activation_size_0 + activation_size_1)
            elif layer_name.endswith("proj"):
                if layer_name == "k_proj" or layer_name == "v_proj":
                    is_kv = True
                else:
                    is_kv = False
                compute_cycles, num_ops, dense_ops = self.sim_compute.run_linear_cmc(layer_config, linear_sparsity, num_frames, num_patches, is_kv)
                mem_counter = self.sim_memory.run_linear_cmc(layer_config, linear_sparsity, num_frames, num_patches, is_kv)
                mem_counter_total += mem_counter
                total_compute_cycles += compute_cycles
                activation_counter.add(layer_name, layer_config['in_features'] * layer_config['seq_len'] * (1 - linear_sparsity) * FocusData.elemment_size)
            else:
                raise ValueError(f"Unsupported layer type: {layer_name}")

            total_num_ops += num_ops
            total_dense_ops += dense_ops
                
        total_compute_cycles *= num_blocks
        mem_counter_total *= num_blocks
        activation_counter *= num_blocks
        total_num_ops *= num_blocks
        total_dense_ops *= num_blocks

        dram_access_total = 0
        for name in mem_counter_total.dram_read:
            dram_access_total += mem_counter_total.dram_read[name] + mem_counter_total.dram_write[name]

        self.result_dict["mem_counter_total"] = mem_counter_total
        self.result_dict["execution_time"] = total_compute_cycles / self.accelerator.frequency
        self.result_dict["total_cycles"] = total_compute_cycles
        self.result_dict["total_compute_cycles"] = total_compute_cycles
        self.result_dict["total_dram_access"] = dram_access_total
        self.result_dict["dram_bandwidth"] = dram_access_total / total_compute_cycles
        self.result_dict["num_ops"] = total_num_ops
        self.result_dict["dense_ops"] = total_dense_ops
        self.result_dict["total_activation"] = activation_counter
        self.result_ready = True

        print(f"Execution time for CMC simulation: {self.result_dict['execution_time']} seconds")
        print(f"Total compute cycles for CMC simulation: {self.result_dict['total_compute_cycles']}")
        print(f"Total DRAM access for CMC simulation: {self.result_dict['total_dram_access']} bytes")
        print("dram bandwidth: ", self.result_dict['dram_bandwidth'], "bytes/cycle")
        print(f"Total activation: {activation_counter}")

    def get_detailed_power_area_breakdown(self, output_dir):
        assert self.result_ready, "Result not ready"
        
        detail_dict = {}
        for core_component in self.accelerator.components:
            detail_dict[core_component] = {}
            detail_dict[core_component]['area'] = self.accelerator.components[core_component]['area'] * self.accelerator.components[core_component]['count']
            detail_dict[core_component]['power'] = self.accelerator.components[core_component]['power'] * self.accelerator.components[core_component]['count']
            detail_dict[core_component]['energy'] = self.accelerator.components[core_component]['power'] * self.accelerator.components[core_component]['count'] * self.result_dict["execution_time"]

        for buffer_name in self.accelerator.buffer_dict:
            detail_dict[buffer_name] = {}
            detail_dict[buffer_name]['area'] = self.accelerator.buffer_dict[buffer_name]['area']
            static_energy = self.accelerator.buffer_dict[buffer_name]['leak_power'] * self.result_dict["execution_time"]

            num_bytes_read = self.result_dict["mem_counter_total"].sram_read[buffer_name] if buffer_name in self.result_dict["mem_counter_total"].sram_read else 0
            num_bytes_write = self.result_dict["mem_counter_total"].sram_write[buffer_name] if buffer_name in self.result_dict["mem_counter_total"].sram_write else 0

            dynamic_energy = num_bytes_read * self.accelerator.buffer_dict[buffer_name]['read_energy_per_byte'] + num_bytes_write * self.accelerator.buffer_dict[buffer_name]['write_energy_per_byte']

            detail_dict[buffer_name]['energy'] = static_energy + dynamic_energy
            detail_dict[buffer_name]['power'] = static_energy / self.result_dict["execution_time"] + dynamic_energy / self.result_dict["execution_time"]

        detail_dict['dram'] = {}
        detail_dict['dram']['area'] = 0
        detail_dict['dram']['power'] = self.result_dict['dram_energy'] / self.result_dict["execution_time"]
        detail_dict['dram']['energy'] = self.result_dict['dram_energy']

        # save the dict to a csv file using pandas
        data = []
        for component in detail_dict:
            data.append({
                'component': component,
                'area': detail_dict[component]['area'],
                'power': detail_dict[component]['power'],
                'energy': detail_dict[component]['energy']
            })
        
        df = pd.DataFrame(data)
        df.to_csv(f'{output_dir}/detailed_power_area_breakdown.csv', index=False)

        return detail_dict


if __name__ == "__main__":
    model = "llava_vid"
    dataset = "videomme"
    
    model_config = ModelConfig(model, dataset)

    sparse_info = SparseInfo("focus", model, dataset, model_config)

    accelerator = Accelerator("focus")
    # accelerator = Accelerator("dense")

    # accelerator = Accelerator("adaptiv")
    # accelerator = Accelerator("cmc")

    simulator = Simulator(model_config, accelerator, sparse_info)
    simulator.run()


    # simulator.run_dense()
    # simulator.run_adaptiv(0.4)
    # simulator.run_cmc(0.4, 0.5, 0.6)