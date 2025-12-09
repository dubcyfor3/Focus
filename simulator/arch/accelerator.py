import torch
import pandas as pd
import os
import sys
# Add parent directory to path so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from memory.cacti import get_buffer_area_power_energy
from memory.buffer import get_buffer_stats_from_config_list


class Accelerator:
    def __init__(self, type: str, SEC_only: bool = False, focus_m_tile_size: int = 1024, num_scatter_vector: int = 2, block_size: int = 8, force_cacti: bool = False):
        self.type = type
        self.frequency = 500 * 1000 * 1000 # 500 MHz
        self.buffer_evaluated = False

        self.dram_config = {
            'config_name': 'DDR4_4Gb_x16_2133',
            'bandwidth': 64 * 1024 * 1024 * 1024, # 64 GB/s
            'energy_per_byte': 99.98 * 1e-9, # 99.98 * 1e-9 mJ/byte from DRAMsim3
        }

        if SEC_only:
            assert type == 'focus', "Semantic Concentration only is only supported for focus"
            self.SEC_only = SEC_only
        else:
            self.SEC_only = False

        self.focus_m_tile_size = focus_m_tile_size
        self.block_size = block_size
        
        if type == 'focus':
            self.set_focus_config(num_scatter_vector)
        elif type == 'dense':
            self.set_dense_config()
        elif type == 'adaptiv':
            self.set_adaptiv_config()
        elif type == 'cmc':
            self.set_cmc_config()
        else:
            raise ValueError(f"Unknown accelerator type: {type}")

        print(f"Configuring Accelerator {self.type}...")
        self.evaluate_buffer(force_cacti=force_cacti)
        self.get_total_area()
        self.get_on_chip_power()

    def evaluate_buffer(self, force_cacti: bool = False):
        if force_cacti:
            print(f"Force using cacti for buffer evaluation")
            self.evaluate_buffer_with_cacti()
        elif self.type == 'focus':
            if self.focus_m_tile_size != 1024 or self.num_scatter_vector != 2:
                print(f"Buffer eval using memory compiler is not supported for Focus with m_tile_size != 1024 or num_scatter_vector != 2, fall back to using cacti")
                self.evaluate_buffer_with_cacti()
            else:
                self.evaluate_buffer_with_compiler()
        else:
            self.evaluate_buffer_with_compiler()


    def set_cmc_config(self):
        m_in_size = 768
        m_out_size = 8
        n_size = 64
        k_size = 8 * 8

        self.buffer_config = {  
            "m_in_size": m_in_size,
            "m_out_size": m_out_size,
            "n_size": n_size,
            "k_size": k_size,
        }

        self.buffer_dict = {
            'input': {'buffer_size': 96 * 1024 * 2, 'block_size': 32 * 2, 'count': 1}, # 192 KB
            'wgt': {'buffer_size': 96 * 1024 * 2, 'block_size': 32 * 2, 'count': 1}, # 192 KB
            'output': {'buffer_size': 12 * 1024 * 2, 'block_size': 16 * 2, 'count': 1}, # 24 KB
            'qs_buffer': {'buffer_size': 75 * 1024 * 2, 'block_size': 32 * 2, 'count': 1}, # 150 KB
            'kv_buffer': {'buffer_size': 24.5 * 1024 * 2, 'block_size': 32 * 2, 'count': 1}, # 49 KB
            'frame_buffer': {'buffer_size': 150 * 1024 * 2, 'block_size': 32 * 2, 'count': 1}, # 300 KB
        }

        self.buffer_model_dict = {
            'input': [('1024x128', 6, 2)],
            'wgt': [('1024x128', 6, 2)],
            'output': [('512x16', 25, 1)],
            'qs_buffer': [('1024x128', 10, 1)],
            'kv_buffer': [('512x16', 50, 1)],
            'frame_buffer': [('1024x128', 20, 1)],
        }

        self.systolic_config = {
            "array_height": 32,
            "array_width": 32,
        }

        self.codec_config = {
            'PE_width': 256,
            'PE_height': 4,
            'frames_per_partition': 8,
        }

        self.components = {
            'Systolic Array': {'count': 1},
            'Codec PE': {'count': 16},
            'Codec Adder Tree': {'count': 4},
            'FP16 Exp (SFU)': {'count': 32 * 3 * 2},
            'FP16 Sqrt (SFU)': {'count': 1 * 3 * 2},
            'FP16 Reciprocal (SFU)': {'count': 1 * 3 * 2},
            'FP16 Mul (SFU)': {'count': 32 * 3 * 2},
            'FP16 Add (SFU)': {'count': 32 * 3 * 2},
        }
        self.rtl_csv_path = os.path.join(os.path.dirname(__file__), 'cmc_rtl.csv')
        self.get_components_area_power()

    def set_adaptiv_config(self):

        m_in_size = 256
        m_out_size = 256
        n_size = 256
        k_size = 256

        MAC_line_width = 64
        MAC_line_num = 16

        self.buffer_config = {
            "m_out_size": m_out_size,
            "m_in_size": m_in_size,
            "n_size": n_size,
            "k_size": k_size,
        }

        self.buffer_dict = {
            'input': {'buffer_size': m_in_size * k_size * 2 * 2, 'block_size': 32 * 2, 'count': 1}, # 256 KB
            'wgt': {'buffer_size': k_size * n_size * 2 * 2, 'block_size': 32 * 2, 'count': 1}, # 256 KB
            'output': {'buffer_size': m_in_size * n_size * 2 * 2, 'block_size': 32 * 2, 'count': 1}, # 256 KB
        }


        self.buffer_model_dict = {
            'input': [('512x128', 16, 2)],
            'wgt': [('512x128', 16, 2)],
            'output': [('512x128', 16, 2)],
        }

        self.array_config = {
            "MAC_line_width": MAC_line_width,
            "MAC_line_num": MAC_line_num,
        }

        self.systolic_config = {
            "array_height": MAC_line_width,
            "array_width": MAC_line_num,
            "dataflow": "ws",
        }

        self.components = {
            'MAC Array': {'count': 1},
            'AdapTME': {'count': 1},
            'FP16 Exp (SFU)': {'count': 32 * 3 * 2},
            'FP16 Sqrt (SFU)': {'count': 1 * 3 * 2},
            'FP16 Reciprocal (SFU)': {'count': 1 * 3 * 2},
            'FP16 Mul (SFU)': {'count': 32 * 3 * 2},
            'FP16 Add (SFU)': {'count': 32 * 3 * 2},
        }
        self.rtl_csv_path = os.path.join(os.path.dirname(__file__), 'adaptiv_rtl.csv')
        self.get_components_area_power()


    def set_dense_config(self):

        # SA use the same configs as Focus
        self.set_focus_config()

        # remove the (SIC) or (SEC) components only exist in Focus, not in SA
        components_to_remove = [component for component in self.components 
                                 if component.endswith('(SIC)') or component.endswith('(SEC)')]
        for component in components_to_remove:
            del self.components[component]


    def set_focus_config(self, num_scatter_vector: int = 2):

        if self.focus_m_tile_size == -1:
            self.focus_m_tile_size = 11648
        m_in_size = self.focus_m_tile_size
        m_out_size = self.focus_m_tile_size
        n_size = 96
        k_size = 32

        assert m_in_size == m_out_size, "m_in_size and m_out_size must be the same"

        array_height = 32 # K dim size
        array_width = 32 # N dim size
        self.num_scatter_vector = num_scatter_vector

        self.buffer_config = {
            "m_in_size": m_in_size,
            "m_out_size": m_out_size,
            "n_size": n_size,
            "k_size": k_size,
        }

        self.buffer_dict = {
            'input': {'buffer_size': m_in_size * k_size * 2 * 2, 'block_size': array_height * 2, 'count': 1},
            'wgt': {'buffer_size': k_size * n_size * 2 * 2, 'block_size': 2, 'count': 1},
            'concentrate_out': {'buffer_size': m_in_size * array_width * 2 * 2, 'block_size': array_width * 2, 'count': 1, 'extra_read_port': self.num_scatter_vector - 1},
            'output': {'buffer_size': m_out_size * n_size * 2 * 2, 'block_size': array_width * 2 * self.num_scatter_vector, 'count': 1},
            'layouter': {'buffer_size': 32 * 32 * 2, 'block_size': 32 * 2, 'count': 8, 'extra_read_port': 0},
            'similarity_map': {'buffer_size': m_out_size * 2, 'block_size': 2, 'count': 1},
            'similarity_table': {'buffer_size': m_in_size * 32 * 2, 'block_size': 32 * 2, 'count': 1},
        }

        self.buffer_model_dict = {
            'input': [('1024x128', 4, 2)],
            'wgt': [('2048x16', 1, 3), ('1024x16', 1, 2)],
            'concentrate_out': [('1024x128', 4, 2)],
            'output': [('1024x128', 8, 2), ('512x128', 8, 2)],
            'layouter': [('32x256', 2, 8)],
            'similarity_map': [('1024x16', 1, 1)],
            'similarity_table': [('1024x128', 4, 1)],
        }


        self.systolic_config = {
            "array_height": array_height, # K dim size
            "array_width": array_width, # N dim size
            "dataflow": "ws",
        }

        self.components = {
            'Systolic Array': {'count': 1},
            'Cosine Similarity (SIC)': {'count': 1},
            'L2 Norm (SIC)': {'count': 1},
            'Max Unit (SIC)': {'count': 1},
            'Average Update (SIC)': {'count': 1},
            'Accumulator (SIC)': {'count': self.num_scatter_vector},
            'Max Unit (SEC)': {'count': 32},
            'Importance Vector Buffer (SEC)': {'count': 1},
            'FP16 Exp (SFU)': {'count': 32 * 3 * 2},
            'FP16 Sqrt (SFU)': {'count': 1 * 3 * 2},
            'FP16 Reciprocal (SFU)': {'count': 1 * 3 * 2},
            'FP16 Mul (SFU)': {'count': 32 * 3 * 2},
            'FP16 Add (SFU)': {'count': 32 * 3 * 2},
        }
        self.rtl_csv_path = os.path.join(os.path.dirname(__file__), 'focus_rtl.csv')
        self.get_components_area_power()

    def evaluate_buffer_with_compiler(self):
        if self.buffer_evaluated:
            return
        self.buffer_evaluated = True
        total_buffer_area = 0
        total_buffer_power = 0
        total_buffer_capacity = 0
        for name, config in self.buffer_dict.items():
            capacity = config['buffer_size'] * config['count']
            bitwidth = config['block_size']
            model_list = self.buffer_model_dict[name]
            area, peak_power = get_buffer_stats_from_config_list(name, model_list, capacity, bitwidth)

            self.buffer_dict[name]['area'] = area
            self.buffer_dict[name]['leak_power'] = peak_power
            self.buffer_dict[name]['read_energy_per_byte'] = 0
            self.buffer_dict[name]['write_energy_per_byte'] = 0

            total_buffer_area += area
            total_buffer_power += peak_power
            total_buffer_capacity += capacity

        self.total_buffer_area = total_buffer_area
        self.total_buffer_power = total_buffer_power
        self.total_buffer_capacity = total_buffer_capacity // (1024) # unit in KB

        print(f"Total buffer area: {total_buffer_area}")
        print(f"Total buffer power: {total_buffer_power}")
        print(f"Total buffer capacity: {self.total_buffer_capacity} KB")

    def evaluate_buffer_with_cacti(self):
        if self.buffer_evaluated:
            return
        self.buffer_evaluated = True
        total_buffer_area = 0
        total_buffer_capacity = 0
        total_buffer_power = 0
        for name, config in self.buffer_dict.items():
            capacity = config['buffer_size']
            area, leak_power, read_energy_per_byte, write_energy_per_byte = get_buffer_area_power_energy(config)
                
            print(f"{name}: {area}")
            # update config with result
            self.buffer_dict[name]['area'] = area * config['count']
            self.buffer_dict[name]['leak_power'] = leak_power * config['count']
            self.buffer_dict[name]['read_energy_per_byte'] = read_energy_per_byte
            self.buffer_dict[name]['write_energy_per_byte'] = write_energy_per_byte
            total_buffer_area += area * config['count']
            total_buffer_capacity += capacity * config['count']
            total_buffer_power += leak_power * config['count']
        self.total_buffer_area = total_buffer_area
        self.total_buffer_capacity = total_buffer_capacity
        self.total_buffer_power = total_buffer_power

        print(f"Total buffer area: {total_buffer_area}")
        print(f"Total buffer capacity: {total_buffer_capacity // (1024)} KB")
        print(f"Total buffer power: {total_buffer_power}")

    def get_total_area(self):
        total_area = 0
        assert self.buffer_evaluated, "Buffer area not evaluated"
        total_area += self.total_buffer_area
        for component in self.components:
            total_area += self.components[component]['area'] * self.components[component]['count']

        print(f"Total area: {total_area}")
        return total_area

    def get_on_chip_power(self):
        core_power = self.get_core_power()
        buffer_power = self.total_buffer_power
        print(f"On-chip power: {core_power + buffer_power}")
        return core_power + buffer_power

    def get_core_power(self):
        self.core_power = 0
        for component in self.components:
            self.core_power += self.components[component]['power'] * self.components[component]['count']
        print(f"Core power: {self.core_power}")
        return self.core_power
    
    def print_buffer_size_and_io_width(self):
        assert self.type == "focus"
        for name, config in self.buffer_dict.items():
            print(f"{name}: {config['buffer_size']} bytes, {config['block_size']} bytes, {config['count']} count")

    def get_components_area_power(self):
        # read the rtl_csv_path file
        df = pd.read_csv(self.rtl_csv_path)
        for component in self.components:
            component_area = df.loc[df['Module'] == component, 'Area'].values[0] # unit in mm^2
            component_power = df.loc[df['Module'] == component, 'Power'].values[0] # unit in mW
            self.components[component]['area'] = component_area
            self.components[component]['power'] = component_power

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='results')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # create pd dataframe to store the area and power of each accelerator
    df = pd.DataFrame(columns=['Accelerator', 'Area', 'On-chip Power', 'Buffer Size'])
    accelerator = Accelerator('dense')
    df.loc[len(df)] = ['systolic array', accelerator.get_total_area(), accelerator.get_on_chip_power(), accelerator.total_buffer_capacity]
    print("--------------------------------")
    accelerator = Accelerator('adaptiv')
    df.loc[len(df)] = ['adaptiv', accelerator.get_total_area(), accelerator.get_on_chip_power(), accelerator.total_buffer_capacity]
    print("--------------------------------")
    accelerator = Accelerator('cmc')
    df.loc[len(df)] = ['cmc', accelerator.get_total_area(), accelerator.get_on_chip_power(), accelerator.total_buffer_capacity]
    print("--------------------------------")
    accelerator = Accelerator('focus')
    df.loc[len(df)] = ['focus', accelerator.get_total_area(), accelerator.get_on_chip_power(), accelerator.total_buffer_capacity]
    df.to_csv(f'{args.output_dir}/accelerator_area_power_buffer.csv', index=False)

