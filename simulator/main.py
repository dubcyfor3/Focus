import torch
import argparse
import os

from utils import save_result
from core.simulator import Simulator
from models.models import ModelConfig
from models.sparse_info import SparseInfo
from arch.accelerator import Accelerator

def main(args):
    if args.all_models_datasets:
        models = ['llava_vid', 'llava_onevision', 'minicpm_v'] 
        datasets = ['videomme', 'mlvu', 'mvbench']
    elif args.image_models_datasets:
        models = ['llava_onevision', 'qwen2_5_vl']
        datasets = ['vqav2', 'mme', 'mmbench']
    else:
        models = [args.model]
        datasets = [args.dataset]

    for model in models:
        for dataset in datasets:
            print("--------------------------------")
            print(model, dataset)
            model_config = ModelConfig(model, dataset, args.trace_dir)
            sparse_info = SparseInfo(args.accelerator, model, dataset, model_config, args.trace_dir)
            if args.SEC_only:
                focus_m_tile_size = 1024 // 2
            else:
                focus_m_tile_size = 1024
            accelerator = Accelerator(args.accelerator, args.SEC_only, focus_m_tile_size=focus_m_tile_size)
            simulator = Simulator(model_config, accelerator, sparse_info)
            simulator.run()
            result_dict = simulator.get_result()
            result_dict = simulator.get_energy_breakdown()
            
            file_name = 'main_' + args.accelerator if not args.SEC_only else 'main_' + args.accelerator + '_SEC_only'
            save_result(result_dict, f"{args.output_dir}/{file_name}.csv")
            if model == 'llava_vid' and dataset == 'videomme' and args.accelerator == 'focus' and not args.SEC_only:
                simulator.get_detailed_power_area_breakdown(args.output_dir)

def dse_block_size(args):
    model = 'llava_vid'
    dataset = 'videomme'
    trace_dir = os.path.join(args.trace_dir, 'block_size_dse')
    # get all the files in the config_path
    files = os.listdir(trace_dir)
    for file in files:
        print("--------------------------------")
        print(file)
        block_size_pth = file.split('_')[-1]
        block_size = block_size_pth.split('.')[0]
        block_size_int = sum(int(x) for x in block_size.split('x'))
        model_config = ModelConfig(model, dataset, args.trace_dir)
        sparse_info = SparseInfo(args.accelerator, model, dataset, model_config, args.trace_dir, dse="block_size_" + str(file))
        accelerator = Accelerator(args.accelerator, args.SEC_only, focus_m_tile_size=1024, block_size=block_size_int)
        simulator = Simulator(model_config, accelerator, sparse_info)
        simulator.run()
        result_dict = simulator.get_result()
        result_dict = simulator.get_energy_breakdown()
        result_dict['block_size'] = block_size

        file_name = "dse_c_block_size"
        save_result(result_dict, f"{args.output_dir}/{file_name}.csv")

def dse_m_tile_size(args):
    models = [args.model]
    datasets = [args.dataset]
    m_tile_sizes = [-1, 4096, 2048, 1024, 512, 128, 32] 
    for model in models:
        for dataset in datasets:
            for m_tile_size in m_tile_sizes:
                print("--------------------------------")
                print(model, dataset, m_tile_size)
                model_config = ModelConfig(model, dataset, args.trace_dir)
                sparse_info = SparseInfo(args.accelerator, model, dataset, model_config, args.trace_dir, dse="m_tile_size_" + str(m_tile_size))
                accelerator = Accelerator(args.accelerator, False, focus_m_tile_size=m_tile_size, force_cacti=True)
                simulator = Simulator(model_config, accelerator, sparse_info)
                simulator.run()
                result_dict = simulator.get_result()
                result_dict = simulator.get_energy_breakdown()
                
                result_dict['m_tile_size'] = m_tile_size
                result_dict['total_buffer_capacity'] = accelerator.total_buffer_capacity
                file_name = "dse_a_m_tile_size"
                save_result(result_dict, f"{args.output_dir}/{file_name}.csv")

def dse_vector_size(args):
    models = [args.model]
    datasets = [args.dataset]
    vector_sizes = [4096, 2048, 512, 128, 32, 8]
    assert args.accelerator == 'focus'
    for model in models:
        for dataset in datasets:
            for vector_size in vector_sizes:
                print("--------------------------------")
                print(model, dataset, vector_size)
                model_config = ModelConfig(model, dataset, args.trace_dir)
                sparse_info = SparseInfo(args.accelerator, model, dataset, model_config, args.trace_dir, dse="vector_size_" + str(vector_size))
                accelerator = Accelerator(args.accelerator, False, focus_m_tile_size=1024, force_cacti=True)
                if vector_size < 32:
                    accelerator.systolic_config['array_height'] = vector_size
                    accelerator.systolic_config['array_width'] = 32 * 32 // vector_size
                else:
                    accelerator.systolic_config['array_height'] = 32
                    accelerator.systolic_config['array_width'] = 32
                simulator = Simulator(model_config, accelerator, sparse_info)
                result_dict_list = simulator.run_layer_wise_focus(sparse_info, "o_proj", 9, vector_size)
                file_name = f"dse_b_vector_size"
                for result_dict in result_dict_list:
                    result_dict['vector_size'] = vector_size
                    save_result(result_dict, f"{args.output_dir}/{file_name}.csv")

def dse_num_scatter(args):
    models = [args.model]
    datasets = [args.dataset]
    num_scatters = [1, 2, 3, 4, 5] # one scatter vector is 32 accumulators
    for model in models:
        for dataset in datasets:
            for num_scatter in num_scatters:
                print("--------------------------------")
                print(model, dataset, num_scatter)
                model_config = ModelConfig(model, dataset, args.trace_dir)
                sparse_info = SparseInfo(args.accelerator, model, dataset, model_config, args.trace_dir)
                accelerator = Accelerator(args.accelerator, False, focus_m_tile_size=1024, num_scatter_vector=num_scatter, force_cacti=True)
                simulator = Simulator(model_config, accelerator, sparse_info)
                simulator.run()
                result_dict = simulator.get_result()
                result_dict = simulator.get_energy_breakdown()
                concentrate_out_buffer_area = simulator.accelerator.buffer_dict['concentrate_out']['area']
                output_buffer_area = simulator.accelerator.buffer_dict['output']['area']
                result_dict['buffer_area'] = concentrate_out_buffer_area + output_buffer_area
                result_dict['num_scatter'] = num_scatter
                file_name = "dse_d_num_scatter_accumulator"
                save_result(result_dict, f"{args.output_dir}/{file_name}.csv")

def run_quantization(args):
    if args.all_models_datasets:
        models = ['llava_vid', 'llava_onevision', 'minicpm_v']
        datasets = ['videomme', 'mlvu', 'mvbench']
    else:
        models = [args.model]
        datasets = [args.dataset]
    assert args.accelerator == 'focus'
    for model in models:
        for dataset in datasets:
            print("--------------------------------")
            print(model, dataset)
            model_config = ModelConfig(model, dataset, args.trace_dir)
            sparse_info = SparseInfo(args.accelerator, model, dataset, model_config, args.trace_dir, dse="quantization")
            accelerator = Accelerator(args.accelerator, args.SEC_only, focus_m_tile_size=1024)
            simulator = Simulator(model_config, accelerator, sparse_info)
            simulator.run()
            result_dict = simulator.get_result()
            result_dict = simulator.get_energy_breakdown()
            file_name = "int8_focus"
            save_result(result_dict, f"{args.output_dir}/{file_name}.csv")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--trace_dir', type=str, default='../algorithm/example_output')
    parser.add_argument('--accelerator', type=str, default='focus')
    parser.add_argument('--model', type=str, default='llava_vid')
    parser.add_argument('--dataset', type=str, default='videomme')
    parser.add_argument('--all_models_datasets', action='store_true')
    parser.add_argument('--image_models_datasets', action='store_true')
    parser.add_argument('--SEC_only', action='store_true')
    parser.add_argument('--m_tile_size_dse', action='store_true')
    parser.add_argument('--block_size_dse', action='store_true')
    parser.add_argument('--vector_size_dse', action='store_true')
    parser.add_argument('--num_scatter_dse', action='store_true')
    parser.add_argument('--quantization', action='store_true')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.m_tile_size_dse:
        dse_m_tile_size(args)
    elif args.block_size_dse:
        dse_block_size(args)
    elif args.vector_size_dse:
        dse_vector_size(args)
    elif args.num_scatter_dse:
        dse_num_scatter(args)
    elif args.quantization:
        run_quantization(args)
    else:
        main(args)