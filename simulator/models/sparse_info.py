import torch
import pandas as pd
import os

class SparseInfo:
    def __init__(self, type, model, dataset, model_config, trace_dir, dse=""):

        self.type = type
        self.model = model
        self.dataset = dataset
        self.dse = dse
        self.model_config = model_config
        self.trace_dir = trace_dir

        if self.dse.startswith("m_tile_size"):
            # path = 'configs/' + 'm_tile_size_' + model + '_' + dataset + '/' + self.dse + '.pth'
            m_tile_size_str = self.dse.split('_')[-1]
            path = "m_tile_size_dse/" + model + "_" + dataset + "_" + m_tile_size_str + ".pth"
            path = os.path.join(self.trace_dir, path)
            self.info_dict = torch.load(path)
            info_dict_seq_len = self.info_dict['mask_zero']['q_proj'].shape[2]
            assert info_dict_seq_len == model_config.seq_len

        elif self.dse.startswith("block_size"):
            # file_name is dse remove block_size_
            file_name = self.dse.replace("block_size_", "")
            path = 'block_size_dse/' + file_name
            path = os.path.join(self.trace_dir, path)
            self.info_dict = torch.load(path)
            info_dict_seq_len = self.info_dict['mask_zero']['q_proj'].shape[2]
            assert info_dict_seq_len == model_config.seq_len

        elif self.dse.startswith("vector_size"):
            file_name = self.dse + '.pth'
            vector_size_str = self.dse.split('_')[-1]
            path = 'vector_size_dse/' + model + "_" + dataset + "_" + vector_size_str + ".pth"
            path = os.path.join(self.trace_dir, path)
            self.info_dict = torch.load(path)
            info_dict_seq_len = self.info_dict['mask_zero']['q_proj'].shape[2]
            assert info_dict_seq_len == model_config.seq_len
        
        elif self.dse.startswith("quantization"):
            path = 'focus_int8/' + model + '_' + dataset + '.pth'
            path = os.path.join(self.trace_dir, path)
            self.info_dict = torch.load(path)
            info_dict_seq_len = self.info_dict['mask_zero']['q_proj'].shape[2]
            assert info_dict_seq_len == model_config.seq_len

        elif type == 'focus':
            # path = "configs/focus/" + model + "_" + dataset + ".pth"
            path = "focus_main/" + model + "_" + dataset + ".pth"
            path = os.path.join(self.trace_dir, path)
            self.info_dict = torch.load(path)
            info_dict_seq_len = self.info_dict['mask_zero']['q_proj'].shape[2]
            assert info_dict_seq_len == model_config.seq_len
        elif type == 'cmc':
            path = 'cmc_sparsity.csv'
            path = os.path.join(self.trace_dir, path)
            df = pd.read_csv(path)
            # find the row where model name and dataset name match
            row = df[(df['Model'] == model) & (df['Dataset'] == dataset)]
            assert len(row) == 1, "Model and dataset name must match exactly one row"
            self.info_dict = {'linear_sparsity': row['linear_sparsity'].values[0], 'q_sparsity': row['query_sparsity'].values[0], 's_sparsity': row['attn_score_sparsity'].values[0]}
        elif type == 'adaptiv':
            path = 'adaptiv_sparsity.csv'
            path = os.path.join(self.trace_dir, path)
            df = pd.read_csv(path)
            # find the row where model name and dataset name match
            row = df[(df['Model'] == model) & (df['Dataset'] == dataset)]
            assert len(row) == 1, "Model and dataset name must match exactly one row"
            self.info_dict = {'sparsity': row['Sparsity'].values[0]}
        elif type == 'dense':
            pass

        # expected shape (28, 28, self.image_token_length, 3584 // (self.tile_size * 28), dtype=torch.bool)
        if type == 'focus':
            for key in self.info_dict:
                cur_dict = self.info_dict[key]
                if 'query' in cur_dict:
                    query_shape = cur_dict['query'].shape
                    assert query_shape[1] == 1 or query_shape[1] == model_config.num_heads, f"query_shape[1] must be 1 or {model_config.num_heads}, got {query_shape[1]}"
                    # if query_shape[1] == 1:
                    #     print('query_shape', query_shape)
                    #     self.info_dict[key]['query'] = self.info_dict[key]['query'].view(model_config.num_blocks, model_config.seq_len, model_config.num_heads, -1)
                    #     self.info_dict[key]['query'] = self.info_dict[key]['query'].permute(0, 2, 1, 3).contiguous()


if __name__ == "__main__":
    
    # info_path = "configs/image/llava_onevision_mme.pth"
    # sparse_info = SparseInfo("focus", "llava_onevision", "mme", info_path)
    # print(sparse_info.info_dict)

    models = ['llava_onevision', 'qwen2_5_vl']
    datasets = ['vqa', 'mme', 'mmbench']
    for model in models:
        for dataset in datasets:
            path = "configs/partition_size_1024/" + model + "_" + dataset + ".pth"
            info_dict = torch.load(path)
            print(info_dict['mask_zero']['q_proj'].shape[2])
            