from typing import List, Tuple
import torch.nn as nn
from transformers import AutoProcessor, AutoModelForCausalLM
from lmms_eval.models import get_model
import pandas as pd
import os

class ModelConfig:
    def __init__(self, model: str, dataset: str, trace_dir: str):
        self.model = model
        self.dataset = dataset
        self.trace_dir = trace_dir
        self.path = "meta_data.csv"
        self.path = os.path.join(self.trace_dir, self.path)
        self.add_seq_len()
        self.get_QWen2_7B_architecture()
        # self.retrieve_model_architecture(model, model_args)

    def get_QWen2_7B_architecture(self):
        self.dim = 3584
        self.num_heads = 28
        self.num_blocks = 28
        self.dim_per_head = self.dim // self.num_heads  # 128

        self.layers = {
            'q_proj': {
                'in_features': self.dim,
                'out_features': self.dim,
                'seq_len': self.seq_len,
            },
            'k_proj': {
                'in_features': self.dim,
                'out_features': 512,
                'seq_len': self.seq_len,
            },
            'v_proj': {
                'in_features': self.dim,
                'out_features': 512,
                'seq_len': self.seq_len,
            },
            'o_proj': {
                'in_features': self.dim,
                'out_features': self.dim,
                'seq_len': self.seq_len,
            },
            'gate_proj': {
                'in_features': self.dim,
                'out_features': 18944,
                'seq_len': self.seq_len,
            },
            'up_proj': {
                'in_features': self.dim,
                'out_features': 18944,
                'seq_len': self.seq_len,
            },
            'down_proj': {
                'in_features': 18944,
                'out_features': self.dim,
                'seq_len': self.seq_len,
            },
            'attn': {
                'dim_per_head': self.dim_per_head,
                'num_heads': self.num_heads,
                'seq_len': self.seq_len,
            },
        }


    def retrieve_model_architecture(self, model: str, model_args: str):
        """
        Extracts the architecture of the model: (layer name, number of parameters, module type)
        """
        try:
            if model_args is None:
                model_args = ""
            lm = get_model(model).create_from_arg_string(model_args)
        except Exception as e:
            raise ValueError(f"Failed to load model: {e}")

        self.dim = lm.model.model.embed_tokens.weight.shape[1]
        self.num_heads = 28

        self.num_blocks = len(lm.model.model.layers)
        self.layers = dict()
        for name, module in lm.model.model.layers[0].named_modules():
            name = name.split(".")[-1]  # Get the last part of the name
            if isinstance(module, nn.Linear):
                layer_config = {
                    "in_features": module.in_features,
                    "out_features": module.out_features,
                }
                self.layers[name] = layer_config

        dim_per_head = self.dim // self.num_heads
        self.layers["attn"] = {
            "dim_per_head": dim_per_head,
            "num_heads": self.num_heads,
        }


    def add_seq_len(self):
        df = pd.read_csv(self.path)
        row = df[(df['Model'] == self.model) & (df['Dataset'] == self.dataset)]
        assert len(row) == 1, f"Model and dataset name must match exactly one row, got {len(row)}"
        self.seq_len = row['Sequence length'].values[0]
        self.num_frames = row['Num frames'].values[0]
        self.num_patches = row['Num patches'].values[0]
        assert self.seq_len >= self.num_frames * self.num_patches # pad token may be added

        # for name, layer in self.layers.items():
        #     layer["seq_len"] = self.seq_len





if __name__ == "__main__":
    # Example usage
    model_type = "llava_vid"
    model_args = "pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,conv_template=qwen_1_5,max_frames_num=64,mm_spatial_pool_mode=average"
    
    model_config = ModelConfig(model_type, model_args)

    print(model_config.layers)
