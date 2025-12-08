import math
import os
from typing import Any


import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
import torch.nn.functional as F
import pandas as pd


# meta
TEXT_TOKEN = -1
IGNORE_TOKEN = -2


def get_attr_by_name(obj: Any, name: str) -> Any:
    """
    Get an attribute from an object using a dot notation string.
    e.g., get_attr_by_name(model, "layers.0.self_attn.q_proj") will return model.layers[0].self_attn.q_proj
    """
    levels = name.split(".")
    current = obj
    for level in levels:
        if level.isdigit():
            current = current[int(level)]
        else:
            current = getattr(current, level)
    return current


# Efficient implementation equivalent to the following:
def naive_scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0).to(query.device)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.nn.functional.softmax(attn_weight, dim=-1, dtype=torch.float32).to(
        query.dtype
    )

    return attn_weight @ value


class AverageMeter:
    """Computes and stores the average, current value, sum, and count."""
    def __init__(self):
        self.reset()

    def reset(self):
        """Resets all statistics."""
        self.val = 0.0      # current value
        self.avg = 0.0       # average
        self.sum = 0.0       # sum of all values
        self.count = 0.0     # number of updates

    def update(self, val, n=1):
        """Updates the meter with a new value.
        
        Args:
            val (float): New value to add.
            n (int): Weight of the new value (e.g., batch size).
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_result_to_csv(result_dict, csv_path):
    if not os.path.exists(csv_path):
        df = pd.DataFrame(columns=result_dict.keys())
        df.to_csv(csv_path, index=False)
    df = pd.read_csv(csv_path)
    
    # Get the first two column names
    first_two_cols = list(result_dict.keys())[:2]
    
    # Check if a row exists with matching first two column values
    if len(df) > 0 and len(first_two_cols) >= 2:
        mask = (df[first_two_cols[0]] == result_dict[first_two_cols[0]]) & \
               (df[first_two_cols[1]] == result_dict[first_two_cols[1]])
        matching_indices = df.index[mask].tolist()
        
        if matching_indices:
            # Overwrite the first matching row
            for idx in matching_indices:
                for key, value in result_dict.items():
                    df.at[idx, key] = value
        else:
            # Append new row if no match found
            df = pd.concat([df, pd.DataFrame([result_dict])], ignore_index=True)
    else:
        # Append new row if dataframe is empty or columns don't match
        df = pd.concat([df, pd.DataFrame([result_dict])], ignore_index=True)
    
    df.to_csv(csv_path, index=False)

def read_from_csv(query_dict, csv_path):
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        return None
    mask = (df[list(query_dict.keys())[0]] == query_dict[list(query_dict.keys())[0]]) & \
           (df[list(query_dict.keys())[1]] == query_dict[list(query_dict.keys())[1]])
    matching_indices = df.index[mask].tolist()
    if matching_indices:
        return df.iloc[matching_indices]
    else:
        return None