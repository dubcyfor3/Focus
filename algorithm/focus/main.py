from typing import List
import copy

import torch
from torch import nn
import numpy as np
import pandas as pd
import os

from focus.utils import AverageMeter, save_result_to_csv, read_from_csv

import matplotlib.pyplot as plt

TEXT_TOKEN = -1
IGNORE_TOKEN = -2


class Focus(nn.Module):
    def __init__(self, 
    similarity_threshold=0.9,  
    vector_size=32,
    block_size=2, 
    frame_block_size=2, 
    gemm_m_size=None,
    alpha_list="",
    selected_layers="",
    SEC_only=False,
    model_name="",
    dataset_name="",
    export_focus_trace=False,
    trace_dir=None,
    trace_meta_dir=None,
    trace_name=None,
    use_median=False,
    num_layers=None,
    hidden_dim=None,
    intermediate_dim=None,
    gqa_group_size=None,
    ):
        super(Focus, self).__init__()
        self.similarity_threshold = similarity_threshold

        self.vector_size = vector_size

        self.maximum_sparsity = 0.0
        self.maximum_sparsity_idx = -1
        self.cur_idx = 0
        self.sparsity_dict = {}

        self.block_size = block_size
        self.frame_block_size = frame_block_size

        self.selected_layer = [int(layer) for layer in selected_layers.split(",")] if selected_layers != "" else []
        self.alpha = [float(alpha) for alpha in alpha_list.split(",")] if alpha_list != "" else []
        
        self.training = False

        self.SEC_only = SEC_only

        self.sparsity_dict_cur = {}
        self.token_importance = None
        self.start_drop = False

        self.gemm_m_size = gemm_m_size
        self.export_focus_trace = export_focus_trace
        self.trace_dir = trace_dir
        self.trace_meta_dir = trace_meta_dir
        self.trace_name = trace_name
        self.use_median = use_median
        self.limit = 10

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.gqa_group_size = gqa_group_size
        self.sparse_layer_by_layer = False
        self.similarity_analysis_mode = False

        self.model_name = model_name
        self.dataset_name = dataset_name

    def forward(self, hidden_states: torch.Tensor, is_attention=False, name=None) -> torch.Tensor:
        """
        Args:
            hidden_states (torch.Tensor): A tensor of shape (bsz, seq_len, hidden_dim) containing the input embeddings.

        Returns:
            torch.Tensor: A tensor of shape (bsz, seq_len, seq_len) that is approximated through focus
        """
        if self.sparsity_dict.get(name) is None:
            self.sparsity_dict[name] = AverageMeter()

        if self.sparsity_dict_cur.get(name) is None:
            self.sparsity_dict_cur[name] = AverageMeter()

        if len(hidden_states.shape) == 3:
            assert is_attention == False
            bsz, q_len, dim = hidden_states.size()
            assert bsz == 1
        elif len(hidden_states.shape) == 4:
            assert is_attention == True
            bsz, num_head, q_len, dim_per_head = hidden_states.size()
            assert bsz == 1
        else:
            raise NotImplementedError
        if q_len == 1:
            return hidden_states
        
        hidden_states_out = hidden_states.clone()

        if self.start_drop:
            hidden_states_out = self.recover_tokens(hidden_states_out)


        if is_attention:
            image_tokens = hidden_states_out[:, :, self.image_token_start_index:self.image_token_start_index+self.image_token_length, :].clone()
        else:
            image_tokens = hidden_states_out[:, self.image_token_start_index:self.image_token_start_index+self.image_token_length, :].clone()

        q_len_image = image_tokens.size(1) if len(image_tokens.shape) == 3 else image_tokens.size(2)

        if self.SEC_only:
            # count number of all zero tokens
            if len(image_tokens.shape) == 3:
                num_zero_tokens = torch.sum(torch.all(image_tokens == 0.0, dim=-1))
                sparsity = num_zero_tokens / q_len_image
            elif len(image_tokens.shape) == 4:
                num_zero_tokens = torch.sum(torch.all(image_tokens == 0.0, dim=(1,-1)))
                sparsity = num_zero_tokens / q_len_image

            self.sparsity_dict[name].update(sparsity)
            self.sparsity_dict_cur[name].update(sparsity)

            return hidden_states

        if self.similarity_analysis_mode:
            if is_attention:
                image_tokens = image_tokens.squeeze(0)
            self.similarity_analysis(image_tokens, name, self.cur_layer)

            if name == "down_proj":
                self.cur_layer += 1

            return hidden_states

        if is_attention:
            image_tokens = image_tokens.transpose(1, 2).contiguous().view(bsz, -1, num_head * dim_per_head)

        # Handle padding for vector_size compatibility specifically for adaptive_range
        padding_size = 0
        
        # Get the hidden dimension (last dimension)
        hidden_dim = image_tokens.shape[-1]
        
        # Check if hidden_dim is divisible by vector_size
        if hidden_dim % self.vector_size != 0 and hidden_dim > self.vector_size:
            # Pad the image_tokens to make hidden_dim divisible by vector_size
            image_tokens, padding_size = self._pad_hidden_dim(image_tokens, self.vector_size)

        image_tokens, sparsity = self.focus_similarity_concentration(image_tokens, name)
        
        # Remove padding if it was added
        if padding_size > 0:
            image_tokens = self._unpad_hidden_dim(image_tokens, padding_size)
        
        # print(f"name: {name}, sparsity: {sparsity}")
        self.sparsity_dict[name].update(sparsity)
        self.sparsity_dict_cur[name].update(sparsity)

        if self.sparse_layer_by_layer:
            # save the sparsity to a csv file by appending a new row
            # the name should contain vector_size
            with open(f'output/sparsity_{self.vector_size}.csv', 'a') as f:
                f.write(f"{name},{sparsity}\n")

        # print(f"{name} sparsity: {sparsity}")
        if is_attention:
            image_tokens = image_tokens.view(bsz, -1, num_head, dim_per_head).transpose(1, 2).contiguous()


        if is_attention:
            hidden_states_out[:, :, self.image_token_start_index:self.image_token_start_index+self.image_token_length, :] = image_tokens
        else:
            hidden_states_out[:, self.image_token_start_index:self.image_token_start_index+self.image_token_length, :] = image_tokens

        if self.start_drop:
            hidden_states_out = self.drop_tokens(hidden_states_out)


        if name == "down_proj":
            self.cur_layer += 1

        return hidden_states_out


    def focus_similarity_concentration(self, image_tokens: torch.Tensor, name: str) -> torch.Tensor:
        '''
        The core function of Focus, implement the similarity concentration algorithm.
        It treat each token in turn as a key token, and compare it with other tokens in the same spatial and temporal block.
        The granularity of similarity comparison is determined by the vector_size.
        '''

        device = image_tokens.device

        # Define the desired spatial and temporal matching window sizes.
        base_match_height = self.block_size
        base_match_width = self.block_size
        base_match_frames = self.frame_block_size

        if image_tokens.dim() != 3:
            raise ValueError("Expected image_tokens to have 3 dimensions.")

        num_heads, seq_len, hidden_dim = image_tokens.shape
        if hidden_dim > self.vector_size: # already padded to multiple of vector_size
            vector_num = hidden_dim // self.vector_size
            cur_vector_size = self.vector_size
        else:
            vector_num = 1
            cur_vector_size = hidden_dim
        

        updated_tokens = (
            image_tokens
            .clone()
            .view(num_heads, seq_len, vector_num, cur_vector_size)
        )

        # ---------- prune out zero tokens (introduced by semantic concentration)----------
        zero_dist   = torch.sum(torch.abs(updated_tokens), dim=-1)     # [H, S, Tn]
        mask_zero   = zero_dist == 0.0
        num_zero_vector = torch.sum(mask_zero).item()
        mask_zero_token = torch.all(mask_zero, dim=-1)
        mask_zero_token = torch.all(mask_zero_token, dim=0)

        # ---------- compute gemm_m_tile indices if needed (tokens in different gemm_m_tiles are not comparable) ----------
        gemm_m_tile_indices = None
        if self.gemm_m_size is not None and self.gemm_m_size > 0:
            gemm_m_tile_indices = self._compute_gemm_m_tile_indices(
                mask_zero_token, seq_len, self.gemm_m_size, device
            )

        match_height = base_match_height if self.patch_height >= base_match_height else self.patch_height
        match_width  = base_match_width  if self.patch_width  >= base_match_width  else self.patch_width
        match_frames = base_match_frames if self.num_frames >= base_match_frames else self.num_frames

        # ---------- build matching candidates ----------
        matching_candidate = torch.zeros(
            (match_frames, match_height, match_width, num_heads, seq_len, vector_num, cur_vector_size),
            device=device,
        )

        dh_offsets = torch.arange(match_height, device=device).view(match_height, 1, 1, 1, 1, 1)
        dw_offsets = torch.arange(match_width, device=device).view(1, match_width, 1, 1, 1, 1)
        linear_offsets = dh_offsets * self.height_stride + dw_offsets * self.width_stride
    
        pos_idx = torch.arange(seq_len, device=device).view(1, 1, 1, seq_len, 1).expand(match_height, match_width, num_heads, seq_len, vector_num)
        
        # Generate candidates for each frame offset
        for frame_offset in range(match_frames):
            for dh in range(match_height):
                for dw in range(match_width):
                    linear_offset = linear_offsets[dh, dw, 0, 0, 0].item()
                    
                    shifted_tokens = _shift_seq(updated_tokens, linear_offset)
                    temporal_shift = frame_offset * self.frame_stride
                    temporal_shifted = _shift_seq(shifted_tokens, temporal_shift)
                    
                    # avoid matching between tokens in different gemm_m_tiles
                    if self.gemm_m_size is not None and self.gemm_m_size > 0:
                        candidate_pos = (pos_idx[dh, dw] - linear_offset - temporal_shift).clamp(0, seq_len - 1)                       
                        gemm_m_tile_mask = self._get_gemm_m_tile_mask_vectorized(
                            candidate_pos, seq_len, num_heads, vector_num, gemm_m_tile_indices, device
                        )                 
                        matching_candidate[frame_offset, dh, dw] = torch.where(
                            gemm_m_tile_mask.unsqueeze(-1).expand_as(temporal_shifted),
                            temporal_shifted,
                            torch.zeros_like(temporal_shifted)
                        )
                    else:
                        matching_candidate[frame_offset, dh, dw] = temporal_shifted

        # remove self-matching
        matching_candidate[0, 0, 0] = torch.zeros_like(matching_candidate[0, 0, 0])

        # similarity matching
        updated_tokens_exp = updated_tokens.unsqueeze(0).unsqueeze(0).expand(
            match_frames, match_height, match_width, num_heads, seq_len, vector_num, cur_vector_size
        )
        similarity = cosine_similarity(updated_tokens_exp, matching_candidate)
        similarity = similarity.view(match_frames * match_height * match_width, num_heads, seq_len, vector_num)
        max_similarity, max_index = similarity.max(dim=0)
        mask = max_similarity > self.similarity_threshold

        # merge all the similar tokens (vectors) in a group (here we make vectors in the same group identical)
        
        # we use a pointer-chasing algorithm to merge the similar tokens (vectors) in a group
        # first, derive offset of candidate tokens
        total_candidates = match_height * match_width
        candidate_type = max_index // total_candidates
        candidate_offset = max_index % total_candidates
        offset_i = candidate_offset // match_width  # Row offset
        offset_j = candidate_offset % match_width   # Column offset

        candidate_type = torch.where(mask, candidate_type, torch.zeros_like(candidate_type))
        offset_i = torch.where(mask, offset_i, torch.zeros_like(offset_i))
        offset_j = torch.where(mask, offset_j, torch.zeros_like(offset_j))

        h_idx   = torch.arange(num_heads, device=device).view(num_heads, 1, 1).expand(num_heads, seq_len, vector_num)
        pos_idx = torch.arange(seq_len,   device=device).view(1,        seq_len, 1).expand(num_heads, seq_len, vector_num)
        vector_idx= torch.arange(vector_num,  device=device).view(1, 1,  vector_num).expand(num_heads, seq_len, vector_num)

        spatial_offset  = offset_i * self.height_stride + offset_j * self.width_stride
        temporal_offset = candidate_type * self.frame_stride
        candidate_pos   = (pos_idx - spatial_offset - temporal_offset).clamp(0, seq_len - 1)

        tokens_per_head = seq_len * vector_num
        linear_idx      = h_idx * tokens_per_head + pos_idx      * vector_num + vector_idx
        candidate_linear= h_idx * tokens_per_head + candidate_pos* vector_num + vector_idx

        # flatten the linear and candidate indices
        M = num_heads * seq_len * vector_num
        linear_idx_flat       = linear_idx.view(M)
        candidate_linear_flat = candidate_linear.view(M)
        mask_flat             = mask.view(M)

        # Initialize a pointer for each token. 
        # If mask is False, the token points to itself (i.e. linear_idx_flat), otherwise to its candidate.
        pointer = torch.where(mask_flat, candidate_linear_flat, linear_idx_flat)

        # ----- Perform pointer-chasing (union-find style) -----
        # Iteratively update pointer until convergence.
        max_iters = self.num_frames  # chase for max_iters times
        for _ in range(max_iters):
            new_pointer = pointer[pointer]  # pointer = pointer(pointer)
            if torch.equal(new_pointer, pointer):
                break
            pointer = new_pointer
        group_idx = pointer.view(num_heads, seq_len, vector_num)


        tokens_flat = updated_tokens.reshape(M, cur_vector_size)
        group_idx_flat = group_idx.view(M)


        num_groups = int(group_idx_flat.max().item()) + 1

        group_sum = torch.zeros((num_groups, cur_vector_size), device=device, dtype=torch.float32)
        group_count = torch.zeros(num_groups, device=device, dtype=torch.float32)

        group_sum = group_sum.index_add(0, group_idx_flat, tokens_flat.to(torch.float32))
        ones = torch.ones_like(group_idx_flat, dtype=torch.float32)
        group_count = group_count.index_add(0, group_idx_flat, ones)
        group_count[group_count == 0] = 1
        group_avg = group_sum / group_count.unsqueeze(-1)
        group_avg = group_avg.to(tokens_flat.dtype)

        # Assign the averaged value to each token in the group:
        tokens_avg_flat = group_avg[group_idx_flat]
        tokens_avg = tokens_avg_flat.view(num_heads, seq_len, vector_num, cur_vector_size)
        updated_tokens_out = tokens_avg.view(num_heads, seq_len, hidden_dim)

        # Compute sparsity metric
        num_similar_vectors = torch.sum(mask_flat).item()
        num_total_vectors = num_heads * seq_len * vector_num
        sparsity = (num_similar_vectors + num_zero_vector) / num_total_vectors

        # sparse trace used by hardware simulator
        if self.export_focus_trace:
            self.info_dict["mask_zero"][name][self.cur_layer] = mask_zero.cpu()
            self.info_dict["mask_similar"][name][self.cur_layer] = mask.cpu()
            self.info_dict["group_idx"][name][self.cur_layer] = group_idx.cpu()


        return updated_tokens_out, sparsity

    def update_alpha(self, layer_idx):
        # find idx of layer_idx in self.selected_layer
        idx = self.selected_layer.index(layer_idx)
        self.cur_alpha = self.alpha[idx]

    def semantic_concentration(self, position_embeddings, attention_mask):
        if self.token_importance is None:
            raise ValueError("Token importance is not set. Please set it before pruning.")
        
        # Add safety check for state consistency
        if self.start_drop and self.retained_ids is None:
            raise ValueError("start_drop is True but retained_ids is None. This indicates inconsistent state.")
        
        # get top k important tokens
        k = int(self.image_token_length * self.cur_alpha)
        if k > self.image_token_length_cur:
            k = self.image_token_length_cur
        
        # Ensure k is at least 1
        if k <= 0:
            k = 1
            
        seq_len = self.original_length
        device = position_embeddings[0].device

        # Validate token_importance
        if self.token_importance.shape[0] == 0:
            raise ValueError("token_importance is empty")
        
        if k > self.token_importance.shape[0]:
            k = self.token_importance.shape[0]

        try:
            retained_ids_local = torch.topk(self.token_importance, k=k)[1]
            # sort the retained ids
            retained_ids_local = torch.sort(retained_ids_local)[0]

            retained_ids = self.retained_ids if self.retained_ids is not None else torch.arange(0, seq_len, device=device)
            
            # Validate retained_ids structure
            if retained_ids.shape[0] < self.image_token_start_index + self.query_token_length:
                raise ValueError(f"retained_ids too short: {retained_ids.shape[0]} < {self.image_token_start_index + self.query_token_length}")
            
            retained_ids_image = retained_ids[self.image_token_start_index:-self.query_token_length]
            retained_ids_before_image = retained_ids[:self.image_token_start_index]
            retained_ids_after_image = retained_ids[-self.query_token_length:]

            # Validate that we have enough image tokens
            if retained_ids_image.shape[0] == 0:
                raise ValueError("No image tokens found in retained_ids")
            
            # Ensure retained_ids_local indices are within bounds
            if retained_ids_local.max().item() >= retained_ids_image.shape[0]:
                # Clamp indices to valid range
                retained_ids_local = torch.clamp(retained_ids_local, 0, retained_ids_image.shape[0] - 1)
                # Remove duplicates
                retained_ids_local = torch.unique(retained_ids_local)

            # use retained_ids_local to select the retained ids
            retained_ids_image = retained_ids_image[retained_ids_local]

            retained_ids_full = torch.cat([retained_ids_before_image, retained_ids_image, retained_ids_after_image], dim=0).contiguous()

            # Validate the final retained_ids_full
            if retained_ids_full.shape[0] == 0:
                raise ValueError("No tokens retained after pruning")

            # gather the hidden states
            # hidden_states = hidden_states[:, retained_ids_full, :]

            if self.start_drop:
                position_embeddings, attention_mask = self.recover_PE_and_AM(position_embeddings, attention_mask)
            else:
                self.start_drop = True

            # Add bounds checking before indexing
            if retained_ids_full.max().item() >= position_embeddings[0].shape[-2]:
                raise ValueError(f"retained_ids_full index {retained_ids_full.max().item()} out of bounds for position_embeddings with shape {position_embeddings[-2].shape}")
            
            # Prune position embeddings based on their dimensions
            if position_embeddings[0].dim() == 3:
                # 3D case: (B, seq_len, C) - prune along second dimension
                position_embeddings[0] = position_embeddings[0][:, retained_ids_full, :]
                position_embeddings[1] = position_embeddings[1][:, retained_ids_full, :]
            elif position_embeddings[0].dim() == 4:
                # Check if this is the special Qwen2.5-VL case: (3, B, seq_len, C)
                if position_embeddings[0].shape[0] == 3:
                    # Special Qwen2.5-VL case: (3, B, seq_len, C) - prune along third dimension
                    position_embeddings[0] = position_embeddings[0][:, :, retained_ids_full, :]
                    position_embeddings[1] = position_embeddings[1][:, :, retained_ids_full, :]
                else:
                    # General 4D case: (B, H, seq_len, C) - prune along third dimension
                    position_embeddings[0] = position_embeddings[0][:, :, retained_ids_full, :]
                    position_embeddings[1] = position_embeddings[1][:, :, retained_ids_full, :]
            else:
                raise ValueError(f"Unsupported position_embeddings dimension: {position_embeddings[0].dim()}")

            if attention_mask is not None:
                if retained_ids_full.max().item() >= attention_mask.shape[-1]:
                    raise ValueError(f"retained_ids_full index {retained_ids_full.max().item()} out of bounds for attention_mask with shape {attention_mask.shape}")
                attention_mask = attention_mask[:, :, retained_ids_full, :][:, :, :, retained_ids_full]

            self.retained_ids = retained_ids_full

        except RuntimeError as e:
            if "CUDA" in str(e):
                # Clear CUDA cache and try to recover
                torch.cuda.empty_cache()
                raise RuntimeError(f"CUDA error in semantic_concentration: {e}. Try reducing batch size or sequence length.")
            else:
                raise e

        return position_embeddings, attention_mask

    def prepare(self, patch_type, num_frames, patch_height, patch_width, frame_stride, height_stride, width_stride, image_token_start_index, image_token_end_index, image_token_length, original_length, query_token_start_index=None, query_token_length=None):
        self.patch_type = patch_type
        self.patch_num = patch_height * patch_width
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.num_frames = num_frames

        self.frame_stride = frame_stride
        self.height_stride = height_stride
        self.width_stride = width_stride

        self.image_token_start_index = image_token_start_index
        self.image_token_end_index = image_token_end_index
        self.image_token_length = image_token_length
        self.query_token_start_index = query_token_start_index if query_token_start_index is not None else image_token_end_index + 1
        self.query_token_length = query_token_length if query_token_length is not None else original_length - self.query_token_start_index
        self.original_length = original_length
        self.image_token_length_cur = image_token_length

        self.token_importance = None
        self.start_drop = False
        self.retained_ids = None

        self.attn_cnt = 0
        
        self.act_dict = {}

        self.cur_layer = 0
        if self.export_focus_trace:
            if not hasattr(self, 'focus_info'):
                self.focus_info = []

            num_vector = (self.hidden_dim + self.vector_size - 1) // self.vector_size
            num_vector_intermediate = (self.intermediate_dim + self.vector_size - 1) // self.vector_size
            
            mask_zero_dict = {"q_proj": torch.empty(self.num_layers, 1, self.image_token_length, num_vector, dtype=torch.bool),
                   "o_proj": torch.empty(self.num_layers, 1, self.image_token_length, num_vector, dtype=torch.bool),
                   "query": torch.empty(self.num_layers, 1, self.image_token_length, num_vector, dtype=torch.bool),
                   "gate_proj": torch.empty(self.num_layers, 1, self.image_token_length, num_vector, dtype=torch.bool),
                   "down_proj": torch.empty(self.num_layers, 1, self.image_token_length, num_vector_intermediate, dtype=torch.bool),}

            mask_similar_dict = {"q_proj": torch.empty(self.num_layers, 1, self.image_token_length, num_vector, dtype=torch.bool),
                   "o_proj": torch.empty(self.num_layers, 1, self.image_token_length, num_vector, dtype=torch.bool),
                   "query": torch.empty(self.num_layers, 1, self.image_token_length, num_vector, dtype=torch.bool),
                   "gate_proj": torch.empty(self.num_layers, 1, self.image_token_length, num_vector, dtype=torch.bool),
                   "down_proj": torch.empty(self.num_layers, 1, self.image_token_length, num_vector_intermediate, dtype=torch.bool),}
            
            group_idx_dict = {"q_proj": torch.empty(self.num_layers, 1, self.image_token_length, num_vector, dtype=torch.int32),
                   "o_proj": torch.empty(self.num_layers, 1, self.image_token_length, num_vector, dtype=torch.int32),
                   "query": torch.empty(self.num_layers, 1, self.image_token_length, num_vector, dtype=torch.int32),
                   "gate_proj": torch.empty(self.num_layers, 1, self.image_token_length, num_vector, dtype=torch.int32),
                   "down_proj": torch.empty(self.num_layers, 1, self.image_token_length, num_vector_intermediate, dtype=torch.int32),}

            self.info_dict = {"mask_zero": mask_zero_dict,
                              "mask_similar": mask_similar_dict,
                              "group_idx": group_idx_dict}



    def set_token_importance(self, attn_weights):
        b, num_head, q_len_0, q_len_1 = attn_weights.size()
        if q_len_0 == 1:
            return
        assert q_len_0 == q_len_1 
        assert b == 1

        # get text to image attentions
        image_start_index = self.image_token_start_index
        query_token_length = self.query_token_length

        text_to_image_attn = attn_weights[:, :, -query_token_length:, image_start_index:-query_token_length]

        # store text to image attn
        # torch.save(text_to_image_attn, 'output/simi/text_to_image_attn.pth')
        # raise NotImplementedError("text_to_image_attn is not used in the current version")

        text_to_image_attn = text_to_image_attn[0].max(dim=0)[0].max(dim=0)[0]
        # assert text_to_image_attn.shape == (self.image_token_length,)

        self.token_importance = text_to_image_attn
        # self.visualize_token_importance(self.token_importance)


    def visualize_token_importance(self, token_importance):
        median_value = torch.median(token_importance).item()
        max_value = torch.max(token_importance).item()
        min_value = torch.min(token_importance).item()
        # add median max min to the title
        plt.title(f'token_importance_{self.attn_cnt}_{median_value}_{max_value}_{min_value}')
        plt.hist(token_importance.cpu().numpy(), bins=100)
        plt.savefig(f'output/token_importance_{self.attn_cnt}.png')
        self.attn_cnt += 1
        # clear the figure
        plt.clf()

    def post_process(self):
        
        q_proj_size = self.hidden_dim * self.hidden_dim + self.hidden_dim * self.hidden_dim // self.gqa_group_size * 2
        o_proj_size = self.hidden_dim * self.hidden_dim
        query_size = self.hidden_dim * self.image_token_length
        gate_proj_size = self.hidden_dim * self.intermediate_dim * 2
        down_proj_size = self.hidden_dim * self.intermediate_dim
        
        config_dict = {"q_proj": q_proj_size,
                   "o_proj": o_proj_size,
                   "query": query_size,
                   "gate_proj": gate_proj_size,
                   "down_proj": down_proj_size,}
        
        # assert all keys in sparsity_cur_dict is in config_dict
        for key in self.sparsity_dict_cur.keys():
            assert key in config_dict, f"{key} not in config_dict"


        # get the weighted average sparsity according to config_dict
        weighted_avg_sparsity = 0
        sum_weight = 0
        for key, value in self.sparsity_dict_cur.items():
            weighted_avg_sparsity += value.avg * config_dict[key]
            sum_weight += config_dict[key]
        weighted_avg_sparsity /= sum_weight

        if weighted_avg_sparsity > self.maximum_sparsity:
            self.maximum_sparsity = weighted_avg_sparsity
            self.maximum_sparsity_idx = self.cur_idx

        self.cur_idx += 1
        self.sparsity_dict_cur = {}

        if not hasattr(self, 'weighted_avg_sparsity_list'):
            self.weighted_avg_sparsity_list = []
        self.weighted_avg_sparsity_list.append(weighted_avg_sparsity)

        if len(self.weighted_avg_sparsity_list) >= self.limit:
            # Convert list to tensor for easier computation
            sparsity_tensor = torch.tensor(self.weighted_avg_sparsity_list)
                
            # Calculate median value
            median_value = torch.median(sparsity_tensor).item()
                
            # Find the index of the median value (or closest to it if there are duplicates)
            median_idx = torch.argmin(torch.abs(sparsity_tensor - median_value)).item()
                
            # Store the results as instance variables
            self.median_sparsity = median_value
            self.median_sparsity_idx = median_idx
            self.weighted_avg_sparsity_list = []

        if self.export_focus_trace:

            # append a deep copy of self.info_dict to self.focus_info and clear self.info_dict
            self.focus_info.append(copy.deepcopy(self.info_dict))
            self.info_dict = {}

            if not hasattr(self, "seq_len_info"):
                self.seq_len_info = []
            self.seq_len_info.append((self.image_token_length, self.num_frames))

            if len(self.focus_info) >= self.limit:

                # save the focus_info to a file
                if self.use_median:
                    if not os.path.exists(self.trace_meta_dir):
                        os.makedirs(self.trace_meta_dir)
                    median_info_dict = self.focus_info[self.median_sparsity_idx]
                    seq_len = self.seq_len_info[self.median_sparsity_idx][0]
                    num_frames = self.seq_len_info[self.median_sparsity_idx][1]
                    meta_info_dict = {
                        "Model": self.model_name,
                        "Dataset": self.dataset_name,
                        "Sequence length": seq_len,
                        "Num frames": num_frames,
                        "Num patches": seq_len // num_frames,
                        "Median index": self.median_sparsity_idx,
                    }
                    save_result_to_csv(meta_info_dict, f'{self.trace_meta_dir}/meta_data.csv')
                else:
                    assert os.path.exists(f'{self.trace_meta_dir}/meta_data.csv'), f"meta info {f'{self.trace_meta_dir}/meta_data.csv'} does not exist"
                    # use index in meta to get the focus_info
                    meta_info_dict = read_from_csv({"Model": self.model_name, "Dataset": self.dataset_name}, f'{self.trace_meta_dir}/meta_data.csv')
                    index = meta_info_dict["Median index"].values[0]
                    median_info_dict = self.focus_info[index]

                # make dir if not exists
                if not os.path.exists(self.trace_dir):
                    os.makedirs(self.trace_dir)

                trace_file_name = f'{self.trace_dir}/{self.trace_name}.pth'
                torch.save(median_info_dict, trace_file_name)
                print(f"Saved focus trace to {trace_file_name}")

                self.focus_info = []
                self.seq_len_info = []
    
    def recover_PE_and_AM(self, position_embeddings, attention_mask):
        assert self.retained_ids.shape[0] == position_embeddings[0].shape[-2], "The number of retained ids must be equal to the number of tokens in the position embeddings."
        
        # Handle position embeddings based on their dimensions
        if position_embeddings[0].dim() == 3:
            # 3D case: (B, seq_len, C)
            B, _, C = position_embeddings[0].shape
            tmp = torch.zeros(B, self.original_length, C, device=position_embeddings[0].device, dtype=position_embeddings[0].dtype)
            tmp[:, self.retained_ids, :] = position_embeddings[0]
            position_embeddings[0] = tmp
            tmp = torch.zeros(B, self.original_length, C, device=position_embeddings[1].device, dtype=position_embeddings[1].dtype)
            tmp[:, self.retained_ids, :] = position_embeddings[1]
            position_embeddings[1] = tmp
        elif position_embeddings[0].dim() == 4:
            # Check if this is the special Qwen2.5-VL case: (3, B, seq_len, C)
            if position_embeddings[0].shape[0] == 3:
                # Special Qwen2.5-VL case: (3, B, seq_len, C)
                _, B, _, C = position_embeddings[0].shape
                tmp = torch.zeros(3, B, self.original_length, C, device=position_embeddings[0].device, dtype=position_embeddings[0].dtype)
                tmp[:, :, self.retained_ids, :] = position_embeddings[0]
                position_embeddings[0] = tmp
                tmp = torch.zeros(3, B, self.original_length, C, device=position_embeddings[1].device, dtype=position_embeddings[1].dtype)
                tmp[:, :, self.retained_ids, :] = position_embeddings[1]
                position_embeddings[1] = tmp
            else:
                # General 4D case: (B, H, seq_len, C)
                B, H, _, C = position_embeddings[0].shape
                tmp = torch.zeros(B, H, self.original_length, C, device=position_embeddings[0].device, dtype=position_embeddings[0].dtype)
                tmp[:, :, self.retained_ids, :] = position_embeddings[0]
                position_embeddings[0] = tmp
                tmp = torch.zeros(B, H, self.original_length, C, device=position_embeddings[1].device, dtype=position_embeddings[1].dtype)
                tmp[:, :, self.retained_ids, :] = position_embeddings[1]
                position_embeddings[1] = tmp
        else:
            raise ValueError(f"Unsupported position_embeddings dimension: {position_embeddings[0].dim()}")
        if attention_mask is not None:
            tmp = torch.zeros(B, B, self.original_length, self.original_length, device=attention_mask.device, dtype=attention_mask.dtype)
            tmp[:, :, self.retained_ids, :][:, :, :, self.retained_ids] = attention_mask
            attention_mask = tmp
        return position_embeddings, attention_mask



    def recover_tokens(self, hidden_states):
        """
        Recover the full hidden_states by restoring the pruned positions with zeros.
        """
        # Create a zero tensor of the original shape
        if hidden_states.dim() == 3:
            B, _, C = hidden_states.shape
            recovered = torch.zeros(B, self.original_length, C, device=hidden_states.device, dtype=hidden_states.dtype)
        
            # Scatter the retained hidden states back to their original positions
            recovered[:, self.retained_ids, :] = hidden_states

        elif hidden_states.dim() == 4:
            B, H, _, C = hidden_states.shape
            recovered = torch.zeros(B, H, self.original_length, C, device=hidden_states.device, dtype=hidden_states.dtype)
            # Scatter the retained hidden states back to their original positions
            recovered[:, :, self.retained_ids, :] = hidden_states

        else:
            raise ValueError("hidden_states must be 3D or 4D tensor.")

        return recovered
    
    def drop_tokens(self, hidden_states):
        """
        Drop the tokens that are not retained.
        """
        if hidden_states.dim() == 3:
            dropped = hidden_states[:, self.retained_ids, :]
            # make contiguous
            dropped = dropped.contiguous()
        elif hidden_states.dim() == 4:

            dropped = hidden_states[:, :, self.retained_ids, :]
            # make contiguous
            dropped = dropped.contiguous()
        else:
            raise ValueError("hidden_states must be 3D or 4D tensor.")
        
        return dropped

    def _compute_gemm_m_tile_indices(
        self, 
        mask_zero_token: torch.Tensor, 
        seq_len: int, 
        gemm_m_size: int, 
        device: torch.device
    ) -> torch.Tensor:
        """
        Compute gemm_m_tile indices based on nonzero tokens.
        
        Args:
            mask_zero_token: Boolean tensor indicating which tokens are zero (shape: [seq_len])
            seq_len: Length of the sequence
            gemm_m_size: Size of each gemm_m_tile
            device: Device to create tensors on
            
        Returns:
            gemm_m_tile_indices: Tensor of shape [num_gemm_m_tile, 2] containing (start, end) indices for each tile
        """
        # count the number of non_zero tokens
        num_non_zero_token = torch.sum(~mask_zero_token).item()

        # determine gemm_m_tile based on number of nonzero tokens
        num_gemm_m_tile = (num_non_zero_token + gemm_m_size - 1) // gemm_m_size  # ceiling division

        # create cumulative sum of nonzero tokens to find gemm_m_tile boundaries
        nonzero_cumsum = torch.cumsum(~mask_zero_token, dim=0)

        # create gemm_m_tile index ranges ensuring each gemm_m_tile has focus_gemm_m_tile nonzero tokens
        gemm_m_tile_indices = []
        for i in range(num_gemm_m_tile):
            target_count = (i + 1) * gemm_m_size

            # find the index where cumulative sum reaches target_count
            if target_count <= num_non_zero_token:
                # find the first index where cumsum >= target_count
                gemm_m_tile_end_idx = torch.searchsorted(nonzero_cumsum, target_count, right=True)
            else:
                # for the last gemm_m_tile, use the end of sequence
                gemm_m_tile_end_idx = torch.tensor(seq_len, device=device, dtype=torch.long)

            # find the start index for this gemm_m_tile
            if i == 0:
                gemm_m_tile_start_idx = torch.tensor(0, device=device, dtype=torch.long)
            else:
                prev_target_count = i * gemm_m_size
                gemm_m_tile_start_idx = torch.searchsorted(nonzero_cumsum, prev_target_count, right=True)

            gemm_m_tile_indices.append((gemm_m_tile_start_idx.item(), gemm_m_tile_end_idx.item()))

        # convert to tensor for easier handling
        gemm_m_tile_indices = torch.tensor(gemm_m_tile_indices, device=device, dtype=torch.long)
        
        return gemm_m_tile_indices

    def _get_gemm_m_tile_mask_vectorized(
        self,
        candidate_positions: torch.Tensor,
        seq_len: int,
        num_heads: int,
        vector_num: int,
        gemm_m_tile_indices: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        """
        Vectorized version to check if candidate positions are in the same gemm_m_tile as current positions.
        
        Args:
            candidate_positions: tensor of shape (num_heads, seq_len, vector_num) containing candidate positions
            seq_len: Length of the sequence
            num_heads: Number of attention heads
            vector_num: Number of vectors per token
            gemm_m_tile_indices: Tensor of shape [num_gemm_m_tile, 2] containing (start, end) indices for each tile
            device: Device to create tensors on
            
        Returns:
            gemm_m_tile_mask: boolean tensor of same shape, True if candidate is in same gemm_m_tile as current position
        """
        # Create current position indices
        current_positions = torch.arange(seq_len, device=device).view(1, seq_len, 1).expand(num_heads, seq_len, vector_num)
        
        # Find which gemm_m_tile each current position belongs to
        current_gemm_m_tile_idx = torch.zeros_like(current_positions, dtype=torch.long)
        candidate_gemm_m_tile_idx = torch.zeros_like(candidate_positions, dtype=torch.long)
        
        # Vectorized gemm_m_tile assignment
        for i, (start, end) in enumerate(gemm_m_tile_indices):
            current_mask = (current_positions >= start) & (current_positions < end)
            candidate_mask = (candidate_positions >= start) & (candidate_positions < end)
            
            current_gemm_m_tile_idx[current_mask] = i
            candidate_gemm_m_tile_idx[candidate_mask] = i
        
        # Check if current and candidate positions are in the same gemm_m_tile
        gemm_m_tile_mask = (current_gemm_m_tile_idx == candidate_gemm_m_tile_idx)
        
        return gemm_m_tile_mask

    def similarity_analysis(self, image_tokens, name, layer_idx):
        device = image_tokens.device

        # Define the desired spatial matching window sizes.
        base_match_height = self.block_size
        base_match_width = self.block_size

        if image_tokens.dim() != 3:
            raise ValueError("Expected image_tokens to have 3 dimensions.")


        num_heads, seq_len, hidden_dim = image_tokens.shape
        if num_heads != 1:
            image_tokens = image_tokens.permute(1, 0, 2).contiguous()
            image_tokens = image_tokens.view(1, seq_len, num_heads * hidden_dim)
            hidden_dim = num_heads * hidden_dim
            num_heads = 1


        # ---------- 4. spatial window size (same as before) ----------
        match_height = base_match_height if self.patch_height >= base_match_height else self.patch_height
        match_width  = base_match_width  if self.patch_width  >= base_match_width  else self.patch_width

        # ---------- 5. container for matching candidates ----------
        matching_candidate = torch.zeros(
            (2, match_height, match_width, num_heads, seq_len, hidden_dim),
            device=device,
        )

        # ============================
        # 1. Build candidate set for same-frame matching (vectorized)
        # ============================
        # Pre-compute all spatial offsets
        dh_offsets = torch.arange(match_height, device=device).view(match_height, 1, 1, 1, 1)
        dw_offsets = torch.arange(match_width, device=device).view(1, match_width, 1, 1, 1)
        linear_offsets = dh_offsets * self.height_stride + dw_offsets * self.width_stride
        
        # Handle the identity case (dh=0, dw=0) separately
        matching_candidate[0, 0, 0] = image_tokens
        
        # Vectorized processing for non-identity cases
        for dh in range(match_height):
            for dw in range(match_width):
                if dh == 0 and dw == 0:
                    continue  # Already handled above
                
                linear_offset = linear_offsets[dh, dw, 0, 0, 0].item()
                shifted_tokens = _shift_seq(image_tokens, linear_offset)
                
                matching_candidate[0, dh, dw] = shifted_tokens

        # ============================
        # 2. Build candidate set for previous-frame matching (vectorized)
        # ============================
        for dh in range(match_height):
            for dw in range(match_width):
                linear_offset = linear_offsets[dh, dw, 0, 0, 0].item()
                
                # First get the spatial shift
                shifted_tokens = _shift_seq(image_tokens, linear_offset)
                
                # Then apply temporal shift
                temporal_shifted = _shift_seq(shifted_tokens, self.frame_stride)
                
                matching_candidate[1, dh, dw] = temporal_shifted

        # ---------- 3. remove self-matching ----------
        matching_candidate[0, 0, 0] = torch.zeros_like(matching_candidate[0, 0, 0])


        # ============================
        # 3. Compute cosine similarity between the tokens and all candidates.
        # ============================
        # First, expand updated_tokens so that its shape matches matching_candidate:
        # (2, match_height, match_width, num_heads, num_frames, patch_height, patch_width, vector_num, vector_size)
        # updated_tokens_exp = updated_tokens.unsqueeze(0).unsqueeze(0).expand(
        #     2, match_height, match_width, num_heads, self.num_frames, self.patch_height, self.patch_width, vector_num, self.vector_size
        # )
        image_tokens_exp = image_tokens.unsqueeze(0).unsqueeze(0).expand(
            2, match_height, match_width, num_heads, seq_len, hidden_dim
        )

        image_tokens_exp = image_tokens_exp.view(2 * match_height * match_width, num_heads, seq_len, hidden_dim)
        matching_candidate = matching_candidate.view(2 * match_height * match_width, num_heads, seq_len, hidden_dim)

        # vector_length_list = [4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8]
        vector_length_list = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

        hidden_dim = image_tokens_exp.shape[-1]

        ratio_0_9_list = []
        ratio_0_8_list = []

        for vector_length in vector_length_list:
            image_tokens_exp_padded, padding_size = self._pad_hidden_dim(image_tokens_exp, vector_length)
            matching_candidate_padded, padding_size = self._pad_hidden_dim(matching_candidate, vector_length)

            hidden_dim_padded = image_tokens_exp_padded.shape[-1]

            vector_num = hidden_dim_padded // vector_length
            assert hidden_dim_padded % vector_length == 0, (
                f"The hidden dimension must be divisible by vector_length. {hidden_dim_padded} % {vector_length} != 0"
            )

            image_tokens_exp_padded = image_tokens_exp_padded.view(2 * match_height * match_width, num_heads, seq_len, vector_num, vector_length)
            matching_candidate_padded = matching_candidate_padded.view(2 * match_height * match_width, num_heads, seq_len, vector_num, vector_length)

            similarity = cosine_similarity(image_tokens_exp_padded, matching_candidate_padded)

            max_similarity = torch.max(similarity, dim=0)[0]

            # flatten the max_similarity
            max_similarity = max_similarity.view(-1)

            # plot the CCDF (Complementary CDF) of the max_similarity
            max_sim_np = max_similarity.cpu().numpy()
            sorted_max_sim = np.sort(max_sim_np)

            # save the sorted_max_sim to a file
            np.save(f"output/sorted_max_sim_new/{name}_{layer_idx}_{vector_length}_{self.model_name}_{self.dataset_name}.npy", sorted_max_sim)

            # CCDF shows P(X > x) = 1 - P(X <= x)
            # ccdf_max_sim = 1 - np.arange(1, len(sorted_max_sim) + 1) / len(sorted_max_sim)
            # plt.plot(sorted_max_sim, ccdf_max_sim)
            # plt.xlabel('Max Similarity')
            # plt.ylabel('Ratio of Values > Threshold')
            # plt.title(f'CCDF of Max Similarity - {name} Layer {layer_idx} - Vector Length {vector_length}')
            # plt.grid(True, alpha=0.3)
            # plt.savefig(f"output/ccdf/max_similarity_ccdf_{name}_{layer_idx}_{vector_length}.png")
            # plt.close()
            # plt.clf()

            # Calculate ratios for cosine similarity values larger than similarity_thresholds

            ratio_0_9 = np.sum(max_sim_np > 0.9) / (len(max_sim_np))
            ratio_0_8 = np.sum(max_sim_np > 0.8) / (len(max_sim_np))

            ratio_0_9_list.append(ratio_0_9)
            ratio_0_8_list.append(ratio_0_8)


        ratio_0_9_diff = ratio_0_9_list[-1] - ratio_0_9_list[0]
        ratio_0_8_diff = ratio_0_8_list[-1] - ratio_0_8_list[0]
        
        print(f"ratio_0_9_diff in name {self.model_name} layer {layer_idx}: {ratio_0_9_diff}")
        print(f"ratio_0_8_diff in name {self.model_name} layer {layer_idx}: {ratio_0_8_diff}")

        # Create or append to CSV file
        # csv_data = {
        #     'name': [name],
        #     'layer_idx': [layer_idx],
        #     'ratio_0_9_diff': [ratio_0_9_diff],
        #     'ratio_0_8_diff': [ratio_0_8_diff]
        # }
        
        # df = pd.DataFrame(csv_data)
        # csv_path = f"output/cdf_csv/{self.model_name}_{self.dataset_name}_similarity_ratio_diffs.csv"
        
        # # Create output directory if it doesn't exist
        # os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        # # Append to existing file or create new one
        # if os.path.exists(csv_path):
        #     # Read existing data
        #     existing_df = pd.read_csv(csv_path)
        #     # Combine with new data
        #     combined_df = pd.concat([existing_df, df], ignore_index=True)
        #     # Write back to CSV
        #     combined_df.to_csv(csv_path, index=False)
        # else:
        #     df.to_csv(csv_path, index=False)


            # plot the CDF of the min_mse
            # min_mse_np = min_mse.cpu().numpy()
            # sorted_min_mse = np.sort(min_mse_np)
            # cdf_min_mse = np.arange(1, len(sorted_min_mse) + 1) / len(sorted_min_mse)
            # plt.plot(sorted_min_mse, cdf_min_mse)
            # plt.xlabel('Min MSE')
            # plt.ylabel('Cumulative Probability')
            # plt.title(f'CDF of Min MSE - {name} Layer {layer_idx} - Vector Length {vector_length}')
            # plt.grid(True, alpha=0.3)
            # plt.savefig(f"output/cdf/min_mse_cdf_{name}_{layer_idx}_{vector_length}.png")
            # plt.close()
            # plt.clf()


    def _pad_hidden_dim(self, hidden_states, target_multiple):
        """
        Pad hidden_states to make the hidden dimension divisible by target_multiple.
        
        Args:
            hidden_states: Input tensor of shape (..., hidden_dim)
            target_multiple: The target multiple for the hidden dimension
            
        Returns:
            padded_hidden_states: Padded tensor
            padding_size: Number of elements padded (to be removed later)
        """
        hidden_dim = hidden_states.shape[-1]
        remainder = hidden_dim % target_multiple
        
        if remainder == 0:
            return hidden_states, 0
        
        padding_size = target_multiple - remainder
        
        # Pad the last dimension
        if hidden_states.dim() == 3:
            # (batch, seq_len, hidden_dim)
            padded = torch.nn.functional.pad(hidden_states, (0, padding_size), mode='constant', value=0)
        elif hidden_states.dim() == 4:
            # (batch, num_heads, seq_len, hidden_dim)
            padded = torch.nn.functional.pad(hidden_states, (0, padding_size), mode='constant', value=0)
        else:
            # For other dimensions, pad the last dimension
            pad_tuple = [0] * (hidden_states.dim() * 2)
            pad_tuple[-2] = 0  # No padding before the last dimension
            pad_tuple[-1] = padding_size  # Padding after the last dimension
            padded = torch.nn.functional.pad(hidden_states, pad_tuple, mode='constant', value=0)
        
        return padded, padding_size

    def _unpad_hidden_dim(self, hidden_states, padding_size):
        """
        Remove padding from hidden_states.
        
        Args:
            hidden_states: Padded tensor
            padding_size: Number of elements to remove from the end
            
        Returns:
            unpadded_hidden_states: Original sized tensor
        """
        if padding_size == 0:
            return hidden_states
        
        # Remove the padding from the last dimension
        if hidden_states.dim() == 3:
            # (batch, seq_len, hidden_dim)
            return hidden_states[:, :, :-padding_size]
        elif hidden_states.dim() == 4:
            # (batch, num_heads, seq_len, hidden_dim)
            return hidden_states[:, :, :, :-padding_size]
        else:
            # For other dimensions, slice the last dimension
            slice_obj = [slice(None)] * (hidden_states.dim() - 1) + [slice(None, -padding_size)]
            return hidden_states[slice_obj]


# ---------- helper : shift along sequence axis ----------
def _shift_seq(x: torch.Tensor, offset: int) -> torch.Tensor:
    """
    Shift `x` (H, S, Tn, Ts) along the sequence axis (dim=1).
    Pads with zeros where data rolled out.
    Positive offset ==> roll _right_ (towards larger index).
    Negative offset ==> roll _left_  (towards smaller index).
    """
    if offset == 0:
        return x
    if offset > 0:
        pad = torch.zeros_like(x[:, :offset])               # left zeros
        return torch.cat([pad, x[:, :-offset]], dim=1)
    else:                                                   # offset < 0
        offset = -offset
        pad = torch.zeros_like(x[:, :offset])               # right zeros
        return torch.cat([x[:, offset:], pad], dim=1)

def cosine_similarity(mat1, mat2):
    dot_product = torch.sum(mat1 * mat2, dim=-1)
    norm_vec1 = torch.norm(mat1, dim=-1)
    norm_vec2 = torch.norm(mat2, dim=-1)
    denominator = norm_vec1 * norm_vec2
    return torch.where(denominator != 0, dot_product / denominator, torch.zeros_like(denominator))