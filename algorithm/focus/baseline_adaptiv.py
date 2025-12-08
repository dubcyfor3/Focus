from typing import List

import torch
from torch import nn

from focus.utils import AverageMeter, save_result_to_csv

TEXT_TOKEN = -1
IGNORE_TOKEN = -2

class Adaptiv(nn.Module):
    def __init__(self, threshold: float, model_name: str, dataset_name: str, trace_meta_dir: str, write_sparsity=False):
        super(Adaptiv, self).__init__()
        self.threshold = threshold
        self.sparsity_dict = {}
        self.write_sparsity = write_sparsity
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.trace_meta_dir = trace_meta_dir
        self.limit = 10
        self.cur_idx = 0

        self.training = False

    def prepare(self, num_frames, patch_height, patch_width, frame_stride, height_stride, width_stride, image_token_start_index, image_token_end_index, image_token_length, original_length, query_token_start_index=None, query_token_length=None):
        
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

        self.merged_ids = None
        self.keep_ids = None


    def post_process(self):

        self.cur_idx += 1
        if self.cur_idx == self.limit:

            output_sparsity_dict = {'Model': self.model_name, 'Dataset': self.dataset_name, 
            'Sparsity': self.sparsity_dict['pre_attn'].avg}

            if self.write_sparsity:
                save_result_to_csv(output_sparsity_dict, f'{self.trace_meta_dir}/adaptiv_sparsity.csv')
                print(f"Saved Adaptiv sparsity to {f'{self.trace_meta_dir}/adaptiv_sparsity.csv'}")
            self.cur_idx = 0

    def forward(self, hidden_states: torch.Tensor, position_embeddings: torch.Tensor, attention_mask: torch.Tensor, name: str):
        assert hidden_states.dim() == 3, "hidden_states should be a 3D tensor (batch_size, sequence_length, hidden_size)"
        bsz, seq_len, dim = hidden_states.shape
        assert bsz == 1, "This implementation currently supports batch size of 1 only"
        if seq_len == 1:
            return hidden_states, position_embeddings, attention_mask

        if self.keep_ids is not None:
            assert self.merged_ids is not None, "merged_ids must be set when keep_ids is set"
            # recover the merged ids
            assert self.keep_ids.shape[0] == hidden_states.shape[1], f"keep ids shape {self.merged_ids.shape} does not match hidden states sequence length {hidden_states.shape[1]}"
            # scatter tokens based on keep_ids
            tmp = torch.zeros((bsz, self.original_length, dim), device=hidden_states.device, dtype=hidden_states.dtype)
            tmp[:, self.keep_ids, :] = hidden_states
            hidden_states = tmp

            # Handle position embeddings recovery based on their dimensions
            if position_embeddings[0].dim() == 3:
                # 3D case: (B, seq_len, C)
                assert self.keep_ids.shape[0] == position_embeddings[0].shape[1], "The number of retained ids must be equal to the number of tokens in the position embeddings."
                tmp = torch.zeros(bsz, self.original_length, position_embeddings[0].shape[-1], device=position_embeddings[0].device, dtype=position_embeddings[0].dtype)
                tmp[:, self.keep_ids, :] = position_embeddings[0]
                position_embeddings[0] = tmp
                tmp = torch.zeros(bsz, self.original_length, position_embeddings[1].shape[-1], device=position_embeddings[1].device, dtype=position_embeddings[1].dtype)
                tmp[:, self.keep_ids, :] = position_embeddings[1]
                position_embeddings[1] = tmp
            elif position_embeddings[0].dim() == 4:
                # 4D case: handle both Qwen2.5-VL (3, B, seq_len, C) and general (B, H, seq_len, C)
                if position_embeddings[0].shape[0] == 3:
                    # Special Qwen2.5-VL case: (3, B, seq_len, C)
                    assert self.keep_ids.shape[0] == position_embeddings[0].shape[2], "The number of retained ids must be equal to the number of tokens in the position embeddings."
                    tmp = torch.zeros(3, bsz, self.original_length, position_embeddings[0].shape[-1], device=position_embeddings[0].device, dtype=position_embeddings[0].dtype)
                    tmp[:, :, self.keep_ids, :] = position_embeddings[0]
                    position_embeddings[0] = tmp
                    tmp = torch.zeros(3, bsz, self.original_length, position_embeddings[1].shape[-1], device=position_embeddings[1].device, dtype=position_embeddings[1].dtype)
                    tmp[:, :, self.keep_ids, :] = position_embeddings[1]
                    position_embeddings[1] = tmp
                else:
                    # General 4D case: (B, H, seq_len, C)
                    assert self.keep_ids.shape[0] == position_embeddings[0].shape[2], "The number of retained ids must be equal to the number of tokens in the position embeddings."
                    tmp = torch.zeros(bsz, position_embeddings[0].shape[1], self.original_length, position_embeddings[0].shape[-1], device=position_embeddings[0].device, dtype=position_embeddings[0].dtype)
                    tmp[:, :, self.keep_ids, :] = position_embeddings[0]
                    position_embeddings[0] = tmp
                    tmp = torch.zeros(bsz, position_embeddings[1].shape[1], self.original_length, position_embeddings[1].shape[-1], device=position_embeddings[1].device, dtype=position_embeddings[1].dtype)
                    tmp[:, :, self.keep_ids, :] = position_embeddings[1]
                    position_embeddings[1] = tmp
            else:
                raise ValueError(f"Unsupported position_embeddings dimension: {position_embeddings[0].dim()}")
            if attention_mask is not None:
                tmp = torch.zeros(bsz, bsz, self.original_length, self.original_length, device=attention_mask.device, dtype=attention_mask.dtype)
                tmp[:, :, self.keep_ids, :][:, :, :, self.keep_ids] = attention_mask
                attention_mask = tmp

        image_tokens = hidden_states[:, self.image_token_start_index:self.image_token_start_index+self.image_token_length, :]
        if self.merged_ids is not None:
            assert self.merged_ids.shape[0] == self.image_token_length, f"merged ids shape {self.merged_ids.shape} does not match image token length {self.image_token_length}"
            image_tokens = image_tokens[:, self.merged_ids, :]

        top_tokens = torch.cat([
            torch.zeros_like(image_tokens[:, :self.height_stride, :]),
            image_tokens[:, :-self.height_stride, :]
        ], dim=1)

        left_tokens = torch.cat([
            torch.zeros_like(image_tokens[:, :self.width_stride, :]),
            image_tokens[:, :-self.width_stride, :]
        ], dim=1)


        top_tokens_similarity = torch.sum(torch.sign(top_tokens) * torch.sign(image_tokens), dim=-1)
        left_tokens_similarity = torch.sum(torch.sign(left_tokens) * torch.sign(image_tokens), dim=-1)

        max_similarity = torch.max(top_tokens_similarity, left_tokens_similarity)
        max_index = (left_tokens_similarity > top_tokens_similarity).long()

        offsets = (max_index == 0).long() * self.height_stride + (max_index == 1).long() * self.width_stride

        merge_mask = max_similarity / dim >= self.threshold
        merge_mask = merge_mask.flatten()

        margin_mask = torch.ones_like(merge_mask, dtype=torch.bool, device=hidden_states.device)

        # 1D array of token indices [0, 1, 2, …, L-1]
        idx = torch.arange(self.image_token_length, device=hidden_states.device)

        # compute frame IDs
        frame_ids = idx.div(self.frame_stride, rounding_mode='floor')
        # within each frame, compute “row” (height) IDs
        within_frame = idx.fmod(self.frame_stride)
        height_ids = within_frame.div(self.height_stride, rounding_mode='floor')
        # within each row, compute “col” (width) IDs
        within_row = within_frame.fmod(self.height_stride)
        width_ids = within_row.div(self.width_stride, rounding_mode='floor')

        # build a mask of all positions where any of the IDs == 0
        is_margin = (frame_ids == 0) | (height_ids == 0) | (width_ids == 0) if self.num_frames > 1 else (height_ids == 0) | (width_ids == 0)

        # set those positions False
        margin_mask[is_margin] = False

        merge_mask = merge_mask & margin_mask


        offsets = offsets * merge_mask.to(offsets.dtype)
        offsets = offsets.flatten()

        pointer = torch.arange(0, self.image_token_length, device=hidden_states.device)
        assert offsets.shape == pointer.shape, f"Offsets shape {offsets.shape} does not match merged_ids shape {pointer.shape}"
        pointer = (pointer - offsets).clamp(min=0, max=self.image_token_length - 1)

        max_iters = self.patch_num
        for _ in range(max_iters):
            new_pointer = pointer[pointer]  # pointer = pointer(pointer)
            if torch.equal(new_pointer, pointer):
                break
            pointer = new_pointer
        assert torch.equal(pointer, pointer[pointer]), "Pointer should be idempotent after merging"

        self.merged_ids = pointer

        # keep the tokens where merge_mask is False
        keep_mask = ~merge_mask
        # keep ids are local indices within the image tokens
        keep_ids_local = torch.arange(self.image_token_length, device=hidden_states.device)[keep_mask]
        
        # get global keep indices
        keep_ids_global = torch.cat([
            torch.arange(0, self.image_token_start_index, device=hidden_states.device),  # tokens before image tokens
            keep_ids_local + self.image_token_start_index,  # pruned image tokens
            torch.arange(self.image_token_start_index + self.image_token_length, self.original_length, device=hidden_states.device)  # tokens after image tokens
        ])
        self.keep_ids = keep_ids_global

        # prune tokens and position embeddings
        image_tokens = image_tokens[:, keep_ids_local, :].contiguous()
        
        # Prune position embeddings based on their dimensions
        if position_embeddings[0].dim() == 3:
            # 3D case: (B, seq_len, C) - prune along second dimension
            position_embeddings[0] = position_embeddings[0][:, self.keep_ids, :].contiguous()
            position_embeddings[1] = position_embeddings[1][:, self.keep_ids, :].contiguous()
        elif position_embeddings[0].dim() == 4:
            # Check if this is the special Qwen2.5-VL case: (3, B, seq_len, C)
            if position_embeddings[0].shape[0] == 3:
                # Special Qwen2.5-VL case: (3, B, seq_len, C) - prune along third dimension
                position_embeddings[0] = position_embeddings[0][:, :, self.keep_ids, :].contiguous()
                position_embeddings[1] = position_embeddings[1][:, :, self.keep_ids, :].contiguous()
            else:
                # General 4D case: (B, H, seq_len, C) - prune along third dimension
                position_embeddings[0] = position_embeddings[0][:, :, self.keep_ids, :].contiguous()
                position_embeddings[1] = position_embeddings[1][:, :, self.keep_ids, :].contiguous()
        else:
            raise ValueError(f"Unsupported position_embeddings dimension: {position_embeddings[0].dim()}")
        if attention_mask is not None:
            attention_mask = attention_mask[:, :, self.keep_ids, :][:, :, :, self.keep_ids]

        hidden_states = torch.cat([
            hidden_states[:, :self.image_token_start_index, :],  # tokens before image tokens
            image_tokens,  # pruned image tokens
            hidden_states[:, self.image_token_start_index + self.image_token_length:, :]  # tokens after image tokens
        ], dim=1)

        sparsity = torch.sum(merge_mask).item() / merge_mask.numel()
        # print(f"Adaptiv sparsity: {sparsity:.4f}")
        if name not in self.sparsity_dict:
            self.sparsity_dict[name] = AverageMeter()
        self.sparsity_dict[name].update(sparsity)

        return hidden_states, position_embeddings, attention_mask