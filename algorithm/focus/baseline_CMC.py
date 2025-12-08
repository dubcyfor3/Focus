from typing import List

import torch
from torch import nn

from focus.utils import AverageMeter, save_result_to_csv

TEXT_TOKEN = -1
IGNORE_TOKEN = -2


class CMC(nn.Module):
    def __init__(self, interval_size, threshold, threshold_query, threshold_score, model_name, dataset_name, trace_meta_dir, write_sparsity=False, simplified=False):
        super(CMC, self).__init__()
        self.interval_size = interval_size
        self.I_frame_position = 3
        self.threshold = threshold
        self.base_threshold = threshold
        self.threshold_query = threshold_query
        self.threshold_score = threshold_score
        self.sparsity_dict = {'fc': AverageMeter(), 'query': AverageMeter(), 'attn_score': AverageMeter()}
        self.trace_meta_dir = trace_meta_dir
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.write_sparsity = write_sparsity
        self.simplified = simplified

        self.training = False
        self.limit = 10
        self.cur_idx = 0

    def post_process(self):

        self.cur_idx += 1
        if self.cur_idx == self.limit:

            output_sparsity_dict = {'Model': self.model_name, 'Dataset': self.dataset_name, 
            'linear_sparsity': self.sparsity_dict['fc'].avg, 
            'query_sparsity': self.sparsity_dict['query'].avg, 
            'attn_score_sparsity': self.sparsity_dict['attn_score'].avg}
            

            if self.write_sparsity: 
                save_result_to_csv(output_sparsity_dict, f'{self.trace_meta_dir}/cmc_sparsity.csv')
                print(f"Saved CMC sparsity to {f'{self.trace_meta_dir}/cmc_sparsity.csv'}")
            self.cur_idx = 0



    def prepare(self, patch_type, patch_height, patch_width, image_token_start_index, image_token_end_index, image_token_length, original_length, input_embeds: torch.Tensor):
        self.patch_type = patch_type
        self.patch_num = patch_height * patch_width
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.image_token_start_index = image_token_start_index
        self.image_token_end_index = image_token_end_index
        self.image_token_length = image_token_length
        self.original_length = original_length

        I_frame_idx, min_position, mask_non_informative, num_non_informative_tokens, num_informative_tokens = self.detect_non_informative(input_embeds)

        self.I_frame_idx = I_frame_idx
        self.non_informative_table = min_position
        self.non_informative_mask = mask_non_informative
        self.num_non_informative_tokens = num_non_informative_tokens
        self.num_informative_tokens = num_informative_tokens
        self.sparsity = num_non_informative_tokens / (num_non_informative_tokens + num_informative_tokens)
        assert self.sparsity <= 0.875, f"Sparsity {self.sparsity} exceeds 0.875"
        self.sparsity_dict['fc'].update(self.sparsity.item())

        return

    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor, multi_head=False, name=None):

        if multi_head:
            bsz, num_heads, seq_len, head_dim = hidden_states.shape
            if seq_len == 1:
                return hidden_states
            
            I_frame_idx, min_position, mask_non_informative, num_non_informative_tokens, num_informative_tokens = self.detect_non_informative(hidden_states)

            sparsity = num_non_informative_tokens / (num_non_informative_tokens + num_informative_tokens)
            assert sparsity <= 0.875
            if hidden_states.shape[-1] == hidden_states.shape[-2]:
                self.sparsity_dict['attn_score'].update(sparsity.item())
            else:
                self.sparsity_dict['query'].update(sparsity.item())

            hidden_states = self.approximate_non_informative(hidden_states, I_frame_idx, min_position, mask_non_informative)

        else:
            bsz, seq_len, hidden_dim = hidden_states.shape
            if seq_len == 1:
                return hidden_states
            
            hidden_states = self.approximate_non_informative(hidden_states, self.I_frame_idx, self.non_informative_table, self.non_informative_mask)

        
        return hidden_states

    

    def detect_non_informative(self, hidden_states: torch.Tensor):

        if hidden_states.dim() == 3:
            # [1, seq_len, hidden_dim]
            is_multihead = False
            image_tokens = hidden_states[:, self.image_token_start_index:self.image_token_start_index+self.image_token_length, :]  # [1, img_len, dim]
            image_tokens = image_tokens.squeeze(0)  # [img_len, dim]
            image_tokens = image_tokens.view(-1, self.patch_num, image_tokens.size(-1))  # [frames, patches, dim]
        elif hidden_states.dim() == 4:
            # [1, num_heads, seq_len, head_dim]
            is_multihead = True
            
            bsz, num_heads, seq_len, head_dim = hidden_states.shape
            if seq_len == head_dim:
                # softmax attention matrix
                threshold = self.threshold_score
            else:
                threshold = self.threshold_query
            image_tokens = hidden_states[:, :, self.image_token_start_index:self.image_token_start_index+self.image_token_length, :]  # [1, heads, img_len, dim]
            image_tokens = image_tokens.squeeze(0)  # [heads, img_len, dim]
            image_tokens = image_tokens.view(num_heads, -1, self.patch_num, head_dim)  # [heads, frames, patches, dim]
        else:
            raise ValueError("Unsupported hidden_states shape")

        num_frames = self.image_token_length // self.patch_num
        num_intervals = (num_frames + self.interval_size - 1) // self.interval_size

        if is_multihead:
            # Pad per head
            image_tokens, pad_size = pad_to_multiple_of(image_tokens, self.interval_size, dim=1)  # [heads, padded_frames, patches, dim]
        else:
            image_tokens, pad_size = pad_to_multiple_of(image_tokens, self.interval_size, dim=0)  # [padded_frames, patches, dim]

        num_frames_no_pad = num_frames
        num_frames += pad_size

        I_frame_idx = torch.arange(self.I_frame_position, num_frames, self.interval_size, device=hidden_states.device)  # [intervals]

        if is_multihead:
            I_frame_tokens = image_tokens[:, I_frame_idx]  # [heads, intervals, patches, dim]
            I_frame_tokens = I_frame_tokens.unsqueeze(2).expand(-1, -1, self.interval_size, -1, -1)  # [heads, intervals, interval_size, patches, dim]
            I_frame_tokens = I_frame_tokens.contiguous().view(num_heads, -1, self.patch_num, head_dim)  # [heads, total_ref_frames, patches, dim]
            P_frame_tokens = image_tokens.view(num_heads, -1, self.patch_num, head_dim)  # same shape

            if self.simplified:
                SAD_tensor = torch.sum(torch.abs(P_frame_tokens - I_frame_tokens), dim=-1)
                min_SAD_tensor = SAD_tensor.view(num_heads, num_intervals, self.interval_size, self.patch_num)
                min_position = torch.arange(self.patch_num, device=hidden_states.device).unsqueeze(0).expand(num_heads, num_intervals, self.interval_size, -1)  # [heads, intervals, interval_size, patches]

            else:
                # Process interval by interval to reduce memory usage
                min_SAD_list = []
                min_position_list = []
                
                for interval_idx in range(num_intervals):
                    interval_start = interval_idx * self.interval_size
                    interval_end = interval_start + self.interval_size
                    
                    # Extract P and I tokens for this interval
                    P_interval = P_frame_tokens[:, interval_start:interval_end, :, :]  # [heads, interval_size, patches, dim]
                    I_interval = I_frame_tokens[:, interval_start:interval_end, :, :]  # [heads, interval_size, patches, dim]
                    
                    # Process head by head to further reduce memory
                    interval_min_SAD_list = []
                    interval_min_pos_list = []
                    
                    for head_idx in range(num_heads):
                        P_head = P_interval[head_idx]  # [interval_size, patches, dim]
                        I_head = I_interval[head_idx]  # [interval_size, patches, dim]
                        
                        # Compute cdist for this head and interval
                        SAD_head = torch.cdist(P_head.float(), I_head.float(), p=1)  # [interval_size, patches, patches]
                        min_SAD_head, min_pos_head = torch.min(SAD_head, dim=-1)  # [interval_size, patches]
                        
                        interval_min_SAD_list.append(min_SAD_head)
                        interval_min_pos_list.append(min_pos_head)
                    
                    # Stack heads for this interval
                    min_SAD_list.append(torch.stack(interval_min_SAD_list, dim=0))  # [heads, interval_size, patches]
                    min_position_list.append(torch.stack(interval_min_pos_list, dim=0))  # [heads, interval_size, patches]
                
                # Stack intervals
                min_SAD_tensor = torch.stack(min_SAD_list, dim=1)  # [heads, intervals, interval_size, patches]
                min_position = torch.stack(min_position_list, dim=1)  # [heads, intervals, interval_size, patches]

            mask_non_informative = min_SAD_tensor < threshold * head_dim

            mask_non_informative_count = mask_non_informative.clone().view(num_heads, num_intervals * self.interval_size, self.patch_num)
            if pad_size >= (self.interval_size - self.I_frame_position):
                # mask the tail interval with False, assume they are informative
                mask_non_informative_count[:, -self.interval_size:, :] = False
            mask_non_informative_count = mask_non_informative_count[:, :num_frames_no_pad, :]  # remove padded frames
            
            total_informative = mask_non_informative_count.numel() - mask_non_informative_count.sum()
            if pad_size >= (self.interval_size - self.I_frame_position):
                total_informative -= self.patch_num * num_heads
            total_informative += num_intervals * self.patch_num * num_heads

            num_non_informative = self.image_token_length * num_heads - total_informative

            return I_frame_idx, min_position, mask_non_informative, num_non_informative, total_informative

        else:
            # all_zero_I_frame = num_frames % self.interval_size < self.I_frame_position

            I_frame_tokens = image_tokens[I_frame_idx]
            I_frame_tokens = I_frame_tokens.unsqueeze(1).expand(-1, self.interval_size, -1, -1).contiguous()
            I_frame_tokens = I_frame_tokens.view(num_intervals * self.interval_size, self.patch_num, image_tokens.size(-1))
            P_frame_tokens = image_tokens

            if self.simplified:
                SAD_tensor = torch.sum(torch.abs(P_frame_tokens - I_frame_tokens), dim=-1)
                min_SAD_tensor = SAD_tensor.view(num_intervals, self.interval_size, self.patch_num)
                min_position = torch.arange(self.patch_num, device=hidden_states.device).unsqueeze(0).expand(num_intervals, self.interval_size, -1)
            else:
                # Process interval by interval to reduce memory usage
                min_SAD_list = []
                min_position_list = []
                
                for interval_idx in range(num_intervals):
                    interval_start = interval_idx * self.interval_size
                    interval_end = interval_start + self.interval_size
                    
                    # Extract P and I tokens for this interval
                    P_interval = P_frame_tokens[interval_start:interval_end, :, :]  # [interval_size, patches, dim]
                    I_interval = I_frame_tokens[interval_start:interval_end, :, :]  # [interval_size, patches, dim]
                    
                    # Compute cdist for this interval only
                    SAD_interval = torch.cdist(P_interval.float(), I_interval.float(), p=1)  # [interval_size, patches, patches]
                    min_SAD_interval, min_pos_interval = torch.min(SAD_interval, dim=-1)  # [interval_size, patches]
                    
                    min_SAD_list.append(min_SAD_interval)
                    min_position_list.append(min_pos_interval)
                
                # Stack intervals
                min_SAD_tensor = torch.stack(min_SAD_list, dim=0)  # [intervals, interval_size, patches]
                min_position = torch.stack(min_position_list, dim=0)  # [intervals, interval_size, patches]

            mask_non_informative = min_SAD_tensor < self.threshold * image_tokens.size(-1)

            mask_non_informative_count = mask_non_informative.clone().view(num_intervals * self.interval_size, self.patch_num)
            if pad_size >= (self.interval_size - self.I_frame_position):
                # mask the tail interval with False, assume they are informative
                mask_non_informative_count[-self.interval_size:, :] = False
            mask_non_informative_count = mask_non_informative_count[:num_frames_no_pad, :]  # remove padded frames
            informative = mask_non_informative_count.numel() - mask_non_informative_count.sum()
            if pad_size >= (self.interval_size - self.I_frame_position):
                informative -= self.patch_num
            informative += num_intervals * self.patch_num

            non_informative = self.image_token_length - informative
            return I_frame_idx, min_position, mask_non_informative, non_informative, informative

    
    def approximate_non_informative(self, hidden_states: torch.Tensor, I_frame_idx, non_informative_table, non_informative_mask):
        if hidden_states.dim() == 3:
            # [1, seq_len, hidden_dim]
            is_multihead = False
            image_tokens = hidden_states[:, self.image_token_start_index:self.image_token_start_index+self.image_token_length, :].clone()  # [1, img_len, dim]
            image_tokens = image_tokens.squeeze(0)  # [img_len, dim]
            num_frames = image_tokens.size(0) // self.patch_num
            image_tokens = image_tokens.view(num_frames, self.patch_num, image_tokens.size(-1))  # [frames, patches, dim]
            image_tokens, pad_size = pad_to_multiple_of(image_tokens, self.interval_size, dim=0)
            num_intervals = image_tokens.size(0) // self.interval_size

            I_frame_tokens = image_tokens[I_frame_idx]  # [intervals, patches, dim]
            I_frame_tokens = I_frame_tokens.unsqueeze(1).expand(-1, self.interval_size, -1, -1).contiguous()  # [intervals, interval_size, patches, dim]
            index = non_informative_table.unsqueeze(-1).expand(-1, -1, -1, I_frame_tokens.size(-1))  # [intervals, interval_size, patches, dim]
            copied_I_frame_tokens = torch.gather(I_frame_tokens, 2, index)  # [intervals, interval_size, patches, dim]
            mask = non_informative_mask.unsqueeze(-1).expand_as(copied_I_frame_tokens)  # same shape
            image_tokens = image_tokens.view(num_intervals, self.interval_size, self.patch_num, -1)
            image_tokens = torch.where(mask, copied_I_frame_tokens, image_tokens)  # replace with I-frame where needed

            image_tokens = image_tokens.view(-1, self.patch_num, image_tokens.size(-1))
            if pad_size != 0:
                image_tokens = image_tokens[:-pad_size]
            image_tokens = image_tokens.view(-1, image_tokens.size(-1))  # [image_token_length, dim]
            hidden_states[:, self.image_token_start_index:self.image_token_start_index+self.image_token_length, :] = image_tokens.unsqueeze(0)
            return hidden_states

        elif hidden_states.dim() == 4:
            # [1, num_heads, seq_len, head_dim]
            is_multihead = True
            b, h, s, d = hidden_states.shape
            assert b == 1, "Only batch size = 1 is supported"
            image_tokens = hidden_states[:, :, self.image_token_start_index:self.image_token_start_index+self.image_token_length, :]  # [1, heads, img_len, dim]
            image_tokens = image_tokens.squeeze(0)  # [heads, img_len, dim]
            num_frames = image_tokens.size(1) // self.patch_num
            image_tokens = image_tokens.view(h, num_frames, self.patch_num, d)  # [heads, frames, patches, dim]
            image_tokens, pad_size = pad_to_multiple_of(image_tokens, self.interval_size, dim=1)
            num_intervals = image_tokens.size(1) // self.interval_size

            I_frame_tokens = image_tokens[:, I_frame_idx]  # [heads, intervals, patches, dim]
            I_frame_tokens = I_frame_tokens.unsqueeze(2).expand(-1, -1, self.interval_size, -1, -1).contiguous()  # [heads, intervals, interval_size, patches, dim]
            index = non_informative_table.unsqueeze(-1).expand(h, -1, -1, -1, d)  # [heads, intervals, interval_size, patches, dim]
            copied_I_frame_tokens = torch.gather(I_frame_tokens, 3, index)  # gather along patch dim
            mask = non_informative_mask.unsqueeze(-1).expand(h, -1, -1, -1, d)  # [heads, intervals, interval_size, patches, dim]
            image_tokens = image_tokens.view(h, num_intervals, self.interval_size, self.patch_num, d)
            image_tokens = torch.where(mask, copied_I_frame_tokens, image_tokens)

            image_tokens = image_tokens.view(h, -1, self.patch_num, d)
            if pad_size != 0:
                image_tokens = image_tokens[:, :-pad_size]
            image_tokens = image_tokens.view(h, -1, d)  # [heads, image_token_length, dim]

            # assign back to original hidden_states
            hidden_states[:, :, self.image_token_start_index:self.image_token_start_index+self.image_token_length, :] = image_tokens.unsqueeze(0)
            return hidden_states

        else:
            raise ValueError("Unsupported hidden_states shape")

import einops as ein

def cdist_l1(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

    # Merge extra dim if input is 4D: (b, p, l, r) -> (b*p, l, r)
    if x.dim() == 4:
        reshaped = True
        b, p, l, r = x.shape
        x = x.reshape(b * p, l, r)
        y = y.reshape(b * p, l, r)
    else:
        reshaped = False

    if x.dtype is torch.float16:
        x = ein.rearrange(x, "bp l r -> bp l () r")
        y = ein.rearrange(y, "bp l r -> bp () l r")
        dist = (x - y).abs().sum(dim=-1)
    else:
        dist = torch.cdist(x, y, p=1)

    # Restore extra dim if input was 4D
    if 'p' in locals():
        dist = dist.reshape(b, p, l, l)
    # restore x and y to original shape
    if reshaped:
        x = x.reshape(b, p, l, r)
        y = y.reshape(b, p, l, r)
    return dist

def pad_to_multiple_of(tensor: torch.Tensor, size: int, dim: int):
    """
    Pad the input tensor to a multiple of size along the specified dimension.
    """
    if tensor.size(dim) % size == 0:
        return tensor, 0
    pad_size = size - tensor.size(dim) % size
    tensor = torch.cat([tensor, tensor.new_zeros(tensor.size()[:dim] + (pad_size,) + tensor.size()[dim+1:])], dim=dim)
    return tensor, pad_size

if __name__ == '__main__':
    H=13
    W=14
    F=64
    # fp 16 tensor
    input_embeds = torch.randn(1, 12, 13*14*64, 64).to(torch.float16)

    cmc = CMC(8, 1.0)
    cmc.prepare('image', H, W, 0, H*W*F, H*W*F, H*W*W, input_embeds)
    cmc(input_embeds, True)