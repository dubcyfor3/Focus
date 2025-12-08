from typing import List, Optional, Tuple, Union, Callable

import torch
import torch.utils.checkpoint
import torch.nn as nn
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from transformers.models.qwen2.modeling_qwen2 import (
    QWEN2_INPUTS_DOCSTRING,
    apply_rotary_pos_emb,
    logger,
    repeat_kv,
)
from transformers.utils.doc import add_start_docstrings_to_model_forward
from focus.utils import naive_scaled_dot_product_attention, TEXT_TOKEN
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from dataclasses import dataclass

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    """Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        mrope_section(`List(int)`):
            Multimodal rope section is for channel dimension of temporal, height and width in rope calculation.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    mrope_section = mrope_section * 2
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

@dataclass
class Qwen2_5_VLCausalLMOutputWithPast(ModelOutput):
    """
    Base class for Qwen2_5_VL causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None

def Qwen2_5_VLDecoderLayer_focus_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    """
    Args:
        hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
        attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
            `(batch, sequence_length)` where padding elements are indicated by 0.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more detail.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
            (see `past_key_values`).
        past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence.
        position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
            Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
            with `head_dim` being the embedding dimension of each attention head.
        kwargs (`dict`, *optional*):
            Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
            into the model
    """

    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    hidden_states, self_attn_weights = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
    )
    hidden_states = residual + hidden_states

        
    if self.self_attn.layer_idx in self.focus.selected_layer and hidden_states.shape[1] > 1:
        self.focus.update_alpha(layer_idx=self.self_attn.layer_idx)
        if self.focus.start_drop:
            hidden_states = self.focus.recover_tokens(hidden_states)
        position_embeddings, attention_mask = self.focus.semantic_concentration(position_embeddings, attention_mask)
        hidden_states = self.focus.drop_tokens(hidden_states)

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    ### start return the updated position embeddings and attention mask
    outputs += (position_embeddings, attention_mask)
    return outputs
    ### end return the updated position embeddings and attention mask


@add_start_docstrings_to_model_forward(QWEN2_INPUTS_DOCSTRING)
def Qwen2_5_VLModel_focus_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **flash_attn_kwargs,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
    if self.gradient_checkpointing and self.training and use_cache:
        logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.")
        use_cache = False
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
    if use_cache and past_key_values is None:
        past_key_values = DynamicCache()
    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device)
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)
    causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions)
    hidden_states = inputs_embeds
    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)
    # decoder layers
    position_embeddings = list(position_embeddings)

    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    for decoder_layer in self.layers[: self.config.num_hidden_layers]:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                causal_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
                position_embeddings,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **flash_attn_kwargs,
            )

        ### start update the attention mask and position embeddings modified by Focus
        position_embeddings = layer_outputs[-2]
        causal_mask = layer_outputs[-1]
        ### end changing position embedding
        hidden_states = layer_outputs[0]
        if output_attentions:
            all_self_attns += (layer_outputs[1],)
    hidden_states = self.norm(hidden_states)
    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)
    output = BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values if use_cache else None,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )
    
    if inputs_embeds.shape[1] > 1:
        self.focus.post_process()

    return output if return_dict else output.to_tuple()


def Qwen2_5_VLSdpaAttention_focus_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    hidden_states = self.focus(hidden_states, name="q_proj")

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
    )

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    causal_mask = attention_mask
    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

    # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    if query_states.device.type == "cuda" and attention_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
    is_causal = True if causal_mask is None and q_len > 1 else False

    # Apply focus processing
    query_states = self.focus(query_states, is_attention=True, name='query')

    if (self.layer_idx in self.focus.selected_layer and q_len > 1) or (self.layer_idx in self.focus.extract_attention_layer and q_len > 1):
        attn_weights = calc_attn_weights(self, query_states, key_states, attention_mask=attention_mask)
        self.focus.set_token_importance(attn_weights)

        if self.layer_idx in self.focus.extract_attention_layer:
            torch.save(self.focus.token_importance, f"output/attn/token_importance_{self.layer_idx}.pt")

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=causal_mask,
        dropout_p=0.0 if not self.training else self.attention_dropout,
        is_causal=is_causal,
    )

    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()

    attn_output = self.focus(attn_output, name="o_proj")
    attn_output = self.o_proj(attn_output)

    return attn_output, None


def Qwen2_5_VLMLP_focus_forward(self, x):

    x_gate_in = self.focus(x, name="gate_proj")
    x_up_in = x_gate_in.clone()


    gate_output = self.act_fn(self.gate_proj(x_gate_in))
    up_output = self.up_proj(x_up_in)
    down_input = gate_output * up_output

    down_input = self.focus(down_input, name="down_proj")
    
    down_proj = self.down_proj(down_input)
    # down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    return down_proj

import math

def Qwen2_5_VLForConditionalGeneration_focus_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
) -> Union[Tuple, "Qwen2_5_VLCausalLMOutputWithPast"]:
    """
    Custom forward function for Qwen2.5 VL with focus metadata recording.
    This wraps the original forward and adds metadata calculation before self.model() call.
    """
    from transformers.modeling_outputs import BaseModelOutputWithPast
    from torch.nn import CrossEntropyLoss
    
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if inputs_embeds is None:
        inputs_embeds = self.model.embed_tokens(input_ids)
        if pixel_values is not None:
            pixel_values = pixel_values.type(self.visual.dtype)
            image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
            n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
            n_image_features = image_embeds.shape[0]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )

            mask = input_ids == self.config.image_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            image_mask = mask_expanded.to(inputs_embeds.device)

            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
            video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
            n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
            n_video_features = video_embeds.shape[0]
            if n_video_tokens != n_video_features:
                raise ValueError(
                    f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                )

            mask = input_ids == self.config.video_token_id
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
            video_mask = mask_expanded.to(inputs_embeds.device)

            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if attention_mask is not None:
            attention_mask = attention_mask.to(inputs_embeds.device)

    # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
    if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
        # calculate RoPE index once per generation in the pre-fill stage only
        if (
            (cache_position is not None and cache_position[0] == 0)
            or self.rope_deltas is None
            or (past_key_values is None or past_key_values.get_seq_length() == 0)
        ):
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts,
                attention_mask,
            )
            self.rope_deltas = rope_deltas
        # then use the prev pre-calculated rope-deltas to get the correct position ids
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            delta = (
                (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                if cache_position is not None
                else 0
            )
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            if cache_position is not None:  # otherwise `deltas` is an int `0`
                delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

    ### FOCUS METADATA CALCULATION START ###
    # Calculate metadata for focus before calling self.model()
    if hasattr(self, 'focus') and input_ids is not None:
        # Get image token positions
        image_token_id = self.config.image_token_id
        image_token_indices = torch.where(input_ids == image_token_id)
        
        if len(image_token_indices[0]) > 0:
            # Calculate patch information for Qwen2.5 VL using image_grid_thw
            if image_grid_thw is not None and len(image_grid_thw) > 0:
                # image_grid_thw format: [frame, height, width]
                # For images, frame = 1, so we use the first (and only) grid
                grid_t, grid_h, grid_w = image_grid_thw[0]  # [1, h, w]
                
                # Patch dimensions are half of the grid dimensions
                patch_height = grid_h // 2
                patch_width = grid_w // 2
                patch_num = patch_height * patch_width
                n_frames = grid_t  # Should be 1 for images
            else:
                # Fallback to old calculation if image_grid_thw is not available
                if hasattr(self.config, 'mm_spatial_pool_mode') and self.config.mm_spatial_pool_mode == "bilinear":
                    patch_size = math.ceil(self.visual.num_patches_per_side / 2)
                else:
                    patch_size = self.visual.num_patches_per_side // 2
                
                patch_num = patch_size * (patch_size + 1)
                patch_height = patch_size
                patch_width = patch_size + 1
                n_frames = 1
            
            batch_size = input_ids.shape[0]
            
            # Calculate metadata
            image_token_length = len(image_token_indices[0])
            frame_stride = patch_height * patch_width
            height_stride = patch_width
            width_stride = 1
            
            image_token_start_index = image_token_indices[1][0] if len(image_token_indices[1]) > 0 else 0
            image_token_end_index = image_token_start_index + image_token_length - 1
            original_length = input_ids.shape[1]
            
            # Create patch type tensor
            patch_type = [TEXT_TOKEN] * image_token_start_index + list(range(patch_num)) * n_frames + [TEXT_TOKEN] * (original_length - image_token_end_index - 1)
            patch_type = torch.tensor([patch_type], device=input_ids.device)
            
            n_frames = n_frames.item()
            patch_height = patch_height.item()
            patch_width = patch_width.item()
            # Prepare focus with metadata
            self.focus.prepare(patch_type, n_frames, patch_height, patch_width, frame_stride, height_stride, width_stride, image_token_start_index, image_token_end_index, image_token_length, original_length)
    ### FOCUS METADATA CALCULATION END ###

    outputs = self.model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
    )

    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)

    loss = None
    if labels is not None:
        # Upcast to float if we need to compute the loss to avoid potential precision issues
        logits = logits.float()
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Qwen2_5_VLCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.rope_deltas,
    )

def prepare_inputs_labels_for_multimodal_qwen2_5_vl(self, input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities=["image"], image_sizes=None):
    """
    Prepare inputs and labels for multimodal processing with Qwen2.5 VL.
    This function handles the metadata recording for focus.
    """
    # Get vision tower
    vision_tower = self.get_vision_tower()
    
    if vision_tower is None or images is None or input_ids.shape[1] == 1:
        return input_ids, position_ids, attention_mask, past_key_values, None, labels

    if isinstance(modalities, str):
        modalities = [modalities]

    # Handle different image formats
    if type(images) is list or images.ndim == 5:
        if type(images) is list:
            images = torch.stack(images, dim=0)
        if images.ndim == 5:
            images = images.reshape(-1, *images.shape[2:])
    else:
        images = [images]

    # Process images through vision tower
    image_features = []
    for image in images:
        if image.dtype != vision_tower.dtype:
            image = image.to(dtype=vision_tower.dtype)
        image_feature = vision_tower(image)
        image_features.append(image_feature)

    # Get image token indices
    image_token_id = self.config.image_token_id
    video_token_id = getattr(self.config, 'video_token_id', None)
    
    # Find image token positions
    image_token_indices = torch.where(input_ids == image_token_id)
    
    if len(image_token_indices[0]) == 0:
        return input_ids, position_ids, attention_mask, past_key_values, None, labels

    # Calculate metadata for focus
    batch_size = input_ids.shape[0]
    num_images = len(image_features)
    
    # For Qwen2.5 VL, we need to determine patch information
    # This is similar to LLaVA but adapted for Qwen2.5 VL structure
    if hasattr(self.config, 'mm_spatial_pool_mode') and self.config.mm_spatial_pool_mode == "bilinear":
        patch_size = math.ceil(vision_tower.num_patches_per_side / 2)
    else:
        patch_size = vision_tower.num_patches_per_side // 2
    
    patch_num = (patch_size * (patch_size + 1)).item()
    
    assert batch_size == 1
    assert num_images == 1
    
    image_token_length = image_features[0].shape[0].item()
    n_frames = image_token_length // patch_num
    patch_height = patch_size
    patch_width = patch_size + 1
    frame_stride = patch_height * patch_width
    height_stride = patch_width
    width_stride = 1
    
    image_token_start_index = torch.where(input_ids[0] == image_token_id)[0]
    image_token_end_index = image_token_start_index + image_token_length - 1
    original_length = input_ids[0].shape[0] + image_token_length - 1
    
    # Create patch type tensor
    patch_type = [TEXT_TOKEN] * image_token_start_index + list(range(patch_num)) * n_frames + [TEXT_TOKEN] * (original_length - image_token_end_index - 1)
    patch_type = torch.tensor([patch_type], device=input_ids.device)
    
    # Prepare focus with metadata
    if hasattr(self, 'focus'):
        self.focus.prepare(patch_type, n_frames, patch_height, patch_width, frame_stride, height_stride, width_stride, image_token_start_index, image_token_end_index, image_token_length, original_length)
    elif hasattr(self, 'cmc'):
        self.cmc.prepare(patch_type, n_frames, patch_height, patch_width, frame_stride, height_stride, width_stride, image_token_start_index, image_token_end_index, image_token_length, original_length)
    elif hasattr(self, 'adaptiv'):
        self.adaptiv.prepare(patch_type, n_frames, patch_height, patch_width, frame_stride, height_stride, width_stride, image_token_start_index, image_token_end_index, image_token_length, original_length)

    return input_ids, position_ids, attention_mask, past_key_values, None, labels

def calc_attn_weights(
    self,
    query_states,
    key_states,
    attention_mask=None,
):
    is_causal = None
    scale = None
     
    query = query_states.clone()
    key = key_states.clone()

    dropout_p = 0.0 #if not self.training else self.attention_dropout
    # if hasattr(self, "num_key_value_groups"):
    #     key = repeat_kv(key, self.num_key_value_groups)

    causal_mask = attention_mask
    if attention_mask is not None:
        causal_mask = causal_mask[:, :, :, : key.shape[-2]]
    
    query = query.contiguous()
    key = key.contiguous()


    if is_causal is None:
        is_causal = causal_mask is None and query.shape[2] > 1

    if torch.jit.is_tracing() and isinstance(is_causal, torch.Tensor):
        is_causal = is_causal.item()

    attn_mask = causal_mask
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale

    if attn_mask is not None:
        attn_bias = torch.zeros_like(attn_mask, dtype=query.dtype)
    else:
        attn_bias = torch.zeros(L, S, dtype=query.dtype)

    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), -1e4)
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), -1e4)
        else:
            attn_bias += attn_mask
    
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias.to(query.device)
    attn_weight = torch.softmax(attn_weight, dim=-1)

    return attn_weight

def qwen2_5_vl_attention_forward(
    self,
    query_states,
    key_states,
    value_states,
    attention_mask=None,
    scaling=None,
    dropout=0.0,
    **kwargs,
):
            
    is_causal = None
    scale =self.scaling
     
    query = query_states.clone()
    key = key_states.clone()
    value = value_states.clone()
    dropout_p = 0.0 #if not self.training else self.attention_dropout
    if hasattr(self, "num_key_value_groups"):
        key = repeat_kv(key, self.num_key_value_groups)
        value = repeat_kv(value, self.num_key_value_groups)
    causal_mask = attention_mask
    if attention_mask is not None:
        causal_mask = causal_mask[:, :, :, : key.shape[-2]]
    
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    if is_causal is None:
        is_causal = causal_mask is None and query.shape[2] > 1

    if torch.jit.is_tracing() and isinstance(is_causal, torch.Tensor):
        is_causal = is_causal.item()
    attn_mask = causal_mask
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), -1e4)
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), -1e4)
        else:
            attn_bias += attn_mask
    
    # query = self.focus(query, is_attention=True, name='query')
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias.to(query.device)
    # attn_weight = self.softmax(attn_weight).to(torch.float32).to(query_states.dtype)
    attn_weight = torch.softmax(attn_weight, dim=-1)
    # attn_weight = self.focus(attn_weight, is_attention=True, name='attn_weight')
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    attn_output = attn_weight @ value
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_weights = attn_weight

    return attn_output, attn_weights
