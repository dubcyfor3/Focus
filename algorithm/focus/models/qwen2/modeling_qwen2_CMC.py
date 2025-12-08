from typing import List, Optional, Tuple, Union, Callable, Dict

import torch
import torch.utils.checkpoint
from torch import nn
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen2.modeling_qwen2 import (
    QWEN2_INPUTS_DOCSTRING,
    apply_rotary_pos_emb,
    logger,
    repeat_kv,
)
import math
from transformers.utils.doc import add_start_docstrings_to_model_forward
from focus.utils import naive_scaled_dot_product_attention
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


def Qwen2DecoderLayer_CMC_forward(
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
        cache_position=cache_position,
        position_embeddings=position_embeddings,
    )
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    return outputs


@add_start_docstrings_to_model_forward(QWEN2_INPUTS_DOCSTRING)
def Qwen2Model_CMC_forward(
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
        self.cmc.post_process()

    return output if return_dict else output.to_tuple()


def Qwen2SdpaAttention_CMC_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    hidden_states = self.cmc(hidden_states, name='qkv_proj') 

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    sliding_window = None
    if (
        self.config.use_sliding_window
        and getattr(self.config, "sliding_window", None) is not None
        and self.layer_idx >= self.config.max_window_layers
    ):
        sliding_window = self.config.sliding_window

    attention_interface: Callable = qwen_attention_forward
    if self.config._attn_implementation != "eager":
        if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
            logger.warning_once(
                "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
        elif self.layer_idx != 27:
            attention_interface = qwen_attention_forward
        else:
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    query_states = self.cmc(query_states, multi_head=True, name='query')
    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=sliding_window,  # main diff with Llama
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()

    attn_output = self.cmc(attn_output, name='o_proj')

    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def qwen_attention_forward(
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
    
    # query = self.cmc(query, multi_head=True, name='query')
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias.to(query.device)
    # attn_weight = self.softmax(attn_weight).to(torch.float32).to(query_states.dtype)
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = self.cmc(attn_weight, multi_head=True, name='attn_weight')
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    attn_output = attn_weight @ value
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_weights = None

    return attn_output, attn_weights

def Qwen2MLP_CMC_forward(self, x):

    x_gate_in = self.cmc(x, name='gate_proj')
    x_up_in = self.cmc(x, name='up_proj')

    gate_output = self.act_fn(self.gate_proj(x_gate_in))
    up_output = self.up_proj(x_up_in)
    down_input = gate_output * up_output

    down_input = self.cmc(down_input, name='down_proj')
    
    down_proj = self.down_proj(down_input)
    # down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    return down_proj