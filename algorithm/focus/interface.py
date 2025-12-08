# common imports
from types import MethodType
from typing import Callable
import ast

import torch
import torch.nn as nn
from accelerate.hooks import add_hook_to_module
from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM

import pandas as pd

# Focus methods
from focus.main import Focus
from focus.baseline_CMC import CMC
from focus.baseline_adaptiv import Adaptiv
from focus.models.llava_video.modeling_llava_video import (
    prepare_inputs_labels_for_multimodal_get_patch_type,
)
from focus.models.llava_onevision.modeling_llava_onevision import (
    prepare_inputs_labels_for_multimodal_get_patch_type_onevision,
)
from focus.models.minicpmv.modeling_minicpmv import (
    get_vllm_embedding
)
from focus.models.llava_video.modeling_llava_video_CMC import (
    prepare_inputs_labels_for_multimodal_CMC,
)
from focus.models.llava_onevision.modeling_llava_onevision_CMC import (
    prepare_inputs_labels_for_multimodal_onevision_CMC,
)
from focus.models.minicpmv.modeling_minicpmv_CMC import (
    get_vllm_embedding_CMC
)
from focus.models.llava_video.modeling_llava_video_adaptiv import (
    prepare_inputs_labels_for_multimodal_get_patch_type_adaptiv,
)
from focus.models.llava_onevision.modeling_llava_onevision_adaptiv import (
    prepare_inputs_labels_for_multimodal_get_patch_type_onevision_adaptiv,
)
from focus.models.minicpmv.modeling_minicpmv_adaptiv import (
    get_vllm_embedding_adaptiv,
)

from focus.models.qwen2.modeling_qwen2 import (
    Qwen2DecoderLayer_focus_forward,
    Qwen2Model_focus_forward,
    Qwen2SdpaAttention_focus_forward,
    Qwen2MLP_focus_forward
)
from focus.models.qwen2.modeling_qwen2_CMC import (
    Qwen2DecoderLayer_CMC_forward,
    Qwen2Model_CMC_forward,
    Qwen2SdpaAttention_CMC_forward,
    Qwen2MLP_CMC_forward,
)
from focus.models.qwen2.modeling_qwen2_adaptiv import (
    Qwen2DecoderLayer_adaptiv_forward,
    Qwen2Model_adaptiv_forward,
    Qwen2SdpaAttention_adaptiv_forward,
    Qwen2MLP_adaptiv_forward
)
from focus.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLDecoderLayer_focus_forward,
    Qwen2_5_VLModel_focus_forward,
    Qwen2_5_VLSdpaAttention_focus_forward,
    Qwen2_5_VLMLP_focus_forward,
    Qwen2_5_VLForConditionalGeneration_focus_forward
)
from focus.models.qwen2_5_vl.modeling_qwen2_5_vl_adaptiv import (
    Qwen2_5_VLDecoderLayer_adaptiv_forward,
    Qwen2_5_VLModel_adaptiv_forward,
    Qwen2_5_VLSdpaAttention_adaptiv_forward,
    Qwen2_5_VLMLP_adaptiv_forward,
    Qwen2_5_VLForConditionalGeneration_adaptiv_forward
)
from focus.utils import get_attr_by_name

# model types
from transformers import LlavaNextVideoForConditionalGeneration, PreTrainedModel, MllamaForConditionalGeneration
try:
    from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
except ImportError:
    # Handle case where Qwen2VLForConditionalGeneration is not available
    Qwen2VLForConditionalGeneration = None

def apply_focus(model, cli_args=None):
    """
    Apply Focus to the model

    Args:
        model: the model to apply Focus to
        cli_args: the command line arguments
    """


    # LlavaVideo Model and LlavaOneVision Model
    if isinstance(model, LlavaQwenForCausalLM):

        if cli_args.focus:
            if cli_args.model == "llava_onevision":
                model.prepare_inputs_labels_for_multimodal = MethodType(prepare_inputs_labels_for_multimodal_get_patch_type_onevision, model)
            elif cli_args.model == "llava_vid":
                model.prepare_inputs_labels_for_multimodal = MethodType(prepare_inputs_labels_for_multimodal_get_patch_type, model)
            else:
                raise ValueError(f"Invalid model type: {cli_args.model}")

            llm_forward = Qwen2Model_focus_forward
            decoder_forward = Qwen2DecoderLayer_focus_forward
            attention_forward = Qwen2SdpaAttention_focus_forward
            mlp_forward = Qwen2MLP_focus_forward
        elif cli_args.CMC:
            if cli_args.model == "llava_onevision":
                model.prepare_inputs_labels_for_multimodal = MethodType(prepare_inputs_labels_for_multimodal_onevision_CMC, model)
            elif cli_args.model == "llava_vid":
                model.prepare_inputs_labels_for_multimodal = MethodType(prepare_inputs_labels_for_multimodal_CMC, model)
            else:
                raise ValueError(f"Invalid model type: {cli_args.model}")

            llm_forward = Qwen2Model_CMC_forward
            decoder_forward = Qwen2DecoderLayer_CMC_forward
            attention_forward = Qwen2SdpaAttention_CMC_forward
            mlp_forward = Qwen2MLP_CMC_forward

        elif cli_args.adaptiv:
            if cli_args.model == "llava_onevision":
                model.prepare_inputs_labels_for_multimodal = MethodType(prepare_inputs_labels_for_multimodal_get_patch_type_onevision_adaptiv, model)
            elif cli_args.model == "llava_vid":
                model.prepare_inputs_labels_for_multimodal = MethodType(prepare_inputs_labels_for_multimodal_get_patch_type_adaptiv, model)
            else:
                raise ValueError(f"Invalid model type: {cli_args.model}")

            llm_forward = Qwen2Model_adaptiv_forward
            decoder_forward = Qwen2DecoderLayer_adaptiv_forward
            attention_forward = Qwen2SdpaAttention_adaptiv_forward
            mlp_forward = Qwen2MLP_adaptiv_forward

        llm_key = "model"
        decoder_key = "layers"
        attention_key = "self_attn"
        mlp_key = "mlp"
        
        num_layers, hidden_dim, intermediate_dim, gqa_group_size = get_model_dimensions(model, llm_key, decoder_key, mlp_key)


    # MiniCPM Model
    elif model.config.architectures[0] == "MiniCPMV":
        if cli_args.focus:
            model.get_vllm_embedding = MethodType(get_vllm_embedding, model)
            llm_forward = Qwen2Model_focus_forward
            decoder_forward = Qwen2DecoderLayer_focus_forward
            attention_forward = Qwen2SdpaAttention_focus_forward
            mlp_forward = Qwen2MLP_focus_forward

        elif cli_args.CMC:
            model.get_vllm_embedding = MethodType(get_vllm_embedding_CMC, model)
            llm_forward = Qwen2Model_CMC_forward
            decoder_forward = Qwen2DecoderLayer_CMC_forward
            attention_forward = Qwen2SdpaAttention_CMC_forward
            mlp_forward = Qwen2MLP_CMC_forward

        elif cli_args.adaptiv:
            model.get_vllm_embedding = MethodType(get_vllm_embedding_adaptiv, model)
            llm_forward = Qwen2Model_adaptiv_forward
            decoder_forward = Qwen2DecoderLayer_adaptiv_forward
            attention_forward = Qwen2SdpaAttention_adaptiv_forward
            mlp_forward = Qwen2MLP_adaptiv_forward

        llm_key = "llm.model"
        decoder_key = "layers"
        attention_key = "self_attn"
        mlp_key = "mlp"

        num_layers, hidden_dim, intermediate_dim, gqa_group_size = get_model_dimensions(model, llm_key, decoder_key, mlp_key)

    # Qwen2.5 VL Model
    elif Qwen2_5_VLForConditionalGeneration is not None and isinstance(model, Qwen2_5_VLForConditionalGeneration):
        if cli_args.focus:
            # Replace the main forward function with custom forward that includes metadata calculation
            model.forward = MethodType(Qwen2_5_VLForConditionalGeneration_focus_forward, model)
            llm_forward = Qwen2_5_VLModel_focus_forward
            decoder_forward = Qwen2_5_VLDecoderLayer_focus_forward
            attention_forward = Qwen2_5_VLSdpaAttention_focus_forward
            mlp_forward = Qwen2_5_VLMLP_focus_forward
        elif cli_args.adaptiv:
            # Replace the main forward function with custom forward that includes metadata calculation
            model.forward = MethodType(Qwen2_5_VLForConditionalGeneration_adaptiv_forward, model)
            llm_forward = Qwen2_5_VLModel_adaptiv_forward
            decoder_forward = Qwen2_5_VLDecoderLayer_adaptiv_forward
            attention_forward = Qwen2_5_VLSdpaAttention_adaptiv_forward
            mlp_forward = Qwen2_5_VLMLP_adaptiv_forward
        else:
            raise NotImplementedError("Only focus and adaptiv are supported for Qwen2.5 VL models")

        llm_key = "model"
        decoder_key = "layers"
        attention_key = "self_attn"
        mlp_key = "mlp"

        num_layers, hidden_dim, intermediate_dim, gqa_group_size = get_model_dimensions(model, llm_key, decoder_key, mlp_key)

    else:
        raise NotImplementedError


    if cli_args.focus or cli_args.adaptiv or cli_args.CMC:
        replace_focus_forward(
            model,
            llm_forward,
            decoder_forward,
            attention_forward,
            mlp_forward,
            llm_key=llm_key,
            decoder_key=decoder_key,
            attention_key=attention_key,
            mlp_key=mlp_key,
            cli_args=cli_args,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            gqa_group_size=gqa_group_size,
        )



def replace_focus_forward(
    module: torch.nn.Module,
    llm_forward: Callable,
    decoder_forward: Callable,
    attention_forward: Callable,
    mlp_forward: Callable,
    llm_key: str = "model",
    decoder_key: str = "layers",
    decoder_keys: list = None,
    attention_key: str = "self_attn",
    mlp_key: str = "mlp",
    cli_args=None,
    num_layers=None,
    hidden_dim=None,
    intermediate_dim=None,
    gqa_group_size=None,
):
    """
    Replace the forward method of the model with the Focus forward method.
    Make Focus a property of the model.

    The keys are accessed in an hierarchical manner: llm_key -> decoder_key -> attention_key. Each key can have multiple hierarchies, e.g. "llm.model", which will be accessed by module.llm.model
    
    For models with multiple transformers (like Mllama), decoder_keys can be a list of decoder key paths to process multiple transformers in a single call.
    """
    if cli_args.focus:
        if cli_args.similarity_threshold == -1.0:
            # look up configs from focus/configs/focus.csv
            config = pd.read_csv("focus/configs/focus.csv", skipinitialspace=True)
            config_match = config[(config["model"] == cli_args.model) & (config["dataset"] == cli_args.tasks)]
            if config_match.empty:
                config_match = config[config["model"] == "default"]
            cli_args.similarity_threshold = config_match["similarity_threshold"].values[0]
            
            # Convert string representation of list to comma-separated string
            alpha_list_str = str(config_match["alpha_list"].values[0]).strip()
            if alpha_list_str.startswith("[") and alpha_list_str.endswith("]"):
                alpha_list_parsed = ast.literal_eval(alpha_list_str)
                cli_args.alpha_list = ",".join(str(x) for x in alpha_list_parsed)
            else:
                cli_args.alpha_list = alpha_list_str
            
            # Convert string representation of list to comma-separated string
            selected_layers_str = str(config_match["selected_layers"].values[0]).strip()
            if selected_layers_str.startswith("[") and selected_layers_str.endswith("]"):
                selected_layers_parsed = ast.literal_eval(selected_layers_str)
                cli_args.selected_layers = ",".join(str(x) for x in selected_layers_parsed)
            else:
                cli_args.selected_layers = selected_layers_str

        print(f"similarity_threshold: {cli_args.similarity_threshold}")
        print(f"alpha_list: {cli_args.alpha_list}")
        print(f"selected_layers: {cli_args.selected_layers}")

        focus = Focus(
            similarity_threshold=cli_args.similarity_threshold,
            block_size=cli_args.block_size,
            frame_block_size=cli_args.frame_block_size,
            alpha_list=cli_args.alpha_list,
            selected_layers=cli_args.selected_layers,
            gemm_m_size=cli_args.gemm_m_size,
            SEC_only=cli_args.SEC_only,
            vector_size=cli_args.vector_size,
            model_name=cli_args.model,
            dataset_name=cli_args.tasks,
            export_focus_trace=cli_args.export_focus_trace,
            trace_dir=cli_args.trace_dir,
            trace_meta_dir=cli_args.trace_meta_dir,
            trace_name=cli_args.trace_name,
            use_median=cli_args.use_median,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            gqa_group_size=gqa_group_size,
        )
    elif cli_args.CMC:
        if cli_args.CMC_threshold == -1.0:
            # look up configs from focus/configs/cmc.csv
            config = pd.read_csv("focus/configs/cmc.csv", skipinitialspace=True)
            config_match = config[(config["model"] == cli_args.model) & (config["dataset"] == cli_args.tasks)]
            if config_match.empty:
                config_match = config[config["model"] == "default"]
            cli_args.CMC_threshold = config_match["CMC_threshold"].values[0]
            cli_args.CMC_query_threshold = config_match["CMC_query_threshold"].values[0]
            cli_args.CMC_attn_threshold = config_match["CMC_attn_threshold"].values[0]

        print(f"CMC_threshold: {cli_args.CMC_threshold}")
        print(f"CMC_query_threshold: {cli_args.CMC_query_threshold}")
        print(f"CMC_attn_threshold: {cli_args.CMC_attn_threshold}")

        cmc = CMC(interval_size=8, 
                  threshold=cli_args.CMC_threshold, 
                  threshold_query=cli_args.CMC_query_threshold, 
                  threshold_score=cli_args.CMC_attn_threshold, 
                  simplified=cli_args.CMC_simple,
                  model_name=cli_args.model,
                  dataset_name=cli_args.tasks,
                  trace_meta_dir=cli_args.trace_meta_dir,
                  write_sparsity=cli_args.write_sparsity,
                  )
    elif cli_args.adaptiv:
        if cli_args.adaptiv_threshold == -1.0:
            # look up configs from focus/configs/adaptiv.csv
            config = pd.read_csv("focus/configs/adaptiv.csv", skipinitialspace=True)
            config_match = config[(config["model"] == cli_args.model) & (config["dataset"] == cli_args.tasks)]
            if config_match.empty:
                config_match = config[config["model"] == "default"]
            cli_args.adaptiv_threshold = config_match["adaptiv_threshold"].values[0]

        print(f"adaptiv_threshold: {cli_args.adaptiv_threshold}")

        adaptiv = Adaptiv(threshold=cli_args.adaptiv_threshold,
                          model_name=cli_args.model,
                          dataset_name=cli_args.tasks,
                          trace_meta_dir=cli_args.trace_meta_dir,
                          write_sparsity=cli_args.write_sparsity,
                          )



    if cli_args.focus:
        module.focus = focus
    elif cli_args.CMC:
        module.cmc = cmc
    elif cli_args.adaptiv:
        module.adaptiv = adaptiv

    llm = get_attr_by_name(module, llm_key)
    assert isinstance(llm, PreTrainedModel), f"{llm_key} is not a PreTrainedModel"

    if cli_args.focus:
        llm.focus = focus
    elif cli_args.CMC:
        llm.cmc = cmc
    elif cli_args.adaptiv:
        llm.adaptiv = adaptiv

    llm.forward = MethodType(llm_forward, llm)

    # Handle multiple decoder keys if provided, otherwise use single decoder_key
    decoder_keys_to_process = decoder_keys if decoder_keys is not None else [decoder_key]
    
    for current_decoder_key in decoder_keys_to_process:
        decoder_layers = get_attr_by_name(llm, current_decoder_key)
        for i, decoder_layer in enumerate(decoder_layers):
            assert isinstance(decoder_layer, nn.Module), f"{current_decoder_key}[{i}] is not a nn.Module"

            if cli_args.focus:
                decoder_layer.focus = focus
            elif cli_args.CMC:
                decoder_layer.cmc = cmc
            elif cli_args.adaptiv:
                decoder_layer.adaptiv = adaptiv

            decoder_layer.forward = MethodType(decoder_forward, decoder_layer)

            # ensure accelerate hooks are not removed
            if hasattr(decoder_layer, "_hf_hook"):
                decoder_layer._old_forward = MethodType(decoder_forward, decoder_layer)
                add_hook_to_module(decoder_layer, decoder_layer._hf_hook)

            qwen2_attention_instance = get_attr_by_name(decoder_layer, attention_key)
            assert isinstance(qwen2_attention_instance, nn.Module), f"{current_decoder_key}[{i}].self_attn is not a nn.Module"

            # replace the forward method of the attention layer
            if cli_args.focus:
                qwen2_attention_instance.focus = focus
            elif cli_args.CMC:
                qwen2_attention_instance.cmc = cmc
            elif cli_args.adaptiv:
                qwen2_attention_instance.adaptiv = adaptiv

            qwen2_attention_instance.forward = MethodType(attention_forward, qwen2_attention_instance)

            # replace the forward method of the mlp layer
            mlp = get_attr_by_name(decoder_layer, mlp_key)
            assert isinstance(mlp, nn.Module), f"{current_decoder_key}[{i}].mlp is not a nn.Module"

            if cli_args.focus:
                mlp.focus = focus
            elif cli_args.CMC:
                mlp.cmc = cmc
            elif cli_args.adaptiv:
                mlp.adaptiv = adaptiv

            mlp.forward = MethodType(mlp_forward, mlp)

def get_model_dimensions(model, llm_key, decoder_key, mlp_key):
    """
    Extract model architecture dimensions: number of layers, hidden dimension, intermediate dimension, and GQA group size.
    
    Args:
        model: The model to extract dimensions from
        llm_key: Key to access the LLM component (e.g., "model")
        decoder_key: Key to access decoder layers (e.g., "layers")
        mlp_key: Key to access MLP layer (e.g., "mlp")
    
    Returns:
        tuple: (num_layers, hidden_dim, intermediate_dim, gqa_group_size)
    """
    # Get model dimensions
    llm = get_attr_by_name(model, llm_key)
    decoder_layers = get_attr_by_name(llm, decoder_key)
    num_layers = len(decoder_layers)
    
    # Get hidden dimension from config or first layer
    if hasattr(model.config, 'hidden_size'):
        hidden_dim = model.config.hidden_size
    elif len(decoder_layers) > 0:
        # Try to get from first layer's embedding dimension
        first_layer = decoder_layers[0]
        if hasattr(first_layer, 'self_attn') and hasattr(first_layer.self_attn, 'q_proj'):
            hidden_dim = first_layer.self_attn.q_proj.in_features
        else:
            hidden_dim = None
    else:
        hidden_dim = None
    
    # Get intermediate dimension from config or MLP layer
    if hasattr(model.config, 'intermediate_size'):
        intermediate_dim = model.config.intermediate_size
    elif len(decoder_layers) > 0:
        # Try to get from first layer's MLP
        first_layer = decoder_layers[0]
        mlp = get_attr_by_name(first_layer, mlp_key)
        if hasattr(mlp, 'gate_proj') and hasattr(mlp.gate_proj, 'out_features'):
            intermediate_dim = mlp.gate_proj.out_features
        elif hasattr(mlp, 'fc1') and hasattr(mlp.fc1, 'out_features'):
            intermediate_dim = mlp.fc1.out_features
        else:
            intermediate_dim = None
    else:
        intermediate_dim = None
    
    # Get GQA group size from config
    if hasattr(model.config, 'num_attention_heads') and hasattr(model.config, 'num_key_value_heads'):
        num_attention_heads = model.config.num_attention_heads
        num_key_value_heads = model.config.num_key_value_heads
        gqa_group_size = num_attention_heads // num_key_value_heads if num_key_value_heads > 0 else 1
    elif hasattr(model.config, 'num_attention_heads'):
        # If num_key_value_heads is not present, assume MHA (not GQA), so group size is 1
        gqa_group_size = 1
    elif len(decoder_layers) > 0:
        # Try to get from first layer's attention module
        first_layer = decoder_layers[0]
        attention = get_attr_by_name(first_layer, "self_attn")
        if hasattr(attention, 'num_heads') and hasattr(attention, 'num_key_value_heads'):
            num_attention_heads = attention.num_heads
            num_key_value_heads = attention.num_key_value_heads
            gqa_group_size = num_attention_heads // num_key_value_heads if num_key_value_heads > 0 else 1
        elif hasattr(attention, 'num_heads'):
            gqa_group_size = 1
        else:
            gqa_group_size = None
    else:
        gqa_group_size = None
    
    return num_layers, hidden_dim, intermediate_dim, gqa_group_size