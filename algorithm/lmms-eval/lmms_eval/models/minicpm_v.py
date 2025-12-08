import warnings
from typing import List, Optional, Tuple, Union

import torch
from accelerate import Accelerator, DistributedType
from accelerate.state import AcceleratorState
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

warnings.filterwarnings("ignore")

from loguru import logger as eval_logger


@register_model("minicpm_v")
class MiniCPM_V(lmms):
    """
    MiniCPM_V Model
    """

    def __init__(
        self,
        pretrained: str = "openbmb/MiniCPM-V",
        device: Optional[str] = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = torch.bfloat16,
        batch_size: Optional[Union[int, str]] = 1,
        trust_remote_code: Optional[bool] = True,
        quantization_config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        
        # Handle quantization config
        if quantization_config is not None:
            bnb_config = BitsAndBytesConfig(**quantization_config)
            eval_logger.info(f"Using quantization config: {quantization_config}")
        else:
            bnb_config = None
            
        # Handle any remaining kwargs
        if kwargs:
            eval_logger.warning(f"Unused kwargs: {kwargs}")

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
        else:
            self._device = device
            
        # Prepare model loading arguments
        model_kwargs = {
            "trust_remote_code": trust_remote_code,
            "torch_dtype": dtype,
            "device_map": self._device
        }
        
        # Add quantization config if provided
        if bnb_config is not None:
            model_kwargs["quantization_config"] = bnb_config
            
        self._model = AutoModel.from_pretrained(pretrained, **model_kwargs)
        if bnb_config is None:
            self._model = self._model.to(dtype)
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=trust_remote_code)
        self._config = self._model.config
        self.model.eval()
        self.model.tie_weights()
        self.batch_size_per_gpu = int(batch_size)
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            # If you want to use DistributedType.DEEPSPEED, you have to run accelerate config before using the model
            # Also, you have to select zero stage 0 (equivalent to DDP) in order to make the prepare model works
            # I tried to set different parameters in the kwargs to let default zero 2 stage works, but it didn't work.
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")
            if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            if bnb_config is None:
                self.model.to(self._device)
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        """ """
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        # TODO
        assert False, "We have not implemented this function for MiniCPM_V yet"

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests: List[Instance]) -> List[str]:
        import os
        from PIL import Image
        try:
            from decord import VideoReader, cpu
        except ImportError:
            raise ImportError("Please install decord to handle video inputs: pip install decord")

        MAX_NUM_FRAMES = 64  # You can make this configurable

        def encode_video(video_path):
            def uniform_sample(l, n):
                gap = len(l) / n
                idxs = [int(i * gap + gap / 2) for i in range(n)]
                return [l[i] for i in idxs]

            vr = VideoReader(video_path, ctx=cpu(0))
            sample_fps = round(vr.get_avg_fps() / 1)
            frame_idx = [i for i in range(0, len(vr), sample_fps)]
            if len(frame_idx) > MAX_NUM_FRAMES:
                frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
            frames = vr.get_batch(frame_idx).asnumpy()
            frames = [Image.fromarray(v.astype('uint8')) for v in frames]
            return frames

        res = []

        def _collate(x):
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            # visuals is a list of str (paths)
            visual_paths = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            visual_paths = self.flatten(visual_paths)
            gen_kwargs = all_gen_kwargs[0]

            until = [self.tok_decode(self.eot_token_id)]
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}")
            assert self.batch_size_per_gpu == 1, "Do not support batch_size_per_gpu > 1 for now"
            assert len(visual_paths) == 1, "MiniCPM_V interface does not support bn_image > 1 for now"
            context = contexts[0]

            # --- Begin video/image handling ---
            path = visual_paths[0]
            ext = os.path.splitext(path)[-1].lower()
            is_video = ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"]
            if is_video:
                frames = encode_video(path)
                self.model.num_frames = len(frames)
                msgs = [{"role": "user", "content": frames + [context]}]
                gen_kwargs.setdefault("use_image_id", False)
                gen_kwargs.setdefault("max_slice_nums", 2)
                image_arg = None
            else:
                # Assume image
                image = Image.open(path).convert("RGB")
                if "<image>" in context:
                    context = context.replace("<image>", "")
                msgs = [{"role": "user", "content": context}]
                image_arg = image
            # --- End video/image handling ---

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1
            if "max_inp_length" not in gen_kwargs:
                gen_kwargs["max_inp_length"] = 16384
            # try:
            response = self.model.chat(
                    image=image_arg,
                    msgs=msgs,
                    context=None,
                    tokenizer=self.tokenizer,
                    sampling=True if gen_kwargs["temperature"] > 0 else False,
                    temperature=gen_kwargs["temperature"],
                    top_p=gen_kwargs["top_p"],
                    num_beams=gen_kwargs["num_beams"],
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                    max_inp_length=gen_kwargs["max_inp_length"],
                    **{k: v for k, v in gen_kwargs.items() if k in ["use_image_id", "max_slice_nums"]}
                )
            # except Exception as e:
            #     eval_logger.error(f"Error {e} in generating")
            #     response = ""
            res.append(response)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), response)
            pbar.update(1)
        res = re_ords.get_original(res)
        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
