import copy
import logging
import os
import re
from collections import defaultdict
from typing import Optional

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask
import asyncio
import getpass
import inspect
import logging
import os
import pickle
import socket
import time
from contextlib import contextmanager
from dataclasses import asdict
from types import MethodType
from typing import Any, Generator
import requests
from typing import List, Optional, Any
import torch
import re
import ray
import importlib.util
import multiprocessing
import os
import sys
import warnings
from functools import partial
from typing import Any, Optional
import ray
import torch
from omegaconf import DictConfig
from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import get_reward_manager_cls
from verl.workers.reward_manager.abstract import AbstractRewardManager, RawRewardFn
from transformers import AutoTokenizer
import re
from verl import DataProto
import json
import torch
import requests
import re
import ray
from typing import Optional
import threading
import ast
import json, re, ast, requests
import numpy as np
import ray
import torch
import torch.distributed
import zmq
import zmq.asyncio
from filelock import FileLock
from omegaconf import ListConfig
from tensordict import TensorDict
from torch.distributed.device_mesh import DeviceMesh
from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig, CompilationLevel
from vllm.lora.request import LoRARequest
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.worker.worker_base import WorkerWrapperBase

from verl import DataProto
from verl.third_party.vllm import VLLM_SLEEP_LEVEL
from verl.utils.device import is_npu_available
from verl.utils.distributed import initialize_global_process_group_ray
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.ray_utils import ray_noset_visible_devices
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.utils.vllm import TensorLoRARequest, VLLMHijack, is_version_ge
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.base import BaseRollout
from transformers import AutoTokenizer
import requests
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def build_single_prompt(tokenizer, processor, context_str, max_prompt_length, truncation="right"):
    if processor is not None:
        model_inputs = processor(text=[context_str], return_tensors="pt")
        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")
    else:
        model_inputs = tokenizer(
            context_str,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True, 
            max_length=max_prompt_length,
        )
        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")

    eos_id = tokenizer.eos_token_id
    
    if input_ids.size(1) > 0:
        last_token = input_ids[0, -1].item()
        
        if eos_id is not None and last_token == eos_id:
            input_ids = input_ids[:, :-1]
            attention_mask = attention_mask[:, :-1]
        
    input_ids, attention_mask = verl_F.postprocess_data(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_prompt_length,
        pad_token_id=tokenizer.pad_token_id,
        left_pad=True,
        truncation="right",
    )



    position_ids = compute_position_id_with_mask(attention_mask)
    raw_prompt_ids = tokenizer.encode(context_str, add_special_tokens=False)
    if len(raw_prompt_ids) > 0 and raw_prompt_ids[-1] == tokenizer.eos_token_id:
        raw_prompt_ids = raw_prompt_ids[:-1]
    return input_ids[0], attention_mask[0], position_ids[0], raw_prompt_ids


def debug_print_full_batch(batch, non_tensor_batch=None, tokenizer=None, max_elems=10):
    if non_tensor_batch is not None and len(non_tensor_batch) > 0:
        for key, val in non_tensor_batch.items():
            if isinstance(val, np.ndarray):
                print(f"  • {key}: np.ndarray(shape={val.shape}, dtype={val.dtype})")
                for i, v in enumerate(val.tolist()):
                    print(f"      [{i}] {v}")
                    if tokenizer is not None and key.lower() in ["input_ids", "responses", "prompts"]:
                        try:
                            text = tokenizer.decode(
                                [t for t in v if t != tokenizer.pad_token_id],
                                skip_special_tokens=True,
                            )
                            text = text.replace("\n", "\\n")
                            print(f"        🔤 decoded: {text[:200]}{'...' if len(text) > 200 else ''}")
                        except Exception as e:
                            print(f"        ⚠️ decode failed: {e}")
            else:
                print(f"  • {key}: type={type(val).__name__}")
                print(f"      value={val}")



# Note: add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> list[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id
    # is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids
def _split_sentences(text, attach_trailing_whitespace=True):
    abbreviations = [
        "Mr.", "Mrs.", "Ms.", "Dr.", "Rev.", "Adm.", "Gen.",
        "St.", "Sr.", "Jr.", "U.S.", "No.", "a.m.", "p.m.", "Sen."
    ]
    original = text
    working = text
    placeholder_map = {}

    for i, abbr in enumerate(abbreviations):
        ph = f"@@ABBR{i}@@"
        working = working.replace(abbr, abbr.replace('.', ph))
        placeholder_map[ph] = '.'

    working = re.sub(r'\b([A-Z])\.', r'\1@@DOT@@', working)
    placeholder_map['@@DOT@@'] = '.'
    working = re.sub(r'\b(\d+)\.', r'\1@@NUM@@', working)
    placeholder_map['@@NUM@@'] = '.'

    tokens = []
    punct_re = re.compile(r'[。！？.!?]+')
    pos = 0
    for m in punct_re.finditer(working):
        after = working[m.end():]
        qm = re.match(r'["”’)\]]*', after)
        qlen = qm.end() if qm else 0
        wlen = 0
        if attach_trailing_whitespace:
            wm = re.match(r'(?:[ \t]+|(?:\r\n|\r|\n)+)+', after[qlen:])
            if wm:
                wlen = wm.end()
        end_idx = m.end() + qlen + wlen
        token = working[pos:end_idx]
        for ph, dot in placeholder_map.items():
            token = token.replace(ph, dot)
        tokens.append(token)
        pos = end_idx

    if pos < len(working):
        rem = working[pos:]
        for ph, dot in placeholder_map.items():
            rem = rem.replace(ph, dot)
        tokens.append(rem)

    merged_tokens = []
    for t in tokens:
        if t.strip() == "</think>":
            if merged_tokens:
                merged_tokens[-1] += t
            else:
                merged_tokens.append(t)
        elif t.lstrip().startswith("</think>"):
            if merged_tokens:
                merged_tokens[-1] += t
            else:
                merged_tokens.append(t)
        else:
            merged_tokens.append(t)
    tokens = merged_tokens

    reconstructed = ''.join(tokens)
    if reconstructed != original:
        for i, (a, b) in enumerate(zip(original, reconstructed)):
            if a != b:
                break

    return tokens
if is_version_ge(pkg="vllm", minver="0.7.3"):
    VLLMHijack.hijack()


def _get_score(api_url: str, context: str, sentences: list[str]) -> str:
        try:
            r = requests.post(
                api_url, json={"context": context, "sentences": sentences}, timeout=1000
            )
            return r.json().get("sentence_rewards", "")
        except Exception as e:
            return "no"

class vLLMRollout(BaseRollout):
    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
    ):
        super().__init__(config, model_config, device_mesh)

        if config.layered_summon:
            self.sleep_level = 1
        else:
            self.sleep_level = VLLM_SLEEP_LEVEL
        self.current_step = 0
        model_path = model_config.local_path
        tokenizer = model_config.tokenizer
        model_hf_config = model_config.hf_config
        trust_remote_code = model_config.trust_remote_code
        self.lora_kwargs = (
            {"enable_lora": True, "max_loras": 1, "max_lora_rank": model_config.lora_rank}
            if model_config.lora_rank > 0
            else {}
        )

        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), (
            "tensor parallel size should be less than or equal to the world size"
        )
        max_num_batched_tokens = self.config.get("max_num_batched_tokens", 8192)

        rope_scaling_config = getattr(model_hf_config, "rope_scaling", None)
        if not rope_scaling_config:
            max_position_embeddings = None
            if hasattr(model_hf_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.max_position_embeddings
            elif hasattr(model_hf_config, "llm_config") and hasattr(
                model_hf_config.llm_config, "max_position_embeddings"
            ):
                max_position_embeddings = model_hf_config.llm_config.max_position_embeddings
            elif hasattr(model_hf_config, "text_config") and hasattr(
                model_hf_config.text_config, "max_position_embeddings"
            ):
                max_position_embeddings = model_hf_config.text_config.max_position_embeddings
            if max_position_embeddings is None:
                raise ValueError("max_position_embeddings not found in model_hf_config")
            assert max_position_embeddings >= config.prompt_length + config.response_length, (
                "model context length should be greater than total sequence length"
            )
        else:
            rope_scaling_factor = rope_scaling_config.get("factor", 1.0)

            assert (
                model_hf_config.max_position_embeddings * rope_scaling_factor
                >= config.prompt_length + config.response_length
            ), (
                "model context length should be greater than total sequence length, "
                + f"got rope_scaling_factor={rope_scaling_factor} and "
                + f"max_position_embeddings={model_hf_config.max_position_embeddings}"
            )

        max_model_len = int(config.max_model_len or config.prompt_length + config.response_length)

        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError(
                "Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill"
            )

        load_format = "dummy" if config.load_format.startswith("dummy") else config.load_format
        engine_kwargs = config.get("engine_kwargs", {}).get("vllm", {}) or {}
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
        if config.get("limit_images", None):  # support for multi-image data
            engine_kwargs["limit_mm_per_prompt"] = {"image": config.get("limit_images")}

        compilation_config = {}

        cudagraph_capture_sizes = config.get("cudagraph_capture_sizes")
        # enforce_eager must be False to use cudagraph
        if not config.enforce_eager and cudagraph_capture_sizes:
            if isinstance(cudagraph_capture_sizes, ListConfig):
                compilation_config["compilation_config"] = CompilationConfig(
                    level=CompilationLevel.PIECEWISE, cudagraph_capture_sizes=cudagraph_capture_sizes
                )
            else:
                logger.warning(f"cudagraph_capture_sizes must be a list, but got {cudagraph_capture_sizes}")

        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=config.free_cache_engine,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            max_num_seqs=config.max_num_seqs,
            load_format=load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=config.enable_prefix_caching,
            trust_remote_code=trust_remote_code,
            seed=config.get("seed", 0),
            **compilation_config,
            **self.lora_kwargs,
            **engine_kwargs,
        )

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
            repetition_penalty=config.get("repetition_penalty", 1.0),
        )

        kwargs["detokenize"] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)) and k != "seed":
                kwargs[k] = config.get(k)
        kwargs["n"] = 1  # already repeat in ray_trainer
        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @GPUMemoryLogger(role="vllm rollout spmd", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """Generate sequences for a batch of prompts.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
            - prompts: [bsz, prompt_length], prompt token ids from dataset.
            - responses: [bsz, response_length], output token ids include response tokens
              from LLM generation and observation tokens from tool_calls.
            - response_mask: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
            - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
              and response tokens.
            - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - position_ids: [bsz, prompt_length + response_length], incremental position ids.

            For multi-turn conversations:
            responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
            response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
        """
        step = self.current_step
        self.current_step += 1 
        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]
    
        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object
            )
        if "extra_context" not in non_tensor_batch:
            extra_list = []
            for x in prompts.non_tensor_batch.get("extra", []):
                if isinstance(x, np.ndarray):
                    x = x.item()

                if isinstance(x, dict):
                    val = x.get("context", None)
                else:
                    # 否则直接转为字符串
                    val = str(x) if x is not None else None

                extra_list.append(val)

            non_tensor_batch["extra_context"] = np.array(extra_list, dtype=object)
        


        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(
                non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data"), strict=True
            ):
                vllm_inputs.append({"prompt_token_ids": raw_prompt_ids, "multi_modal_data": multi_modal_data})
        else:
            vllm_inputs = [
                {"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
            ]

        for input_data in vllm_inputs:
            if not isinstance(input_data["prompt_token_ids"], list | np.ndarray):
                raise TypeError(
                    f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}"
                )

            input_data["prompt_token_ids"] = list(input_data["prompt_token_ids"])

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        if not do_sample:
            kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,  
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,  
            }

        lora_requests = None
        if self.lora_kwargs:
            lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                lora_requests = [
                    LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/simon-stub-path")
                ] * batch_size
        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                lora_request=lora_requests,
                use_tqdm=False,
            )


            # TODO: disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

            response = []
            rollout_log_probs = []
            for output in outputs:
                for sample_id in range(len(output.outputs)):
                    response_ids = output.outputs[sample_id].token_ids
                    response.append(response_ids)
                    if self.config.calculate_log_probs:
                        curr_log_prob = []
                        for i, logprob in enumerate(output.outputs[sample_id].logprobs):
                            curr_log_prob.append(logprob[response_ids[i]].logprob)
                        rollout_log_probs.append(curr_log_prob)

            response = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.response_length).to(
                idx.device
            )
            if self.config.calculate_log_probs:
                rollout_log_probs = pad_2d_list_to_length(
                    rollout_log_probs, -1, max_length=self.config.response_length
                ).to(idx.device)
                rollout_log_probs = rollout_log_probs.to(torch.float32)

            seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope (batch size, 4, seq len)
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, position_ids.size(1), -1)

        # TODO: fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        from transformers import AutoTokenizer
        from transformers import pipeline
        tokenizer = AutoTokenizer.from_pretrained(
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1___5B",
            trust_remote_code=True
        )
        decoded_inputs = [
            tokenizer.decode(input_ids["prompt_token_ids"], skip_special_tokens=True) 
            for input_ids in vllm_inputs
        ]
        output_ids_list = response.cpu().numpy().tolist()

        decoded_outputs = [
            tokenizer.decode(output_ids, skip_special_tokens=True) 
            for output_ids in output_ids_list
        ]
        prefix_response_list=[]
        need_resample_list = []
        resample_context_list = []
        context_clean=[]
        for context,output_text in zip(non_tensor_batch["extra_context"],decoded_outputs):
            split_marker = "</think>"
            split_idx = output_text.find(split_marker)
            
            if split_idx >= 0:
                cot_text = output_text[:split_idx + len(split_marker)]
                ans_text = output_text[split_idx + len(split_marker):]
            else:
                cot_text, ans_text = output_text, ""
            
            cot_sents = _split_sentences(cot_text)
            ans_sents = _split_sentences(ans_text)
            
            if cot_text.strip() and not cot_sents:
                cot_sents = [cot_text]
            if ans_text.strip() and not ans_sents:
                ans_sents = [ans_text]
            
            sentences = cot_sents + ans_sents
            sentence_labels = []
            data = ast.literal_eval(context)
            context_new = data[0]['context'].strip()
            question =data[0]['question'].strip()
            score=_get_score("http://localhost:8124/predict",context,cot_sents)
            for s in score:

                if s<0.5:
                    sentence_labels.append("no")
                else:
                    sentence_labels.append("yes")

            data_temp = ast.literal_eval(context)
            context_new = data_temp[0]['context']
            # 创建压缩提示
            few_shot_prompt = """Use the following knowledge to answer the given question accurately and only based on the knowledge provided. 
Knowledge:
{knowledge}
Question:
{question}
Your answer MUST be enclosed in a LaTeX box like this: \\boxed{{your answer here}}.

Answer:""".format(question=question,knowledge=context_new)
            messages = [{"role": "user", "content": few_shot_prompt}]
            
            # 使用 tokenizer 创建 chat_str
            if messages[-1]["role"] == "assistant":
                chat_str = tokenizer.apply_chat_template(
                    messages[:-1], tokenize=False, add_generation_prompt=True
                )
                chat_str += messages[-1]["content"] + tokenizer.eos_token
            else:
                chat_str = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

            question_chat = chat_str

            if "no" in sentence_labels:
                first_no_idx = sentence_labels.index("no")
                valid_prefix = " ".join(sentences[:first_no_idx])
                resample_context = question_chat + valid_prefix
                prefix_response=valid_prefix
            else:
                resample_context = question_chat
                prefix_response=""
            need_resample_list.append(True)
            prefix_response_list.append(prefix_response)
            # need_resample_list=[False,True]
            resample_context_list.append(resample_context)

        if any(need_resample_list):

            resample_prompts = []
            for i, (need_resample, context_str) in enumerate(zip(need_resample_list, resample_context_list)):
                if not need_resample:
                    continue
              
                input_ids_r, attention_mask_r, position_ids_r, raw_prompt_ids = build_single_prompt(
                    tokenizer=tokenizer,
                    processor=None,
                    context_str=context_str,
                    max_prompt_length=2048,
                    truncation="right",  
                )

                for _ in range(1):
                    resample_prompts.append({
                        "input_ids": input_ids_r,
                        "attention_mask": attention_mask_r,
                        "position_ids": position_ids_r,
                        "raw_prompt_ids": raw_prompt_ids
                    })

            vllm_inputs_resample = [
                {"prompt_token_ids": r["raw_prompt_ids"]}
                for r in resample_prompts
            ]


            with self.update_sampling_params(**kwargs):
                outputs_resample = self.inference_engine.generate(
                    prompts=vllm_inputs_resample,
                    sampling_params=self.sampling_params,
                    lora_request=lora_requests,
                    use_tqdm=False,
                )

            responses_resample = []
            rollout_log_probs_resample = []
            for output in outputs_resample:
                for sample in output.outputs:
                    resp_ids = sample.token_ids
                    responses_resample.append(resp_ids)
                    if self.config.calculate_log_probs:
                        curr_log_prob = [lp[resp_ids[i]].logprob for i, lp in enumerate(sample.logprobs)]
                        rollout_log_probs_resample.append(curr_log_prob)

            responses_resample = pad_2d_list_to_length(
                responses_resample, self.pad_token_id, max_length=self.config.response_length
            ).to(idx.device)
            if self.config.calculate_log_probs:
                rollout_log_probs_resample = pad_2d_list_to_length(
                    rollout_log_probs_resample, -1, max_length=self.config.response_length
                ).to(idx.device).to(torch.float32)

            idx_resample = torch.stack([r["input_ids"] for r in resample_prompts]).to(idx.device)
            attn_resample = torch.stack([r["attention_mask"] for r in resample_prompts]).to(idx.device)
            pos_resample = torch.stack([r["position_ids"] for r in resample_prompts]).to(idx.device)

            seq_resample = torch.cat([idx_resample, responses_resample], dim=-1)

            resp_len_r = responses_resample.size(1)
            delta_pid_r = torch.arange(1, resp_len_r + 1, device=pos_resample.device).unsqueeze(0).expand(len(resample_prompts), -1)
            if pos_resample.dim() == 3:
                delta_pid_r = delta_pid_r.view(len(resample_prompts), 1, -1).expand(len(resample_prompts), pos_resample.size(1), -1)

            response_pos_r = pos_resample[..., -1:] + delta_pid_r
            pos_resample = torch.cat([pos_resample, response_pos_r], dim=-1)

            resp_mask_r = get_response_mask(response_id=responses_resample, eos_token=eos_token_id, dtype=attn_resample.dtype)
            attn_resample = torch.cat((attn_resample, resp_mask_r), dim=-1)

            # === 拼接回原 batch ===
            idx_all = torch.cat([idx, idx_resample], dim=0)
            response_all = torch.cat([response, responses_resample], dim=0)
            seq_all = torch.cat([seq, seq_resample], dim=0)
            attn_all = torch.cat([attention_mask, attn_resample], dim=0)
            pos_all = torch.cat([position_ids, pos_resample], dim=0)

            batch_size_new = idx_all.size(0)
            batch = TensorDict(
                {
                    "prompts": idx_all,
                    "responses": response_all,
                    "input_ids": seq_all,
                    "attention_mask": attn_all,
                    "position_ids": pos_all,
                },
                batch_size=batch_size_new,
            )
        else:
            empty_prefix = np.array([""] * batch_size, dtype=object)
            non_tensor_batch["prefix_response"] = empty_prefix
            batch = TensorDict(
                {
                    "prompts": idx,
                    "responses": response,
                    "input_ids": seq,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                },
                batch_size=batch_size,
            )
        if self.config.calculate_log_probs:
            batch["rollout_log_probs"] = rollout_log_probs

        if any(need_resample_list):
            new_non_tensor_batch = {}
            repeat_times = 1


            for key, value in non_tensor_batch.items():
                if not isinstance(value, np.ndarray):
                    value = np.array(value, dtype=object)

                if value.ndim > 1 and value.shape[1] == 1:
                    list_of_items = [value[i, 0] for i in range(value.shape[0])]
                else:
                    list_of_items = [value[i] for i in range(value.shape[0])]

                extended_orig = list_of_items.copy()
                extended_resamples = []
                for i, v in enumerate(list_of_items):
                    if need_resample_list[i]:
                        extended_resamples.extend([v] * repeat_times)

                final_list = extended_orig + extended_resamples
                new_non_tensor_batch[key] = np.array(final_list, dtype=object)

            response_final_arr = np.array(prefix_response_list, dtype=object)

            extended_orig_rf = ["" for _ in range(len(response_final_arr))]

            extended_resamples_rf = []
            for i, v in enumerate(response_final_arr.tolist()):
                if need_resample_list[i]:
                    extended_resamples_rf.extend([v] * repeat_times)

            final_rf = extended_orig_rf + extended_resamples_rf
            new_non_tensor_batch["prefix_response"] = np.array(final_rf, dtype=object)

            non_tensor_batch = new_non_tensor_batch


        non_tensor_batch["step"] = np.array([step] * len(non_tensor_batch["extra_context"]), dtype=object)


        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

    async def resume(self, tags: list[str]):
        """Resume rollout weights or kv cache in GPU memory.

        Args:
            tags: weights or kv_cache.
        """
        if not self.config.free_cache_engine:
            return

        if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
            self.inference_engine.wake_up(tags=tags)
        else:
            self.inference_engine.wake_up()

    async def release(self):
        """Release weights and kv cache in GPU memory."""
        self.inference_engine.reset_prefix_cache()

        if not self.config.free_cache_engine:
            return

        self.inference_engine.sleep(level=self.sleep_level)

    async def update_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None], **kwargs):
        """Update the weights of the rollout model.

        Args:
            weights: A generator that yields the name of the weight tensor and the tensor itself.
        """
        peft_config, base_sync_done = kwargs.get("peft_config", None), kwargs.get("base_sync_done", False)
        if peft_config and base_sync_done:
            lora_int_id = int(time.time_ns() % 0x7FFFFFFF)
            lora_reqest = TensorLoRARequest(
                lora_name=f"{lora_int_id}",
                lora_int_id=lora_int_id,
                lora_path="simon_lora_path",
                peft_config=asdict(peft_config),
                lora_tensors=weights,
            )
            self.inference_engine.llm_engine.add_lora(lora_reqest)
            logger.info(f"vLLM load weights, loaded_params: {len(weights)}")
        else:
            from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader

            model = self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner.model
            patch_vllm_moe_model_weight_loader(model)
            model.load_weights(weights)


def _monkey_patch_compute_logits(model, vocab_size: int):
    original_compute_logits = model.compute_logits

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        logits = original_compute_logits(hidden_states, sampling_metadata)
        logits[..., vocab_size:] = float("-inf")
        return logits

    model.compute_logits = MethodType(compute_logits, model)


class vLLMAsyncRollout(BaseRollout):
    """vLLMAsyncRollout is a thin wrapper of WorkerWrapperBase, which is engine in single worker process."""

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
    ):
        super().__init__(config, model_config, device_mesh)

        self.tokenizer = model_config.tokenizer
        self.inference_engine: WorkerWrapperBase = None
        self.address = self._init_zeromq()

        if config.layered_summon or config.expert_parallel_size > 1:
            self.sleep_level = 1
        else:
            self.sleep_level = VLLM_SLEEP_LEVEL

    def _init_zeromq(self) -> str:
        tensor_parallel_size = self.config.tensor_model_parallel_size


        local_world_size = int(os.environ["RAY_LOCAL_WORLD_SIZE"])
        socket_type = "ipc" if tensor_parallel_size <= local_world_size else "tcp"


        with FileLock(f"/tmp/verl_vllm_zmq_{getpass.getuser()}.lock"):
            if socket_type == "ipc":
                pid = os.getpid()
                address = f"ipc:///tmp/verl_vllm_zmq_{pid}_{getpass.getuser()}.ipc"
            else:
                ip, port = self._get_free_port()
                address = f"tcp://{ip}:{port}"
            context = zmq.asyncio.Context()
            self.socket = context.socket(zmq.REP)
            self.socket.bind(address)

        loop = asyncio.get_running_loop()
        self.zmq_loop_task = loop.create_task(self._loop_forever())

        return address

    def _get_free_port(self):
        ip = ray.util.get_node_ip_address()
        with socket.socket() as sock:
            sock.bind(("", 0))
            port = sock.getsockname()[1]
        return ip, port

    async def _loop_forever(self):
        while True:
            try:
                message = await self.socket.recv()
                method, args, kwargs = pickle.loads(message)
                result = await self._execute_method(method, *args, **kwargs)
                await self.socket.send(pickle.dumps(result))
            except Exception as e:
                logger.exception(f"vLLMAsyncRollout _loop_forever error: {e}")
                os._exit(-1)

    def _init_worker(self, all_kwargs: list[dict[str, Any]]):
        """Initialize worker engine."""
        if not torch.distributed.is_initialized():
            initialize_global_process_group_ray()

        all_kwargs[0]["rank"] = int(os.environ["RANK"])
        device_name = "NPU" if is_npu_available else "GPU"
        all_kwargs[0]["local_rank"] = (
            0
            if not ray_noset_visible_devices()
            else int(ray.get_runtime_context().get_accelerator_ids()[device_name][0])
        )
        self.vllm_config = all_kwargs[0]["vllm_config"]
        self.inference_engine = WorkerWrapperBase(vllm_config=self.vllm_config)
        self.inference_engine.init_worker(all_kwargs)

    def _load_model(self, *args, **kwargs):
        self.inference_engine.load_model(*args, **kwargs)
        _monkey_patch_compute_logits(self.inference_engine.worker.model_runner.model, len(self.tokenizer))

    async def _execute_method(self, method: str | bytes, *args, **kwargs):
        if method == "init_worker":
            return self._init_worker(*args, **kwargs)
        elif method == "load_model":
            return self._load_model(*args, **kwargs)
        elif method == "sleep" or method == "wake_up":
            raise ValueError("wake_up and sleep should not be called through ZeroMQ")
        else:
            return self.inference_engine.execute_method(method, *args, **kwargs)

    async def resume(self, tags: list[str]):
        """Resume rollout weights or kv cache in GPU memory.

        Args:
            tags: weights or kv_cache.
        """
        if self.config.free_cache_engine:
            self.inference_engine.wake_up(tags=tags)

    async def release(self):
        """Release weights and kv cache in GPU memory."""
        if self.config.free_cache_engine:
            self.inference_engine.sleep(level=self.sleep_level)

    async def update_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None], **kwargs):
        """Update the weights of the rollout model.

        Args:
            weights: A generator that yields the name of the weight tensor and the tensor itself.
        """
        from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader

        model = self.inference_engine.worker.model_runner.model
        patch_vllm_moe_model_weight_loader(model)
        model.load_weights(weights)

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Batch generate sequences in sync mode."""
        raise NotImplementedError


    def get_zeromq_address(self):
        return self.address

