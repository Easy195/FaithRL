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

logger = logging.getLogger(__name__)
class ResampleRLHFDataset(Dataset):
    def __init__(self, reward_extra_info, tokenizer, max_prompt_length=2048, truncation="error"):
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation

        self.data = []
        for i in range(len(reward_extra_info["resample_context"])):
            self.data.append(
                {
                    "resample_context": reward_extra_info["resample_context"][i],
                    "original_context": reward_extra_info["original_context"][i],
                    "need_resample": reward_extra_info["need_resample"][i],
                    "index": 0,                
                    "tools_kwargs": {},        
                    "interaction_kwargs": {},  
                    "sample_id":reward_extra_info["sample_id"][i]
                }
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        raw_prompt = item["resample_context"]

        encoded = self.tokenizer(
            raw_prompt,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=(self.truncation != "error"),
            max_length=self.max_prompt_length,
        )

        input_ids = encoded["input_ids"][0]
        attention_mask = encoded["attention_mask"][0]
        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0),
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        input_ids = input_ids[0]
        attention_mask = attention_mask[0]
        

        position_ids = compute_position_id_with_mask(attention_mask)
        
        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length:]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[:self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} > max {self.max_prompt_length}")


        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids[0] if position_ids.ndim == 2 else position_ids,
            "raw_prompt_ids": raw_prompt_ids,          
            "index": item["index"],
            "tools_kwargs": item["tools_kwargs"],
            "interaction_kwargs": item["interaction_kwargs"],
            "sample_id":item["sample_id"],
            "extra": [{
                "context": item["original_context"],   
            }],
            "extra_info":{}
        }
