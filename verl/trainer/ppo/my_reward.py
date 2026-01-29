import requests
from typing import List, Optional
import torch
import re
import ray
from functools import partial
from typing import  Optional
import ray
import torch
from verl import DataProto
from verl.workers.reward_manager.abstract import AbstractRewardManager, RawRewardFn
import re
from verl import DataProto
import json
import torch
import requests
import re
import ray
import numpy as np
from typing import Tuple, Optional
from transformers import pipeline
from typing import Optional
import threading
import ast
import json, re, ast, requests
import json
import os
import re
import ast
import string
from collections import Counter
from typing import Tuple, Optional
debug_log_file = ""
import string
def compute_score_answer(answer_text: str, expected_answer: str):
    """Return 1.0 for correct answer and 0.0 for incorrect answer."""
    predicted_answer = normalize_answer(answer_text)
    gold_answer = normalize_answer(expected_answer)

    if predicted_answer == gold_answer:
        return 1.0
    elif gold_answer in predicted_answer.split():
        return 1.0
    elif gold_answer in predicted_answer:
        return 1.0
    else:
        return 0.0
    
def bool_mapping(s):
    if s == "True":
        return "yes"
    elif s == "False":
        return "no"
    else:
        return s


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation + "".join(["‘", "’", "´", "`"]))
        return "".join(ch if ch not in exclude else " " for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace("_", " ")

    return white_space_fix(remove_articles(remove_punc(lower(replace_underscore(s)))))


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(bool_mapping(prediction))
    normalized_ground_truth = normalize_answer(bool_mapping(ground_truth))

    ZERO_METRIC = (0, 0, 0)

    special_answers = ["yes", "no", "no answer"]

    if normalized_prediction in special_answers or normalized_ground_truth in special_answers:
        if normalized_prediction in normalized_ground_truth.split() or normalized_ground_truth in normalized_prediction.split():
            return 1.0, 1.0, 1.0
        else:
            return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return ZERO_METRIC

    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1, precision, recall


def exact_match_score(prediction, ground_truth):
    return normalize_answer(bool_mapping(prediction)) == normalize_answer(bool_mapping(ground_truth))


def cover_exact_match_score_1(prediction, ground_truth):
    # 不考虑顺序和连续
    pre_list = normalize_answer(bool_mapping(prediction)).split()
    ground_list = normalize_answer(bool_mapping(ground_truth)).split()
    return all(token in pre_list for token in ground_list)


def cover_exact_match_score_2(prediction, ground_truth):
    # 考虑顺序和连续
    pre_list = normalize_answer(bool_mapping(prediction)).split()
    ground_list = normalize_answer(bool_mapping(ground_truth)).split()

    for i in range(len(pre_list) - len(ground_list) + 1):
        if pre_list[i : i + len(ground_list)] == ground_list:
            return True

    pre_str = " ".join(pre_list)
    ground_str = " ".join(ground_list)

    if ground_str in pre_str:
        return True

    return False


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    if metric_fn.__name__ == "exact_match_score":
        for ground_truth in ground_truths:
            em_score = metric_fn(prediction, ground_truth)
            scores_for_ground_truths.append(em_score)
        return max(scores_for_ground_truths)
    elif metric_fn.__name__ == "f1_score":
        for ground_truth in ground_truths:
            f1, precision, recall = metric_fn(prediction, ground_truth)
            scores_for_ground_truths.append((f1, precision, recall))
        f1, precision, recall = max(scores_for_ground_truths, key=lambda x: x[0])
        return f1, precision, recall
    elif metric_fn.__name__ == "cover_exact_match_score_1":
        for ground_truth in ground_truths:
            score = metric_fn(prediction, ground_truth)
            scores_for_ground_truths.append(score)
        return max(scores_for_ground_truths)
    elif metric_fn.__name__ == "cover_exact_match_score_2":
        for ground_truth in ground_truths:
            score = metric_fn(prediction, ground_truth)
            scores_for_ground_truths.append(score)
        return max(scores_for_ground_truths)
    else:
        raise NotImplementedError


def compute_metrics(prediction, gold):
    em = metric_max_over_ground_truths(exact_match_score, prediction, gold)
    f1, precision, recall = metric_max_over_ground_truths(f1_score, prediction, gold)
    cover_em_1 = metric_max_over_ground_truths(cover_exact_match_score_1, prediction, gold)
    cover_em_2 = metric_max_over_ground_truths(cover_exact_match_score_2, prediction, gold)

    metrics = dict()
    metrics["em"] = float(em)
    metrics["cover_em_1"] = float(cover_em_1)
    metrics["cover_em_2"] = float(cover_em_2)
    metrics["f1"] = f1
    metrics["precision"] = precision
    metrics["recall"] = recall

    if cover_em_1:
        metrics["acc_num"] = 1

    return metrics


def validate_model_answer(answer_text: str, expected_answer: str):
    """Parses model's answer text into status dictionary.

    Args:
        answer_text: Text extracted from model's <answer> tags
        expected_answer: Text extracted from data

    Returns:
        Dictionary mapping character names to predicted roles, or None if incomplete
    """

    # print(f"  Expected answer: {expected_answer}")
    # print(f"  Predicted answer: {answer_text}")

    if isinstance(expected_answer, list):
        metrics = compute_metrics(answer_text, expected_answer)
    else:
        metrics = compute_metrics(answer_text, [expected_answer])

    return metrics


def extract_solution(solution_str: str) -> Tuple[Optional[str], str]:
    """Extracts the final answer from the model's response string.

    Args:
        solution_str: Raw response string from the language model

    Returns:
        Tuple containing (extracted_answer, processed_string)
    """
    # Split response to isolate assistant output
    if "Assistant:" in solution_str:
        processed_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        processed_str = solution_str.split("<|im_start|>assistant", 1)[1]
    elif "<｜Assistant｜>" in solution_str:
        processed_str = solution_str.split("<｜Assistant｜>", 1)[1]
    elif "<|start_header_id|>assistant<|end_header_id|>" in solution_str:
        processed_str = solution_str.split("<|start_header_id|>assistant<|end_header_id|>", 1)[1]
    else:
        print("  [Error] Failed to locate model response header")
        return None, solution_str

    # Extract reasoning and final answer using XML-style tags
    reasoning_pattern = r'<think>(.*?)</think>'
    matches = list(re.finditer(reasoning_pattern, processed_str, re.DOTALL))
    if not matches:
        print("\n  [Error] No valid reasoning text found")

    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))
    if not matches:
        print("\n  [Error] No valid answer text found")
        answer_text = None
    else:
        answer_text = matches[-1].group(1).strip()

    return answer_text, processed_str


def validate_response_structure(processed_str: str) -> bool:
    """Performs comprehensive validation of response structure.

    Args:
        processed_str: Processed response string from the model

    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    print("\n[Format Validation]")
    validation_passed = True

    # Check required tags
    tags = {
        'think_start': ('<think>', 1),
        'think_end': ('</think>', 1),
        'answer_start': ('<answer>', 1),
        'answer_end': ('</answer>', 1)
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)

        print(f"  {tag_str}: count={count}, position={pos}")

        if count != expected_count:
            print(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
            validation_passed = False

    # Verify tag order
    if (positions['think_start'] > positions['think_end'] or
            positions['think_end'] > positions['answer_start'] or
            positions['answer_start'] > positions['answer_end']):
        # print("  [Error] Incorrect tag order: Expected <think>...</think><answer>...</answer>")
        validation_passed = False

    return validation_passed

import json
import os
import re


def extract_last_bracket_content(text: str) -> str:
    """
    1. [[ Answer ]
    2. [[ Answer ]]
    3. \[ Answer \]
    4. \boxed{Answer}
    """
    if not text:
        return ""

    patterns = [
        r'\[\[\s*(.*?)\s*\]',              # sloppy
        r'\[\[\s*(.*?)\s*\]\]',            # standard [[...]]
        r'(?:\\\[|\[)\s*\[\s*(.*?)\s*\]\s*(?:\\\]|\])',  # LaTeX \[...\]
        r'\\boxed\s*\{(.*?)\}',            # \boxed{...}
    ]

    best_match = None
    best_pos = -1  

    for pattern in patterns:
        for m in re.finditer(pattern, text, re.DOTALL):
            start = m.start()
            content = m.group(1)


            if start >= best_pos:
                best_pos = start
                best_match = content


    if best_match is None:
        return ""

    raw_content = best_match.strip()

    m2 = re.match(r'^\\text\s*\{(.*?)\}$', raw_content, re.DOTALL)
    if m2:
        raw_content = m2.group(1).strip()

    if raw_content.startswith("[") and raw_content.endswith("]"):
        raw_content = raw_content[1:-1].strip()

    return raw_content                         

class HTTPRMReward:
    def __init__(
        self,
        api_url: str,
        cot_reward_value: float = 1.0,
        cot_penalty_value: float = -1.0,
        ans_reward_value: float = 1.0,
        ans_penalty_value: float = -1.0,
        len_reward_scale: float = 1.0,
        timeout: float = 10.0,
        tokenizer=None,
        ngram_size: int = 10,           
        repetition_penalty_value: float = -1.0, 
        
        cot_step_penalty: float = 0.08, 
        ans_step_penalty: float = 1.0,

        ans_safe_word_limit:float=20,
        cot_safe_word_limit:float=50,

        cot_safe_sentence_limit:float=10,
        ans_safe_sentence_limit:float=1,
    ):
        self.api_url = api_url
        self.cot_reward_value = cot_reward_value
        self.cot_penalty_value = cot_penalty_value
        self.ans_reward_value = ans_reward_value
        self.ans_penalty_value = ans_penalty_value
        self.len_reward_scale = len_reward_scale
        self.timeout = timeout
        self.tokenizer = tokenizer
        self.ngram_size = ngram_size
        self.repetition_penalty_value = repetition_penalty_value
        self.cot_step_penalty = cot_step_penalty 
        self.ans_step_penalty = ans_step_penalty
        self.ans_safe_word_limit = ans_safe_word_limit 
        self.cot_safe_word_limit = cot_safe_word_limit
        self.cot_safe_sentence_limit=cot_safe_sentence_limit
        self.ans_safe_sentence_limit=ans_safe_sentence_limit

    def _compute_repetition_penalty(self, text):

        if not text:
            return 0.0
        
       
        tokens = [t for t in text.split() if len(t) > 0]
        
        if len(tokens) < self.ngram_size:
            return 0.0

      
        ngrams = [tuple(tokens[i:i+self.ngram_size]) for i in range(len(tokens) - self.ngram_size + 1)]
        
        if not ngrams:
            return 0.0

        unique_ngrams = set(ngrams)
        total_ngrams = len(ngrams)

        repetition_ratio = 1.0 - (len(unique_ngrams) / total_ngrams)
        
 
        if repetition_ratio > 0.1:
 
            return repetition_ratio * self.repetition_penalty_value
            
        return 0.0

    def __call__(self, batch: "DataProto", return_dict: bool = False, **kwargs):

        non_tensor = batch.non_tensor_batch
        prefix_response = non_tensor.get("prefix_response", []) 
        context_list = non_tensor["extra_context"]
        step_list = non_tensor["step"]

        responses_tensor = batch.batch["responses"]  # [bsz, response_length]


        if isinstance(responses_tensor, torch.Tensor):
            responses_list = responses_tensor.detach().cpu().tolist()
        else:
            responses_list = responses_tensor

        output_list = []
        for r in responses_list:
            if hasattr(r, "tolist"):
                r = r.tolist()
            if len(r) > 0 and isinstance(r[0], (list, tuple)):
                sample_ids = r[0] if hasattr(r[0], "tolist") else r[0]
                text = self.tokenizer.decode(sample_ids, skip_special_tokens=True)
                output_list.append(text)
            else:
                text = self.tokenizer.decode(r, skip_special_tokens=True)
                output_list.append(text)

        output_list = [o if isinstance(o, str) else str(o) for o in output_list]

        token_list = [
            self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(text, add_special_tokens=False))
            for text in output_list
        ]


             
        full_responses_list = [p + r for p, r in zip(prefix_response, output_list)]

        batch_processed_data = []
        all_f1_scores = []  

        for i, (context_raw, output_text, full_text, step, prefix) in enumerate(zip(
            context_list, output_list, full_responses_list, step_list, prefix_response
        )):

            prefix_num = 0
            if prefix and prefix.strip():
                prefix_num = len(self._split_sentences(prefix))


            split_marker = "</think>"
            split_idx = full_text.find(split_marker)
            
            if split_idx >= 0:
                cot_end_in_full = split_idx + len(split_marker)
            else:

                cot_end_in_full = len(full_text)

            prefix_len_in_str = len(prefix)
            local_split_idx = max(0, cot_end_in_full - prefix_len_in_str)
            
            cot_text = output_text[:local_split_idx]
            ans_text = output_text[local_split_idx:]

            cot_sents = self._split_sentences(cot_text)
            ans_sents = self._split_sentences(ans_text)
            

            if cot_text.strip() and not cot_sents: cot_sents = [cot_text]
    


            data = ast.literal_eval(context_raw)
            context_str = data[0]['context'].strip()
            gold_answer = data[0]['gold_answer'].strip()
            question = data[0]['question'].strip()

            is_correct = False
            final_ans = ""
            current_f1 = 0.0
            answer_score = 0.0

            if ans_sents:
                final_ans = extract_last_bracket_content(ans_text)
                if final_ans != "":

                    score = compute_score_answer(final_ans, gold_answer)
                    answer_score = score
                    
                    if score == 1.0:
                        is_correct = True
     
                        f1_res = f1_score(final_ans, gold_answer)

                        if isinstance(f1_res, (tuple, list)):
                            current_f1 = f1_res[0]
                        else:
                            current_f1 = f1_res
                    else:
                        is_correct = False
                        current_f1 = 0.0 
                else:
                    current_f1 = 0.0
            else:
                current_f1 = 0.0
            
            all_f1_scores.append(current_f1)


            batch_processed_data.append({
                "context_str": context_str,
                "output_text": output_text,
                "full_text": full_text,
                "cot_sents": cot_sents,
                "ans_sents": ans_sents,
                "gold_answer": gold_answer,
                "question": question,
                "final_ans": final_ans,
                "is_correct": is_correct,
                "answer_score": answer_score,
                "step": step,
                "f1": current_f1,
                "prefix_num": prefix_num, 
                "sample_id": i
            })

        if not all_f1_scores:
            f1_min, f1_max = 0.0, 0.0
        else:
            f1_min = min(all_f1_scores)
            f1_max = max(all_f1_scores)

        all_token_rewards = []

        for idx, item in enumerate(batch_processed_data):
            token_texts = token_list[idx]
            
            cot_sents = item["cot_sents"]
            ans_sents = item["ans_sents"]
            is_correct = item["is_correct"]
            context_str = item["context_str"]
            raw_f1 = item["f1"]
            prefix_num = item["prefix_num"] 

            sentence_rewards = []


            if not is_correct:
                cot_base_scores = [self.cot_penalty_value] * len(cot_sents)
            else:
                cot_prm_scores = self._get_score(context_str, cot_sents)

                cot_base_scores = [self.cot_reward_value if s < 0.5 else self.cot_penalty_value for s in cot_prm_scores]


            for c_idx, base_r in enumerate(cot_base_scores):
                if base_r < 0:
                    sentence_rewards.append(base_r)
                    continue
                
                cot_r = base_r

                cot_nums = c_idx + prefix_num + 1 
                excess_sentence = max(0, cot_nums - self.cot_safe_sentence_limit)
                cot_num_penalty = excess_sentence * self.cot_step_penalty
                
                word_count = len(cot_sents[c_idx].split())
                excess_words = max(0, word_count - self.cot_safe_word_limit)
                word_num_penalty = excess_words * 0.02
                
                final_r = max(cot_r - cot_num_penalty - word_num_penalty, -1.0)
                sentence_rewards.append(final_r)

            

            if f1_max > f1_min:

                normalized_f1 = (raw_f1 - f1_min) / (f1_max - f1_min)

                ans_base_r = normalized_f1 * 2.0 - 1.0
            else:
                ans_base_r = 1.0 if raw_f1 > 0 else -1.0


            for a_idx, sent in enumerate(ans_sents):
                if raw_f1 == 0:
                    sentence_rewards.append(-1.0)
                    continue
                sentence_rewards.append(ans_base_r)

            rep_penalty = self._compute_repetition_penalty(item["full_text"])

            debug_record = {
                "step": item["step"],
                "sample_id": item["sample_id"],
                "question": item["question"],
                "full_text":item["full_text"],
                "cot_sentences": cot_sents,
                "cot_rewards": sentence_rewards[:len(cot_sents)],
                "ans_sentences": ans_sents,
                "ans_rewards": sentence_rewards[len(cot_sents):],
                "ans_final": item["final_ans"],
                "gold_answer": item["gold_answer"],
                "answer_score": item["answer_score"],
                "f1_score": raw_f1,
                "ans_base_reward": ans_base_r,
                "prefix_len_sents": prefix_num,
                "repetition_penalty": rep_penalty,
            }
            with open(debug_log_file, "a", encoding="utf-8") as dbg:
                dbg.write(json.dumps(debug_record, ensure_ascii=False) + "\n")
            

            sentences = cot_sents + ans_sents

            token_rewards = self._assign_token_rewards(token_texts, sentences, sentence_rewards, self.tokenizer)
            token_rewards_tensor = torch.tensor(token_rewards, dtype=torch.float)

            if rep_penalty != 0 and token_rewards_tensor.numel() > 0:
                token_rewards_tensor += rep_penalty

            all_token_rewards.append(token_rewards_tensor)

        reward_tensor = torch.nn.utils.rnn.pad_sequence(
            all_token_rewards, batch_first=True, padding_value=0.0
        )

        if return_dict:
            return {"reward_tensor": reward_tensor, "extra_info": {}}
        return reward_tensor, {}



    @staticmethod
    def zipngram_tokens(tokens: list[int], ngram_size: int):
        return zip(*[tokens[i:] for i in range(ngram_size)])

    def _get_score(self, context: str, sentences: list[str]) -> str:
        try:
            r = requests.post(
                self.api_url, json={"context": context, "sentences": sentences}, timeout=self.timeout
            )
            return r.json().get("sentence_rewards", "")
        except Exception as e:
            print(f"❌ {e}")
            return "no"

    def _split_sentences(self, text, attach_trailing_whitespace=True):
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

    def _assign_token_rewards(
            self,
            token_texts: list[str],
            sentences: list[str],
            sentence_rewards: list[float],
            tokenizer,
            cot_reward_value: float = 0.5,
            ans_penalty_value: float = -1.0,
        ) -> list[float]:
        if not token_texts:
            return []
        if not sentences:
            return [ans_penalty_value] * len(token_texts)
        

        full_text = "".join(sentences)
        encoding = tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=False)
        offsets = encoding.offset_mapping
       

        sent_boundaries = []
        char_idx = 0
        for s in sentences:
            start = char_idx
            end = char_idx + len(s)
            sent_boundaries.append((start, end))
            char_idx = end

        token_rewards = []
        token_idx = 0

        
        for tok in token_texts:
            if token_idx >= len(offsets):
                token_rewards.append(0.0)
                continue

            start, end = offsets[token_idx]


            if start == end :
                token_rewards.append(0.0)
            else:
                reward = 0.0
                for i, (s_start, s_end) in enumerate(sent_boundaries):
                    if start >= s_start and start < s_end:
                        reward = sentence_rewards[i]
                        break
                token_rewards.append(reward)

            token_idx += 1

        while len(token_rewards) < len(token_texts):
            token_rewards.append(0.0)

        return token_rewards


@ray.remote(num_cpus=1)
def compute_reward_async_http(batch: "DataProto", reward_fn: Optional[HTTPRMReward] = None):
    if reward_fn is None:
        raise ValueError("reward_fn must be provided for HTTPRMReward")
    reward_tensor, reward_extra_info = reward_fn(batch)
    return reward_tensor, reward_extra_info
