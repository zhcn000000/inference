# Copyright 2022-2025 XProbe Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import uuid

import numpy as np

from ....device_utils import empty_cache
from ....types import Document, DocumentObj, Rerank, RerankTokens
from ..core import RERANK_EMPTY_CACHE_COUNT, RerankModel
from ..utils import preprocess_sentence

logger = logging.getLogger(__name__)
SUPPORTED_MODELS_PREFIXES = ["bge", "gte", "text2vec", "m3e", "gte", "Qwen3"]


class VLLMRerankModel(RerankModel):
    def load(self):
        try:
            from vllm import LLM

        except ImportError:
            error_message = "Failed to import module 'vllm'"
            installation_guide = [
                "Please make sure 'vllm' is installed. ",
                "You can install it by `pip install vllm`\n",
            ]

            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

        self._model = LLM(model=self._model_path, task="score")
        self._tokenizer = self._model.get_tokenizer()

    def rerank(
        self,
        documents: list[str],
        query: str,
        top_n: int | None,
        max_chunks_per_doc: int | None,
        return_documents: bool | None,
        return_len: bool | None,
        **kwargs,
    ) -> Rerank:
        assert self._model is not None
        if max_chunks_per_doc is not None:
            raise ValueError("rerank hasn't support `max_chunks_per_doc` parameter.")
        import math

        from vllm import SamplingParams
        from vllm.inputs.data import TokensPrompt

        # 1. 预处理 query
        instruction = kwargs.get("instruction", None)
        pre_query = preprocess_sentence(query, instruction, self._model_spec.model_name)

        # 2. 构造对话消息
        def format_instruction(ins: str, q: str, d: str):
            return [
                {
                    "role": "system",
                    "content": "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be yes or no.",
                },
                {"role": "user", "content": f"<Instruct>: {ins}\n<Query>: {q}\n<Document>: {d}"},
            ]

        pairs = [(pre_query, doc) for doc in documents]
        messages = [format_instruction(instruction, q, d) for q, d in pairs]
        prompts = self._tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False, enable_thinking=False
        )

        # 3. 添加思考后缀
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        suffix_tokens = self._tokenizer.encode(suffix, add_special_tokens=False)
        max_len = getattr(self._model_spec, "max_tokens", None) or 4096
        prompts = [p[: max_len - len(suffix_tokens)] + suffix_tokens for p in prompts]
        token_prompts = [TokensPrompt(prompt_token_ids=p) for p in prompts]

        # 4. 准备采样参数
        true_id = self._tokenizer("yes", add_special_tokens=False).input_ids[0]
        false_id = self._tokenizer("no", add_special_tokens=False).input_ids[0]
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=1,
            logprobs=20,
            allowed_token_ids=[true_id, false_id],
        )

        # 5. 调用 generate 并获取打分
        outputs = self._model.generate(token_prompts, sampling_params, use_tqdm=False)
        similarity_scores: list[float] = []
        for out in outputs:
            last_lp = out.outputs[0].logprobs[-1]
            t_lp = last_lp.get(true_id, type("T", (), {"logprob": -10})).logprob
            f_lp = last_lp.get(false_id, type("F", (), {"logprob": -10})).logprob
            t_sc = math.exp(t_lp)
            f_sc = math.exp(f_lp)
            similarity_scores.append(t_sc / (t_sc + f_sc))

        # 6. 排序截取 top_n
        idxs = list(reversed(np.argsort(similarity_scores)))
        if top_n is not None:
            idxs = idxs[:top_n]

        # 7. 构造返回的 DocumentObj 列表
        results = []
        for i in idxs:
            results.append(
                DocumentObj(
                    index=i,
                    relevance_score=float(similarity_scores[i]),
                    document=Document(text=documents[i]) if return_documents else None,
                )
            )

        # 8. tokens 统计
        tokens_meta = None
        if return_len:
            # 这里只统计输入长度，vllm 生成长度为 0
            total_tokens = sum(len(p.prompt_token_ids) for p in token_prompts)
            tokens_meta = RerankTokens(input_tokens=total_tokens, output_tokens=0)

        meta = {
            "api_version": None,
            "billed_units": None,
            "tokens": tokens_meta,
            "warnings": None,
        }

        self._counter += 1
        if self._counter % RERANK_EMPTY_CACHE_COUNT == 0:
            empty_cache()

        return Rerank(id=str(uuid.uuid1()), results=results, meta=meta)
