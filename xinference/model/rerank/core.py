# Copyright 2022-2023 XProbe Inc.
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

import gc
import importlib
import importlib.util
import logging
import os
import threading
import uuid
from abc import abstractmethod
from collections import defaultdict
from collections.abc import Sequence
from typing import Literal

import numpy as np
import torch
from torch import nn

from ...constants import XINFERENCE_CACHE_DIR
from ...device_utils import empty_cache
from ...types import Document, DocumentObj, Rerank, RerankTokens
from ..core import CacheableModelSpec, ModelDescription, VirtualEnvSettings
from ..utils import is_model_cached
from .utils import preprocess_sentence

logger = logging.getLogger(__name__)

# Used for check whether the model is cached.
# Init when registering all the builtin models.
MODEL_NAME_TO_REVISION: dict[str, list[str]] = defaultdict(list)
RERANK_MODEL_DESCRIPTIONS: dict[str, list[dict]] = defaultdict(list)
RERANK_EMPTY_CACHE_COUNT = int(os.getenv("XINFERENCE_RERANK_EMPTY_CACHE_COUNT", "10"))
assert RERANK_EMPTY_CACHE_COUNT > 0


def get_rerank_model_descriptions():
    import copy

    return copy.deepcopy(RERANK_MODEL_DESCRIPTIONS)


class RerankModelSpec(CacheableModelSpec):
    model_name: str
    language: list[str]
    type: str | None = "unknown"
    max_tokens: int | None
    model_id: str
    model_revision: str | None
    model_hub: str = "huggingface"
    virtualenv: VirtualEnvSettings | None


class RerankModelDescription(ModelDescription):
    def __init__(
        self,
        address: str | None,
        devices: list[str] | None,
        model_spec: RerankModelSpec,
        model_path: str | None = None,
    ):
        super().__init__(address, devices, model_path=model_path)
        self._model_spec = model_spec

    @property
    def spec(self):
        return self._model_spec

    def to_dict(self):
        return {
            "model_type": "rerank",
            "address": self.address,
            "accelerators": self.devices,
            "type": self._model_spec.type,
            "model_name": self._model_spec.model_name,
            "language": self._model_spec.language,
            "model_revision": self._model_spec.model_revision,
        }

    def to_version_info(self):
        from .utils import get_model_version

        if self._model_path is None:
            is_cached = get_cache_status(self._model_spec)
            file_location = get_cache_dir(self._model_spec)
        else:
            is_cached = True
            file_location = self._model_path

        return {
            "model_version": get_model_version(self._model_spec),
            "model_file_location": file_location,
            "cache_status": is_cached,
            "language": self._model_spec.language,
        }


def generate_rerank_description(model_spec: RerankModelSpec) -> dict[str, list[dict]]:
    res = defaultdict(list)
    res[model_spec.model_name].append(RerankModelDescription(None, None, model_spec).to_version_info())
    return res


class _ModelWrapper(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.model = module
        self._local_data = threading.local()

    @property
    def n_tokens(self):
        return getattr(self._local_data, "n_tokens", 0)

    @n_tokens.setter
    def n_tokens(self, value):
        self._local_data.n_tokens = value

    def forward(self, **kwargs):
        attention_mask = kwargs.get("attention_mask")
        # when batching, the attention mask 1 means there is a token
        # thus we just sum up it to get the total number of tokens
        if attention_mask is not None:
            self.n_tokens += attention_mask.sum().item()
        return self.model(**kwargs)

    def __getattr__(self, attr):
        try:
            return super().__getattr__(attr)
        except AttributeError:
            return getattr(self.model, attr)


class RerankModel:
    def __init__(
        self,
        model_spec: RerankModelSpec,
        model_uid: str,
        model_path: str | None = None,
        device: str | None = None,
        use_fp16: bool = False,
        model_config: dict | None = None,
    ):
        self._model_spec = model_spec
        self._model_uid = model_uid
        self._model_path = model_path
        self._device = device
        self._model_config = model_config or dict()
        self._use_fp16 = use_fp16
        self._model = None
        self._counter = 0
        if model_spec.type == "unknown":
            model_spec.type = self._auto_detect_type(model_path)

    @staticmethod
    def _get_tokenizer(model_path):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        return tokenizer

    @staticmethod
    def _auto_detect_type(model_path):
        """This method may not be stable due to the fact that the tokenizer name may be changed.
        Therefore, we only use this method for unknown model types.
        """
        type_mapper = {
            "LlamaTokenizerFast": "LLM-based layerwise",
            "GemmaTokenizerFast": "LLM-based",
            "XLMRobertaTokenizerFast": "normal",
        }

        tokenizer = RerankModel._get_tokenizer(model_path)
        rerank_type = type_mapper.get(type(tokenizer).__name__)
        if rerank_type is None:
            logger.warning(
                f"Can't determine the rerank type based on the tokenizer {tokenizer}, use normal type by default."
            )
            return "normal"
        return rerank_type

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
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
        pass


def get_cache_dir(model_spec: RerankModelSpec):
    return os.path.realpath(os.path.join(XINFERENCE_CACHE_DIR, model_spec.model_name))


def get_cache_status(
    model_spec: RerankModelSpec,
) -> bool:
    return is_model_cached(model_spec, MODEL_NAME_TO_REVISION)


def cache(model_spec: RerankModelSpec):
    from ..utils import cache

    return cache(model_spec, RerankModelDescription)


def create_rerank_model_instance(
    subpool_addr: str,
    devices: list[str],
    model_uid: str,
    model_name: str,
    download_hub: Literal["huggingface", "modelscope", "openmind_hub", "csghub"] | None = None,
    model_path: str | None = None,
    **kwargs,
) -> tuple[RerankModel, RerankModelDescription]:
    from ..utils import download_from_modelscope
    from . import BUILTIN_RERANK_MODELS, MODELSCOPE_RERANK_MODELS
    from .custom import get_user_defined_reranks

    model_spec = None
    for ud_spec in get_user_defined_reranks():
        if ud_spec.model_name == model_name:
            model_spec = ud_spec
            break

    if model_spec is None:
        if download_hub == "huggingface" and model_name in BUILTIN_RERANK_MODELS:
            logger.debug(f"Rerank model {model_name} found in Huggingface.")
            model_spec = BUILTIN_RERANK_MODELS[model_name]
        elif download_hub == "modelscope" and model_name in MODELSCOPE_RERANK_MODELS:
            logger.debug(f"Rerank model {model_name} found in ModelScope.")
            model_spec = MODELSCOPE_RERANK_MODELS[model_name]
        elif download_from_modelscope() and model_name in MODELSCOPE_RERANK_MODELS:
            logger.debug(f"Rerank model {model_name} found in ModelScope.")
            model_spec = MODELSCOPE_RERANK_MODELS[model_name]
        elif model_name in BUILTIN_RERANK_MODELS:
            logger.debug(f"Rerank model {model_name} found in Huggingface.")
            model_spec = BUILTIN_RERANK_MODELS[model_name]
        else:
            raise ValueError(
                f"Rerank model {model_name} not found, available"
                f"Huggingface: {BUILTIN_RERANK_MODELS.keys()}"
                f"ModelScope: {MODELSCOPE_RERANK_MODELS.keys()}"
            )
    if not model_path:
        model_path = cache(model_spec)
    use_fp16 = kwargs.pop("use_fp16", False)
    model = RerankModel(model_spec, model_uid, model_path, use_fp16=use_fp16, model_config=kwargs)
    model_description = RerankModelDescription(subpool_addr, devices, model_spec, model_path=model_path)
    return model, model_description
