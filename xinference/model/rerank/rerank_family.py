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
from threading import Lock
from typing import TYPE_CHECKING, Any, Literal

from ..utils import is_valid_model_name

if TYPE_CHECKING:
    from .core import RerankModel, RerankModelSpec

FLAG_RERANKER_CLASSES: list[type["RerankModel"]] = []
SENTENCE_TRANSFORMER_CLASSES: list[type["RerankModel"]] = []
VLLM_CLASSES: list[type["RerankModel"]] = []

BUILTIN_RERANK_MODELS: dict[str, Any] = {}
MODELSCOPE_RERANK_MODELS: dict[str, Any] = {}

logger = logging.getLogger(__name__)


# Desc: this file used to manage rerank models information.
def match_rerank(
    model_name: str,
    download_hub: Literal["huggingface", "modelscope", "openmind_hub", "csghub"] | None = None,
) -> "RerankModelSpec":
    from ..utils import download_from_modelscope

    # The model info has benn init by __init__.py with model_spec.json file
    from .custom import get_user_defined_reranks

    # first, check whether it is a user-defined rerank model
    for model_spec in get_user_defined_reranks():
        if model_name == model_spec.model_name:
            return model_spec

    if download_hub == "modelscope" and model_name in MODELSCOPE_RERANK_MODELS:
        logger.debug(f"Rerank model {model_name} found in ModelScope.")
        return MODELSCOPE_RERANK_MODELS[model_name]
    elif download_hub == "huggingface" and model_name in BUILTIN_RERANK_MODELS:
        logger.debug(f"Rerank model {model_name} found in Huggingface.")
        return BUILTIN_RERANK_MODELS[model_name]
    elif download_from_modelscope() and model_name in MODELSCOPE_RERANK_MODELS:
        logger.debug(f"Rerank model {model_name} found in ModelScope.")
        return MODELSCOPE_RERANK_MODELS[model_name]
    elif model_name in BUILTIN_RERANK_MODELS:
        logger.debug(f"Rerank model {model_name} found in Huggingface.")
        return BUILTIN_RERANK_MODELS[model_name]
    else:
        raise ValueError(
            f"Rerank model {model_name} not found, available"
            f"Huggingface: {BUILTIN_RERANK_MODELS.keys()}"
            f"ModelScope: {MODELSCOPE_RERANK_MODELS.keys()}"
        )


# { rerank model name -> { engine name -> engine params } }
RERANK_ENGINES: dict[str, dict[str, list[dict[str, type["RerankModel"]]]]] = {}
SUPPORTED_ENGINES: dict[str, list[type["RerankModel"]]] = {}
UD_RERANK_FAMILIES_LOCK = Lock()
# user defined rerank models
UD_RERANK_SPECS: dict[str, "RerankModelSpec"] = {}


def register_rarank(custom_rerank_spec: "RerankModelSpec", persist: bool):
    from ..utils import is_valid_model_uri
    from . import generate_engine_config_by_model_name

    if not is_valid_model_name(custom_rerank_spec.model_name):
        raise ValueError(f"Invalid model name {custom_rerank_spec.model_name}.")

    model_uri = custom_rerank_spec.model_uri
    if model_uri and not is_valid_model_uri(model_uri):
        raise ValueError(f"Invalid model URI {model_uri}.")

    with UD_RERANK_FAMILIES_LOCK:
        if (
            custom_rerank_spec.model_name in BUILTIN_RERANK_MODELS
            or custom_rerank_spec.model_name in MODELSCOPE_RERANK_MODELS
            or custom_rerank_spec.model_name in UD_RERANK_SPECS
        ):
            raise ValueError(f"Model name conflicts with existing model {custom_rerank_spec.model_name}")

    UD_RERANK_SPECS[custom_rerank_spec.model_name] = custom_rerank_spec
    generate_engine_config_by_model_name(custom_rerank_spec)


# TODO: add persist feature
def unregister_rerank(custom_rerank_spec: "RerankModelSpec"):
    with UD_RERANK_FAMILIES_LOCK:
        model_name = custom_rerank_spec.model_name
        UD_RERANK_SPECS.pop(model_name, None)
        RERANK_ENGINES.pop(model_name, None)


def check_engine_by_model_name_and_engine(
    model_name: str,
    model_engine: str,
) -> type["RerankModel"]:
    def get_model_engine_from_spell(engine_str: str) -> str:
        for engine in RERANK_ENGINES[model_name].keys():
            if engine.lower() == engine_str.lower():
                return engine
        return engine_str

    if model_name not in RERANK_ENGINES:
        raise ValueError(f"Model {model_name} not found.")
    model_engine = get_model_engine_from_spell(model_engine)
    if model_engine not in RERANK_ENGINES[model_name]:
        raise ValueError(f"Model {model_name} cannot be run on engine {model_engine}.")
    match_params = RERANK_ENGINES[model_name][model_engine]
    for param in match_params:
        if model_name == param["model_name"]:
            return param["rerank_class"]
    raise ValueError(f"Model {model_name} cannot be run on engine {model_engine}.")
