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

import codecs
import json
import os
import warnings
from typing import Any, Dict, List

from ...constants import XINFERENCE_MODEL_DIR
from .core import (
    MODEL_NAME_TO_REVISION,
    RERANK_MODEL_DESCRIPTIONS,
    RerankModelSpec,
    generate_rerank_description,
    get_cache_status,
    get_rerank_model_descriptions,
)
from .custom import (
    CustomRerankModelSpec,
    get_user_defined_reranks,
    register_rerank,
    unregister_rerank,
)
from .rerank_family import (
    BUILTIN_RERANK_MODELS,
    FLAG_RERANKER_CLASSES,
    MODELSCOPE_RERANK_MODELS,
    RERANK_ENGINES,
    SENTENCE_TRANSFORMER_CLASSES,
    SUPPORTED_ENGINES,
    VLLM_CLASSES,
)

BUILTIN_RERANK_MODELS: dict[str, Any] = {}
MODELSCOPE_RERANK_MODELS: dict[str, Any] = {}


def register_custom_model():
    # if persist=True, load them when init
    user_defined_rerank_dir = os.path.join(XINFERENCE_MODEL_DIR, "rerank")
    if os.path.isdir(user_defined_rerank_dir):
        for f in os.listdir(user_defined_rerank_dir):
            try:
                with codecs.open(os.path.join(user_defined_rerank_dir, f), encoding="utf-8") as fd:
                    user_defined_rerank_spec = CustomRerankModelSpec.parse_obj(json.load(fd))
                    register_rerank(user_defined_rerank_spec, persist=False)
            except Exception as e:
                warnings.warn(f"{user_defined_rerank_dir}/{f} has error, {e}")


def generate_engine_config_by_model_name(model_spec: "RerankModelSpec"):
    model_name = model_spec.model_name
    engines: dict[str, list[dict[str, Any]]] = RERANK_ENGINES.get(model_name, {})  # structure for engine query
    for engine in SUPPORTED_ENGINES:
        CLASSES = SUPPORTED_ENGINES[engine]
        for cls in CLASSES:
            # Every engine needs to implement match method
            if cls.match(model_spec):
                # we only match the first class for an engine
                engines[engine] = [
                    {
                        "model_name": model_name,
                        "embedding_class": cls,
                    }
                ]
                break
    RERANK_ENGINES[model_name] = engines


def _install():
    load_model_family_from_json("model_spec.json", BUILTIN_RERANK_MODELS)
    load_model_family_from_json("model_spec_modelscope.json", MODELSCOPE_RERANK_MODELS)

    # register model description after recording model revision
    for model_spec_info in [BUILTIN_RERANK_MODELS, MODELSCOPE_RERANK_MODELS]:
        for model_name, model_spec in model_spec_info.items():
            if model_spec.model_name not in RERANK_MODEL_DESCRIPTIONS:
                RERANK_MODEL_DESCRIPTIONS.update(generate_rerank_description(model_spec))

    register_custom_model()

    # register model description
    for ud_rerank in get_user_defined_reranks():
        RERANK_MODEL_DESCRIPTIONS.update(generate_rerank_description(ud_rerank))


def load_model_family_from_json(json_filename, target_families):
    _model_spec_json = os.path.join(os.path.dirname(__file__), json_filename)
    target_families.update(
        dict(
            (spec["model_name"], RerankModelSpec(**spec))
            for spec in json.load(codecs.open(_model_spec_json, "r", encoding="utf-8"))
        )
    )
    for model_name, model_spec in target_families.items():
        MODEL_NAME_TO_REVISION[model_name].append(model_spec.model_revision)
    del _model_spec_json
