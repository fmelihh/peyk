"""Generate ready-to-paste run commands for a model. Prints only — peyk never
downloads or executes anything.
"""

from __future__ import annotations

import re
from typing import Dict

from .models import HardwareProfile, ModelVariant

_QUANT_SUFFIX = re.compile(r"-q\d[\w]*$", re.IGNORECASE)


def _ollama_ref(model_id: str) -> str:
    # Drop peyk's synthetic quant suffix (e.g. "llama3.1:8b-q8" -> "llama3.1:8b").
    return _QUANT_SUFFIX.sub("", model_id)


def build_snippets(variant: ModelVariant, hw: HardwareProfile) -> Dict[str, object]:
    """Return a structured set of run recipes for `variant` on `hw`."""
    commands: Dict[str, str] = {}

    if variant.source == "hf-discovered" and variant.model_id.startswith("hf:"):
        # "hf:<repo>:<QUANT>"
        _, repo, quant = variant.model_id.split(":", 2)
        commands["huggingface-cli"] = (
            f'huggingface-cli download {repo} --include "*{quant}*.gguf" --local-dir ./models'
        )
        commands["llama.cpp"] = f'llama-cli -hf {repo}:{quant} -p "Hello"'
        commands["python (llama-cpp)"] = (
            'from llama_cpp import Llama\n'
            f'llm = Llama.from_pretrained(repo_id="{repo}", filename="*{quant}*.gguf")\n'
            'print(llm("Hello", max_tokens=64)["choices"][0]["text"])'
        )
    else:
        ref = _ollama_ref(variant.model_id)
        commands["ollama"] = f"ollama run {ref}"
        commands["python (ollama)"] = (
            'from ollama import chat\n'
            f'r = chat(model="{ref}", messages=[{{"role": "user", "content": "Hello"}}])\n'
            'print(r.message.content)'
        )

    return {
        "model": variant.family,
        "model_id": variant.model_id,
        "params_b": variant.params_b,
        "quant": variant.quant,
        "commands": commands,
    }
