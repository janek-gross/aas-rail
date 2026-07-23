import json
from pathlib import Path
from collections import defaultdict
import importlib
import pkgutil


def load_registry(path: str | Path) -> dict[str, dict[str, str]]:
    """
    Load a JSONL prompt file into a registry:
    {
        family: {
            variant_id: prompt_text
        }
    }
    """
    registry: dict[str, dict[str, str]] = defaultdict(dict)

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")

    with path.open() as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno} invalid JSON") from e

            try:
                family = record["family"]
                variant_id = record["id"]
                text = record["text"]
            except KeyError as e:
                raise ValueError(
                    f"{path}:{lineno} missing required field: {e}"
                )

            if variant_id in registry[family]:
                raise ValueError(
                    f"{path}:{lineno} duplicate prompt id "
                    f"{family}:{variant_id}"
                )

            registry[family][variant_id] = text

    return dict(registry)





PROMPT_REGISTRY = {}


from schema_based_ie import prompts

# TODO: Load dataset-specific prompt registries

# Load prompt registries
for _, prompt_file, _ in pkgutil.iter_modules(prompts.__path__):
    if prompt_file == "prompt_registry":
        continue
    module = importlib.import_module(f"schema_based_ie.prompts.{prompt_file}")
    # load jsonl files in the module path
    for p in module.__path__:
        p = Path(p)
        for jsonl_file in p.glob("*.jsonl"):
            loaded_registry = load_registry(jsonl_file)
            PROMPT_REGISTRY[jsonl_file.stem] = loaded_registry
