import yaml
import itertools
from copy import deepcopy
from typing import Any
import hashlib
import json
import logging
from aas_rail.langgraphs.generic_pipeline import Cfg
from pydantic import TypeAdapter
import os
from pathlib import Path

logger = logging.getLogger(__name__)

def expand_cfg(node: Any) -> list[Any]:
    """
    Recursively expand a sweep config node.
    Rules:
        - dict  => Cartesian product of values
        - list  => alternatives, flatten branches
        - scalar => single literal choice
    """
    # ---- Case 1: scalar value (base case)
    if not isinstance(node, (dict, list)):
        return [node]

    # ---- Case 2: list => alternatives
    if isinstance(node, list):
        expanded = []
        for item in node:
            expanded.extend(expand_cfg(item))
        return expanded

    # ---- Case 3: dict => Cartesian product
    assert isinstance(node, dict)
    keys = list(node.keys())
    value_lists = [expand_cfg(node[k]) for k in keys]

    configs = []
    for combo in itertools.product(*value_lists):
        merged = {}
        for k, v in zip(keys, combo):
            # v may itself be a nested dict; merge deeply
            if isinstance(v, dict):
                merged[k] = deepcopy(v)
            else:
                merged[k] = v
        configs.append(merged)
    return configs


def merge_dicts(a: dict, b: dict) -> dict:
    """
    Deep merge two config dictionaries.
    """
    result = deepcopy(a)
    for k, v in b.items():
        if k not in result:
            result[k] = deepcopy(v)
        elif isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = merge_dicts(result[k], v)
        else:
            result[k] = deepcopy(v)
    return result



def expand_root(config: Any) -> list[dict]:
    expanded = expand_cfg(config)

    for item in expanded:
        if not isinstance(item, dict):
            raise TypeError(f"Root expansion produced non-dict: {item}")

    return expanded

def canonicalize(obj):
    """Recursively sort dict keys to get a canonical representation."""
    if isinstance(obj, dict):
        return {k: canonicalize(obj[k]) for k in sorted(obj)}
    elif isinstance(obj, list):
        return [canonicalize(x) for x in obj]
    else:
        return obj

def config_hash(cfg) -> str:
    canonical = canonicalize(cfg)
    serialized = json.dumps(canonical, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()[:8]

def save_cfg(output_dir: str | Path, cfg: dict) -> bool:
    os.makedirs(output_dir, exist_ok=True)
    cfg = TypeAdapter(Cfg).validate_python(cfg)
    experiment_name = cfg.id_cfg.experiment_series_id
    cfg=cfg.model_dump()
    idx = config_hash(cfg)
    config_name = f"config_{idx}"
    exp_path = os.path.join(output_dir, experiment_name)
    os.makedirs(exp_path, exist_ok=True)
    out_path = os.path.join(exp_path, f"{config_name}.yaml")
    with open(out_path, "w") as f:
        yaml.dump(cfg, f)
    return out_path

def generate_configs(input_path: str, output_dir: str = None):
    import json
    with open(input_path, "r") as f:
        sweep_cfg = yaml.safe_load(f)

    expanded = expand_root(sweep_cfg)

    logger.info("Generated %s configurations", len(expanded))
    file_paths = []
    if output_dir is None:
        # Print as JSON (for debugging)
        print(json.dumps(expanded, indent=2))
    else:
        for cfg in expanded:
            file_paths.append(save_cfg(output_dir=output_dir,cfg=cfg))
    return file_paths

def cli():
    """
    Command-line interface entrypoint
    """
    import argparse
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="Expand YAML sweep config")
    parser.add_argument("input", help="Input YAML sweep file")
    parser.add_argument("--output_dir", default=None,
                        help="Directory to write config YAMLs (optional)")
    args = parser.parse_args()

    generate_configs(args.input, args.output_dir)

if __name__ == "__main__":
    cli()
