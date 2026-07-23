import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

import yaml
from pydantic import BaseModel

from schema_based_ie.experiments.expand_matrix_cfg import generate_configs
from schema_based_ie.langgraphs.generic_pipeline import run

CFG_DIR = "/home/aas-rail/data/inference_results/cfgs/"
RESULTS_DIR = "/home/aas-rail/data/inference_results"

IE_REGISTRY = {
    # "schema_ie": run_schema_ie,
    # "schema_ie_rag": run_schema_ie_rag,
    "generic_pipeline": run
}

def make_json_safe(obj):
    # Preserve native JSON types
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, BaseModel):
        return make_json_safe(obj.model_dump())

    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    
    if isinstance(obj, (list, tuple, set)):
        return [make_json_safe(v) for v in obj]
    # Fallback
    return str(obj)

def get_git_commit_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "unknown"


def load_config(cfg_path: str | Path) -> dict:
    """Load one YAML or JSON inference configuration from disk."""
    with Path(cfg_path).open("r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    if not isinstance(cfg, dict):
        raise ValueError(f"Expected a mapping in configuration file: {cfg_path}")
    return cfg


def run_inference(input_path: str | Path, cfg: dict) -> dict:

    pipeline = IE_REGISTRY[cfg['id_cfg']['graph']]
    result = pipeline(input_path=input_path, cfg=cfg)
    return result

def run_experiment_cfg(input_path: str | Path, cfg_path: str | Path, override = False):
    cfg_idx = Path(cfg_path).stem.split("_")[-1]
    cfg = load_config(cfg_path)

    output_dir = os.path.join(Path(RESULTS_DIR), Path(input_path).name, Path(cfg_path).parent.stem)
    exists = False

    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            if file.startswith(Path(input_path).name + f"_{cfg_idx}_") and file.endswith(".json"):
                exists = True
                break
    if not exists or override:
        result = run_inference(input_path=input_path, cfg=cfg)
        commit_hash = get_git_commit_hash()
        now = datetime.now()
        date_string = now.strftime("%Y-%m-%d_%H-%M-%S")
        file_name = (Path(input_path).name + f"_{cfg_idx}_{commit_hash}_{date_string}.json")
        output_path = Path(output_dir) / file_name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        serializable_result = {
            k: make_json_safe(v)
            for k, v in result.items()
        }
        with open(output_path, "w") as f:
            json.dump(serializable_result, f, indent=4)
    else:
        print("Inference result already exists. Skipping.")

def run_experiment(dataset_path, matrix_cfg_path, override = False):
    cfg_paths = generate_configs(matrix_cfg_path, Path(CFG_DIR) / Path(matrix_cfg_path).stem)
    for cfg_path in cfg_paths:
        for input_file in os.listdir(dataset_path):
            input_path = Path(dataset_path) / input_file
            if input_file.startswith("_") or not os.path.isdir(input_path):
                continue
            run_experiment_cfg(input_path, cfg_path, override)

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on a single sample")
    parser.add_argument("--input", required=True, help="Sample path to run inference on")
    parser.add_argument("--output", help="Output JSON path (defaults to the data results directory)")
    parser.add_argument("--cfg", required=True, help="Path to a YAML or JSON configuration file")
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = load_config(args.cfg)
    result = run_inference(args.input, cfg)

    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_path = Path(RESULTS_DIR) / f"{Path(args.input).name}_{timestamp}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(make_json_safe(result), f, indent=4)
    print(f"Wrote inference result to {output_path}")

if __name__ == "__main__":
    main()
