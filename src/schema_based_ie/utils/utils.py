import sys
#sys.path.append("/home/template-ie/vamos_evaluation_framework") # can be removed after rebuilding of docker imag

from datasets.schema_registry import SCHEMA_REGISTRY

def get_model(dataset_name: str, model_name: str):
    """
    Returns a Pydantic model class for a given dataset.
    Fallbacks to common registry if not found in dataset-specific registry.
    Raises KeyError if model is not found anywhere.
    """
    dataset_registry = SCHEMA_REGISTRY.get(dataset_name, {})
    if model_name in dataset_registry:
        return dataset_registry[model_name]

    common_registry = SCHEMA_REGISTRY.get("common", {})
    if model_name in common_registry:
        return common_registry[model_name]

    raise KeyError(f"Model '{model_name}' not found in dataset '{dataset_name}' or common registry.")
if __name__ == "__main__":
    print(SCHEMA_REGISTRY)