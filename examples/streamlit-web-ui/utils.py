"""
Shared utility functions for the aas-rail Streamlit application.
"""

import io
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel
from pypdfium2 import PdfDocument
from basyx.aas.adapter import aasx
from basyx.aas.model import (
    ConceptDescription,
    MultiLanguageProperty,
    Property,
    Range,
    Submodel,
    SubmodelElementCollection,
    SubmodelElementList,
)
from basyx.aas.model.provider import DictObjectStore


APP_DIR = Path(__file__).resolve().parent
CACHE_DIR = APP_DIR / ".aas_rail_cache"
LAST_INFERENCE_CACHE_PATH = CACHE_DIR / "last_inference_result.json"


def load_pdf_text(file_bytes: bytes) -> str:
    """Extract text from PDF file bytes."""
    document = PdfDocument(io.BytesIO(file_bytes), autoclose=True)
    text_pages = []
    for page in document:
        text_page = page.get_textpage()
        page_text = text_page.get_text_bounded()
        text_pages.append(page_text)
        text_page.close()
        page.close()
    document.close()
    return "\n\n".join(text_pages)


def make_json_safe(obj: Any) -> Any:
    """Convert non-JSON-serializable objects to JSON-safe equivalents."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, BaseModel):
        return make_json_safe(obj.model_dump())
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [make_json_safe(v) for v in obj]
    if hasattr(obj, "model_dump"):
        return make_json_safe(obj.model_dump())
    return str(obj)


def save_cached_inference_result(result: Any, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Persist the last inference result so it survives Streamlit restarts."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "metadata": make_json_safe(metadata or {}),
        "result": make_json_safe(result),
    }
    with open(LAST_INFERENCE_CACHE_PATH, "w", encoding="utf-8") as file:
        json.dump(record, file, indent=2)
    return record


def load_cached_inference_result() -> Optional[Dict[str, Any]]:
    """Load the persisted last inference result, if one exists."""
    if not LAST_INFERENCE_CACHE_PATH.exists():
        return None
    try:
        with open(LAST_INFERENCE_CACHE_PATH, "r", encoding="utf-8") as file:
            record = json.load(file)
        if isinstance(record, dict) and "result" in record:
            return record
    except (OSError, json.JSONDecodeError):
        return None
    return None


def parse_json_definitions(payload: Any) -> Any:
    """
    Parse property definitions from uploaded JSON.
    Preserves dict-based definitions (e.g., 'classes' format) and lists of properties.
    """
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if "classes" in payload and isinstance(payload["classes"], dict):
            return payload
        if "properties" in payload:
            if isinstance(payload["properties"], list):
                return payload["properties"]
            if isinstance(payload["properties"], dict):
                return list(payload["properties"].values())
        if "definitions" in payload:
            return payload["definitions"]
        if "property_definitions" in payload:
            return payload["property_definitions"]
    raise ValueError("Unable to parse property definitions from uploaded JSON")


def get_property_definition_items(definitions: Any) -> List[Any]:
    """Return the actual property definition entries from supported dictionary shapes."""
    if isinstance(definitions, list):
        return definitions

    if not isinstance(definitions, dict):
        return []

    if "properties" in definitions:
        props = definitions["properties"]
        if isinstance(props, list):
            return props
        if isinstance(props, dict):
            return list(props.values())

    if "definitions" in definitions:
        items = definitions["definitions"]
        if isinstance(items, dict):
            return list(items.values())
        return items if isinstance(items, list) else []

    if "property_definitions" in definitions:
        items = definitions["property_definitions"]
        if isinstance(items, dict):
            return list(items.values())
        return items if isinstance(items, list) else []

    if "classes" in definitions and isinstance(definitions["classes"], dict):
        class_properties = []
        seen = set()
        property_lookup = definitions.get("properties", {})
        for class_item in definitions["classes"].values():
            if not isinstance(class_item, dict):
                continue
            for property_ref in class_item.get("properties", []):
                property_id = str(property_ref)
                if property_id in seen:
                    continue
                seen.add(property_id)
                if isinstance(property_lookup, dict) and property_id in property_lookup:
                    class_properties.append(property_lookup[property_id])
                else:
                    class_properties.append(property_ref)
        return class_properties

    if all(isinstance(value, dict) for value in definitions.values()):
        return list(definitions.values())

    return []


def count_property_definitions(definitions: Any) -> int:
    """Count the number of actual property definitions (not top-level keys)."""
    return len(get_property_definition_items(definitions))


def extract_definition_from_aas_object(obj: Any) -> Optional[Dict[str, Any]]:
    """Extract definition metadata from an AAS object."""
    def safe_multilang_text(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, dict):
            return value.get("en") or next(iter(value.values()), None)
        if hasattr(value, "items"):
            items = list(value.items())
            for language, text in items:
                if language == "en" and text is not None:
                    return str(text)
            for _, text in items:
                if text is not None:
                    return str(text)
            return None
        return str(value)

    object_id = getattr(obj, "id_short", None) or getattr(obj, "id", None)
    if object_id is None:
        return None

    display_name = safe_multilang_text(getattr(obj, "display_name", None))
    description = safe_multilang_text(getattr(obj, "description", None))
    if not description:
        description = display_name or f"Extracted {type(obj).__name__} description"

    return {
        "id": object_id,
        "name": display_name or object_id,
        "type": type(obj).__name__,
        "definition": {"en": description},
        "unit": None,
        "values": None,
    }


def collect_definitions_from_aasx(store: DictObjectStore) -> List[Dict[str, Any]]:
    """Collect property definitions from an AASX object store."""
    definitions: List[Dict[str, Any]] = []
    seen_ids = set()

    def add_definition(definition: Optional[Dict[str, Any]]):
        if not definition:
            return
        if definition["id"] in seen_ids:
            return
        seen_ids.add(definition["id"])
        definitions.append(definition)

    def visit(obj: Any):
        if isinstance(obj, (Property, MultiLanguageProperty, Range, ConceptDescription, Submodel)):
            add_definition(extract_definition_from_aas_object(obj))
        if isinstance(obj, Submodel):
            for element in getattr(obj, "submodel_element", ()) or ():
                visit(element)
        if isinstance(obj, (SubmodelElementCollection, SubmodelElementList)):
            for child in getattr(obj, "value", ()) or ():
                visit(child)

    for obj in store:
        visit(obj)

    return definitions


def load_property_definitions_from_aasx(file_bytes: bytes) -> List[Dict[str, Any]]:
    """Load property definitions from an AASX file."""
    object_store = DictObjectStore()
    file_store = aasx.DictSupplementaryFileContainer()
    with aasx.AASXReader(io.BytesIO(file_bytes)) as reader:
        reader.read_into(object_store, file_store)
    definitions = collect_definitions_from_aasx(object_store)
    if not definitions:
        raise ValueError("No AAS property definitions found in AASX file")
    return definitions


def load_property_definitions(file) -> Any:
    """Load property definitions from a file (JSON, TXT, or AASX)."""
    suffix = file.name.rsplit(".", 1)[-1].lower() if "." in file.name else ""
    content = file.getvalue()
    if suffix == "aasx":
        return load_property_definitions_from_aasx(content)
    if suffix in {"json", "txt"}:
        text = content.decode("utf-8")
        payload = json.loads(text)
        return parse_json_definitions(payload)
    raise ValueError("Unsupported definitions file type. Upload JSON, TXT, or AASX.")


def build_generic_pipeline_dataset(
    datasheet_name: Optional[str], datasheet_bytes: Optional[bytes], property_definitions: Any
) -> str:
    """Build a dataset directory for the generic pipeline with datasheet and definitions."""
    sample_name = Path(datasheet_name).stem if datasheet_name is not None else "aas_rail_input"
    tmp_root = Path(tempfile.mkdtemp(prefix="aas_rail_"))
    sample_dir = tmp_root / sample_name
    sample_dir.mkdir(parents=True, exist_ok=True)

    if datasheet_name is not None and datasheet_bytes is not None:
        suffix = datasheet_name.rsplit(".", 1)[-1].lower()
        if suffix == "pdf":
            with open(sample_dir / f"{sample_name}.pdf", "wb") as f:
                f.write(datasheet_bytes)
        elif suffix == "txt":
            with open(sample_dir / f"{sample_name}.txt", "w", encoding="utf-8") as f:
                f.write(datasheet_bytes.decode("utf-8"))
        else:
            raise ValueError("Unsupported datasheet type for generic pipeline execution. Upload PDF or TXT.")

    definitions_path = sample_dir / f"{sample_name}_definitions.json"
    with open(definitions_path, "w", encoding="utf-8") as f:
        json.dump(property_definitions, f, indent=2)

    return str(sample_dir)


def parse_extraction_result(extraction_result: Any) -> Any:
    """Parse extraction result, handling model objects."""
    if hasattr(extraction_result, "model_dump"):
        return extraction_result.model_dump()
    return extraction_result


def normalize_result_payload(result: Any) -> Any:
    """Normalize result payload to a consistent format."""
    if isinstance(result, dict) and "properties" in result and isinstance(result["properties"], list):
        normalized = {}
        for item in result["properties"]:
            name = item.get("name") or item.get("id")
            normalized[name or str(len(normalized))] = item.get("item", item)
        return normalized
    return result


def build_display_rows(parsed_result: Any) -> List[Dict[str, Any]]:
    """Build table rows from parsed extraction result."""
    rows = []
    if isinstance(parsed_result, dict):
        for name, item in parsed_result.items():
            if isinstance(item, dict):
                rows.append(
                    {
                        "property": name,
                        "value": item.get("value"),
                        "unit": item.get("unit"),
                        "reference": item.get("reference"),
                    }
                )
            else:
                rows.append({"property": name, "value": item})
    elif isinstance(parsed_result, list):
        for element in parsed_result:
            if isinstance(element, dict):
                rows.append(
                    {
                        "property": element.get("name") or element.get("id"),
                        "value": element.get("item") or element.get("value"),
                        "unit": element.get("unit"),
                        "reference": element.get("reference"),
                    }
                )
            else:
                rows.append({"result": element})
    else:
        rows.append({"result": parsed_result})
    return rows
