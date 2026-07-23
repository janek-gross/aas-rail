import io
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, TypeAdapter
from pypdfium2 import PdfDocument

from schema_based_ie.model_clients.client_configs import ClientCfg, ExtractionHelperClientCfg

from .extraction_helper_models import (
    ExtractionHelper,
    ExtractionHelperRunResult,
    PropertyRecord,
    ResponseModel,
    extraction_helper_response_factory,
    property_groups_prompt_payload,
)


ExtractionHelperGenerationClientCfg = ClientCfg | ExtractionHelperClientCfg


DEFAULT_EXTRACTION_HELPER_INSTRUCTION = """
You are creating an ExtractionHelper for target property instances in a PDF datasheet extraction benchmark.
Your task is to fill an ExtractionHelper that helps another extractor reproduce the reference value for the same property in similar datasheets.
Even if a value is not directly contained in a table it might still be possible to derive it somehow from the source.

Goal:
Help another extractor reproduce the given reference value in similar datasheets, but only using extraction logic supported by document evidence.

Important:
The reference value shows the expected output. Mimic the reference output as closely as possible, even when the reference is not the cleanest or most semantically ideal extraction.
Do not create a generic property definition. Create a helper for this specific property instance. Ground every extraction rule in document evidence.
Don't mention the reference value if possible to reduce the potential of hallucinations in subsequent steps.

Field meanings:

* grounding: how well the reference is supported by the visible source.
* evidence: where the supporting source evidence is found, or that no visible evidence supports it.
* extraction_rule: how to select, parse, or derive the value (avoid mentioning the value).
* formatting_rule: how to format the result like the reference (avoid mentioning the value).
* avoid: tempting but wrong candidates or strategies.

Rules:

* Only describe rules that can be verified from the PDF.
* Assume the reference value is always correct and describe how to reproduce it from the PDF.
* Never say “select the value after the property label” unless that label or a clear synonym is actually visible.
* If the property label is absent, look for contextual evidence such as titles, markings, headers, footers, tables, product codes, or certification blocks.
* If the value could not be derived from visible source evidence, use grounding="reference_only".
* If visible evidence suggests a different value, use grounding="conflicting".
* Do not repeat the target output value in any response field.
* Preserve the reference convention; do not improve wording, formatting, units, casing, precision, or typos.

Grounding:

* exact: the reference value is directly visible in the source.
* inferred: the reference value is derivable from source evidence.
* conflicting: the source points to a different value than the reference.
* reference_only: the value could not be derived from source evidence.

Few-shot examples:

Example 1 — unsupported reference

Property: Connector Form
Reference: [target value]
PDF situation: The property label is absent. The target value is absent. No synonym or derivable evidence is visible.

Good helper:
{
"grounding": "reference_only",
"evidence": "The value could not be derived from visible source evidence.",
"extraction_rule": "No extraction rule could be derived.",
"formatting_rule": "No formatting rule could be derived.",
"avoid": ["Do not invent a property-label lookup."]
}

Example 2 — value from address context

Property: Manufacturer City
Reference: [target value]
PDF situation: The property label is absent. A footer contains a manufacturer address with street, postal code, locality, and country.

Good helper:
{
"grounding": "inferred",
"evidence": "Footer address containing Linden-Straße 1 | 55929 Singenhausen | Germany.",
"extraction_rule": "Inspect the address and select the locality after the postal code.",
"formatting_rule": "Return the locality text as written, without postal code or country.",
"avoid": ["Do not require the property label to be visible.", "Do not return the postal code or country."]
}

Example 3 — visible label/value pair

Property: Rated Voltage
Reference: 1300
PDF situation: A technical table contains a matching label and value.

Good helper:
{
"grounding": "inferred",
"evidence": "Technical characteristics table with a matching property label.",
"extraction_rule": "Select the value in the same row as the visible matching label.",
"formatting_rule": "Transform the value to kV and return the numeric string with two decimal places and without unit.",
"avoid": ["Ignore similarly named impulse or test voltage rows."]
}

Return only a valid object matching the response schema.

"""


class ExtractionHelperCfg(BaseModel):
    """Configuration for the extraction-helper LangGraph."""

    client_cfg: ExtractionHelperGenerationClientCfg = Field(default_factory=ExtractionHelperClientCfg)
    batch_size: int = Field(default=8, ge=1)
    temperature: float | None = None
    timeout: float | None = 180.0
    retry: int | None = 1
    max_pdf_chars: int = Field(default=60000, ge=1000)
    debug_output_path: str | None = None
    enable_thinking: bool | None = None


class State(BaseModel):
    """LangGraph state for batched extraction-helper generation."""

    cfg: ExtractionHelperCfg = Field(default_factory=ExtractionHelperCfg)
    pdf_text: str
    instruction: str = DEFAULT_EXTRACTION_HELPER_INSTRUCTION
    property_groups: list[PropertyRecord]
    batch_index: int = 0
    batch_start: int = 0
    batch_end: int = 0
    prompt_instance: str = ""
    extraction_helpers: list[ExtractionHelper] = Field(default_factory=list)
    usage: list[dict[str, Any]] = Field(default_factory=list)
    raw_generation_results: list[dict[str, Any]] = Field(default_factory=list)
    generation_errors: list[str] = Field(default_factory=list)


class ExtractionHelperGenerator:
    def __init__(self, cfg: ExtractionHelperCfg):
        from schema_based_ie.model_clients.llm_clients import GENERATION_CLIENT_REGISTRY

        self.cfg = cfg
        self.client: Any = GENERATION_CLIENT_REGISTRY[cfg.client_cfg.client_name]()

    def generate(self, prompt: str, response_schema: type[BaseModel]) -> dict[str, Any]:
        return self.client.generate(
            parameters={
                "model": self.cfg.client_cfg.model,
                "temperature": self.cfg.temperature,
                "timeout": self.cfg.timeout,
                "retry": self.cfg.retry,
                "enable_thinking": self.cfg.enable_thinking
            },
            response_schema=response_schema,
            messages=[{"role": "system", "content": prompt}],
        )


def load_pdf_text_from_bytes(file_bytes: bytes) -> str:
    """Extract text from PDF bytes."""
    document = PdfDocument(io.BytesIO(file_bytes), autoclose=True)
    text_pages = []
    for page in document:
        text_page = page.get_textpage()
        text_pages.append(text_page.get_text_bounded().replace("\r\n", "\n").replace("\r", "\n"))
        text_page.close()
        page.close()
    document.close()
    return "\n\n".join(text_pages)


def select_batch(state: State) -> State:
    state.batch_start = state.batch_index * state.cfg.batch_size
    state.batch_end = min(state.batch_start + state.cfg.batch_size, len(state.property_groups))
    return state


def assemble_prompt(state: State) -> State:
    batch = _current_batch(state)
    pdf_text = state.pdf_text[: state.cfg.max_pdf_chars]
    if len(state.pdf_text) > state.cfg.max_pdf_chars:
        pdf_text += "\n\n[PDF text truncated for model context length.]"

    state.prompt_instance = (
        "[System Instruction]\n"
        f"{state.instruction.strip()}\n\n"
        "[PDF Text]\n"
        f"{pdf_text}\n\n"
        "Return exactly one extraction helper for each property."
        "Use the property keys exactly as the top-level JSON field names. Do not add, omit, or rename keys. "
        "Do not include internal identifiers in the response. "
        "Keep evidence snippets short and copy only the minimal relevant phrase.\n\n"
        "[Properties to derive from PDF text]\n"
        f"{json.dumps(property_groups_prompt_payload(batch), ensure_ascii=False, indent=2)}"
    )
    return state


def generate_instructions(state: State, generator: ExtractionHelperGenerator) -> State:
    result: dict[str, Any] = {}
    batch = _current_batch(state)
    response_schema = extraction_helper_response_factory(batch)
    try:
        result = generator.generate(state.prompt_instance, response_schema=response_schema)
        parsed_instructions = _helpers_from_response(result.get("parsed"), batch)
        state.raw_generation_results.append(
            {
                "batch_index": state.batch_index,
                "batch_start": state.batch_start,
                "batch_end": state.batch_end,
                "response_schema_fields": _response_field_names(batch),
                "text": result.get("text"),
                "parsed": parsed_instructions,
                "usage": result.get("usage"),
            }
        )
    except Exception as exc:
        error_message = f"Batch {state.batch_index} instruction generation failed: {exc}"
        state.generation_errors.append(error_message)
        parsed_instructions = [_fallback_helper_for_group(group) for group in batch]
        state.raw_generation_results.append(
            {
                "batch_index": state.batch_index,
                "batch_start": state.batch_start,
                "batch_end": state.batch_end,
                "response_schema_fields": _response_field_names(batch),
                "error": error_message,
            }
        )

    state.extraction_helpers.extend(parsed_instructions)
    if result.get("usage"):
        state.usage.append(result["usage"])
    state.batch_index += 1
    return state



def has_more_batches(state: State) -> str:
    return "has_more" if state.batch_end < len(state.property_groups) else "done"


def _write_debug_state(debug_output_path: str | None, payload: Mapping[str, Any]) -> None:
    if not debug_output_path:
        return

    try:
        path = Path(debug_output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(_make_json_safe(payload), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        print(f"Warning: failed to write extraction-instruction debug state: {exc}")


def _make_json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, BaseModel):
        return _make_json_safe(value.model_dump())
    if isinstance(value, Mapping):
        return {str(key): _make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_make_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _current_batch(state: State) -> list[PropertyRecord]:
    return state.property_groups[state.batch_start:state.batch_end]


def _response_field_names(groups: Sequence[PropertyRecord]) -> list[str]:
    return [group.response_field_name for group in groups]


def _helpers_from_response(value: Any, groups: Sequence[PropertyRecord]) -> list[ExtractionHelper]:
    if isinstance(value, BaseModel):
        value = value.model_dump()
    if not isinstance(value, Mapping):
        raise TypeError("Extraction-helper response must be a JSON object.")

    payload = dict(value)
    helpers = []
    for group in groups:
        if group.response_field_name not in payload:
            raise ValueError(f"Missing extraction helper for response field '{group.response_field_name}'.")
        helpers.append(_helper_from_group(group, payload[group.response_field_name]))
    return helpers


def _helper_from_group(group: PropertyRecord, payload: Any) -> ExtractionHelper:
    representative = group.records[0]
    data = ResponseModel.model_validate(payload).model_dump()
    data["group_key"] = group.group_key
    data["response_field_name"] = group.response_field_name
    data["semantic_id"] = group.semantic_id or ""
    data["property_record_id"] = representative.record_id
    data["property_name"] = representative.name
    return ExtractionHelper.model_validate(data)


def _complete_helpers(
    groups: Sequence[PropertyRecord],
    helpers: Sequence[ExtractionHelper],
) -> list[ExtractionHelper]:
    by_group_key = {helper.group_key: helper for helper in helpers if helper.group_key}
    by_field_name = {helper.response_field_name: helper for helper in helpers if helper.response_field_name}
    by_semantic_id = {helper.semantic_id: helper for helper in helpers if helper.semantic_id}
    by_record_id = {helper.property_record_id: helper for helper in helpers if helper.property_record_id}
    by_name = {helper.property_name.casefold(): helper for helper in helpers if helper.property_name}

    completed = []
    for group in groups:
        helper = (
            by_group_key.get(group.group_key)
            or by_field_name.get(group.response_field_name)
            or (by_semantic_id.get(group.semantic_id) if group.semantic_id else None)
            or next((by_record_id[record.record_id] for record in group.records if record.record_id in by_record_id), None)
            or next((by_name[record.name.casefold()] for record in group.records if record.name.casefold() in by_name), None)
        )
        completed.append(_complete_helper_metadata(group, helper) if helper else _fallback_helper_for_group(group))
    return completed


def _complete_helper_metadata(group: PropertyRecord, helper: ExtractionHelper) -> ExtractionHelper:
    representative = group.records[0]
    data = helper.model_dump()
    data["group_key"] = group.group_key
    data["response_field_name"] = group.response_field_name
    data["semantic_id"] = group.semantic_id or data.get("semantic_id") or ""
    data["property_record_id"] = data.get("property_record_id") or representative.record_id
    data["property_name"] = data.get("property_name") or representative.name
    return ExtractionHelper.model_validate(data)


def _fallback_helper_for_group(group: PropertyRecord) -> ExtractionHelper:
    representative = group.records[0]
    return ExtractionHelper(
        group_key=group.group_key,
        response_field_name=group.response_field_name,
        semantic_id=group.semantic_id or "",
        property_record_id=representative.record_id,
        property_name=representative.name,
        grounding="reference_only",
        evidence="No source evidence was generated.",
        extraction_rule="No extraction rule could be generated.",
        formatting_rule="No formatting rule could be generated.",
        avoid=[],
    )


def build_graph(generator: ExtractionHelperGenerator):
    """Build the extraction-helper LangGraph."""
    graph_builder = StateGraph(State)
    graph_builder.add_node(select_batch)
    graph_builder.add_node(assemble_prompt)
    graph_builder.add_node("generate_instructions", lambda state: generate_instructions(state, generator))
    graph_builder.add_edge(START, "select_batch")
    graph_builder.add_edge("select_batch", "assemble_prompt")
    graph_builder.add_edge("assemble_prompt", "generate_instructions")
    graph_builder.add_conditional_edges(
        "generate_instructions",
        has_more_batches,
        {
            "has_more": "select_batch",
            "done": END,
        },
    )
    return graph_builder.compile()


def run_extraction_helper_pipeline(
    pdf_text: str,
    property_groups: Sequence[PropertyRecord | Mapping[str, Any]],
    cfg: ExtractionHelperCfg | Mapping[str, Any] | None = None,
    instruction: str = DEFAULT_EXTRACTION_HELPER_INSTRUCTION,
) -> ExtractionHelperRunResult:
    """Generate extraction helpers for prepared property groups from PDF text."""
    cfg_model = TypeAdapter(ExtractionHelperCfg).validate_python(cfg or {})
    groups = [
        group if isinstance(group, PropertyRecord) else PropertyRecord.model_validate(group)
        for group in property_groups
    ]
    if not groups:
        return ExtractionHelperRunResult(debug_output_path=cfg_model.debug_output_path)

    generator = ExtractionHelperGenerator(cfg_model)
    graph = build_graph(generator)
    graph_input = {
        "cfg": cfg_model,
        "pdf_text": pdf_text,
        "instruction": instruction,
        "property_groups": groups,
    }
    try:
        print("Running extraction-helper generation pipeline...")
        result = graph.invoke(graph_input)
        _write_debug_state(cfg_model.debug_output_path, result)
    except Exception as exc:
        _write_debug_state(
            cfg_model.debug_output_path,
            {
                "error": str(exc),
                "input": graph_input,
            },
        )
        raise

    state = State.model_validate(result)
    completed = _complete_helpers(groups, state.extraction_helpers)
    return ExtractionHelperRunResult(
        instructions=completed,
        usage=state.usage,
        errors=state.generation_errors,
        debug_output_path=cfg_model.debug_output_path,
    )
