from pathlib import Path
from pydantic import BaseModel, Field, TypeAdapter
from typing import Any, Annotated, Callable
from pypdfium2 import PdfDocument
import json
import chromadb
import hashlib
import math
import re
from schema_based_ie.langgraphs.node_services import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from schema_based_ie.model_clients.llm_clients import EmbeddingClient, GenerationClient
from schema_based_ie.prompts.prompt_registry import load_registry
from typing import Literal
from schema_based_ie.schemata.rag_schemata.rag_retrieve_schema import RETRIEVE_SCHEMA_REGISTRY
from schema_based_ie.schemata.ie_schemata.aas_ie import AAS_SCHEMA_REGISTRY
from schema_based_ie.model_clients.llm_clients import GENERATION_CLIENT_REGISTRY, CACHED_EMBEDDING_CLIENT_REGISTRY#
from schema_based_ie.model_clients.client_configs import (
    ClientCfg,
    EmbeddingCfg,
    LlamaClientCfg,
)
from schema_based_ie.langgraphs.icl.aasx_io import (
    extract_product_metadata,
    extract_technical_data_properties,
    load_aasx_object_store,
)
from schema_based_ie.langgraphs.icl.neo4j_connection import (
    DEFAULT_NEO4J_PASSWORD,
    DEFAULT_NEO4J_URI,
    DEFAULT_NEO4J_USER,
)
from schema_based_ie.langgraphs.icl.rdf_queries import (
    normalize_property_definition_params,
    query_datasheet_embeddings,
    query_property_values_with_metadata,
)
from langgraph.graph import StateGraph, START, END
from dataclasses import dataclass

CHROMADB_PATH = "/home/aas-rail/data/chromadb/"

################################################
############# Configuration Models #############
################################################

class ICLCfg(BaseModel):
    helper_generation_hash: str = Field(default="", description="Optional helper-generation config hash to restrict retrieved ICL examples.")
    helper_artifact_id: str = Field(default="", description="Optional ICL artifact/experiment ID to restrict retrieved ICL examples.")
    helper_provider: str = Field(default="", description="Optional helper-generation provider filter, e.g. 'llama' or 'openai'.")
    helper_model: str = Field(default="", description="Optional helper-generation model filter, e.g. 'gpt-oss:20b'.")
    candidates_per_property: int = Field(default=20, ge=1, description="Raw ICL candidates to retrieve per requested property.")
    examples_per_property: int = Field(default=1, ge=0, description="Ranked ICL examples to add to each extraction property.")
    query_technical_data_only: bool = Field(default=True, description="Restrict ICL candidate retrieval to TechnicalData submodels.")
    query_technical_properties_only: bool = Field(default=True, description="Restrict ICL candidate retrieval to TechnicalProperties paths.")
    prefer_technical_data: bool = Field(default=True, description="Prefer ICL candidates from TechnicalData paths during ranking.")
    prefer_technical_properties: bool = Field(default=True, description="Prefer ICL candidates from TechnicalProperties paths during ranking.")
    use_datasheet_embedding_similarity: bool = Field(default=True, description="Use source datasheet embedding similarity for ICL ranking when compatible embeddings are available.")
    embedding_cfg: EmbeddingCfg = Field(default_factory=EmbeddingCfg, description="Embedding configuration used to compare the input datasheet with stored ICL datasheet embeddings.")


class PromptCfg(BaseModel):
    type: str = Field(default = "manual", description="The type of the prompt, e.g. 'manual' or 'retrieval_based'.")
    id: str = Field(default = "v0", description="The id of the prompt template to use within the specified type.")


class QueryGeneratorCfg(BaseModel):
    client_cfg: ClientCfg = Field(default_factory=LlamaClientCfg, description="Configuration for the model client to use for query generation.")
    embedding_cfg: EmbeddingCfg = Field(default_factory=EmbeddingCfg, description="Configuration for the embedding generator used for retrieval-based query generation.")
    batch_size: int = Field(default=16, description="The number of properties to generate queries for in each batch.")
    n_queries_per_property: int = Field(default=5, description="The number of queries to generate for each property. The LLM aims to formulate text sections that might appear in the Datasheet")
    top_k: int = Field(default=5, description="Number of retreived chunks per query")
    n_chunks: int = Field(default=10, description="Final number of chunks per property after query aggregation and deduplication")
    response_schema: str = Field(default="retrieve_factory", description="The schema to use for retrieval-based query generation, which determines the structure of the prompt and the expected output.")
    prompt_cfg: PromptCfg = Field(default_factory=PromptCfg, description="Configuration for the prompt used to generate retrieval queries.")
    timeout: float | None = Field(default=180.0, description="API timeout duration in seconds")
    retry: int | None = Field(default=1, description="Number of retries per LLM call")
    enable_thinking: bool | None = None

class SchemaBasedExtractorCfg(BaseModel):
    client_cfg: ClientCfg = Field(default_factory=LlamaClientCfg, description="Configuration for the model client to use for schema-based extraction.")
    prompt: PromptCfg = Field(default_factory=PromptCfg, description="Configuration for the prompt used for schema-based extraction.")
    temperature: float | None = Field(default=None, json_schema_extra={"perturbation": True}, description="The temperature to use for the language model during schema-based extraction, which controls the randomness of the output. A value of 0.0 means deterministic output, while higher values allow for more variability.")
    prompt_degradation_intensity: float = Field(default=0.0, json_schema_extra={"perturbation": True}, description="The intensity of prompt degradation during schema-based extraction.")
    response_schema: str = Field(default="aas_factory", description="The schema to use for schema-based extraction, which determines the structure of the prompt and the expected output.")
    batch_size: int = Field(default=16, description="The number of properties to extract in each batch.")
    timeout: float | None = Field(default=180.0, description="API timeout duration in seconds")
    retry: int | None = Field(default=1, description="Number of retries per LLM call")
    enable_thinking: bool | None = None

class IdCfg(BaseModel):
    graph: Literal['generic_pipeline'] = 'generic_pipeline'
    experiment_series_id: str = Field(default='tests', description="Name of the experiment series")
    tool_development_stage: str = Field(default='alpha', description="Name to discern major tool development stages")

class Cfg(BaseModel):
    id_cfg: IdCfg = Field(default_factory=IdCfg, description="Description for traceability")
    icl_cfg: ICLCfg | None = Field(default=None, description="Configuration for the in-context learning examples store.")
    query_generator_cfg: QueryGeneratorCfg | None = Field(default=None, description="Configuration for the query generation")
    schema_based_extractor_cfg: SchemaBasedExtractorCfg =  Field(default_factory=SchemaBasedExtractorCfg, description="Configuration for the information extraction LLM")

def default_config_to_yaml():
    """Write the current default configuration to the bundled YAML location."""
    import yaml
    file_path = Path(__file__).resolve().parent.parent / "experiments" / "default_config.yaml"
    with open(file_path, "w") as f:
        yaml.dump(Cfg().model_dump(), f, sort_keys=False)


##########################################
############# Node Services ##############
##########################################

class DefinitionQueryGenerator:
    def __init__(self, cfg: Cfg):
        self.n_queries_per_property = cfg.query_generator_cfg.n_queries_per_property
        self.client: GenerationClient = GENERATION_CLIENT_REGISTRY[cfg.query_generator_cfg.client_cfg.client_name]()
        self.model = cfg.query_generator_cfg.client_cfg.model
        self.timeout = cfg.query_generator_cfg.timeout
        self.retry = cfg.query_generator_cfg.retry
        self.enable_thinking = cfg.query_generator_cfg.enable_thinking

    def generate(self, context: str) -> dict[str,Any]:


        res = self.client.generate(
            parameters = {'model': self.model, 'timeout': self.timeout, 'retry': self.retry, 'enable_thinking': self.enable_thinking},
            response_schema=context['schema_instance'],
            messages = [{"role": "system", "content": context['prompt_instance']}]
        )
        usage = res['usage']
        queries = [[query for query in queries.values()] for queries in res['parsed'].model_dump().values()]
        return {'usage': usage, 'queries': queries}

class EphemeralChromaVectorStore(VectorStore):
    def __init__(self, cfg: ClientCfg):
        self.cfg = cfg
        self.client = chromadb.EphemeralClient()
        self.collection = self.client.get_or_create_collection(
            name="docs",
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, texts, embeddings, ids):
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            ids=ids,
        )

    def search(self, queries, top_k: int, embedder: EmbeddingClient | None = None):
        if embedder is not None:
            result = embedder.embed(queries, {'model': self.cfg.model})
            q_embds = result['embeddings']
            usage = result['usage']
        else:
            # assume queries are already embeddings if no embedder provided
            q_embds = queries
            usage = None
        if usage:
            return {'docs': self.collection.query(
            query_embeddings=q_embds,
            n_results=top_k), 'usage': usage}
        else:
            return {'docs': self.collection.query(
            query_embeddings=q_embds,
            n_results=top_k)}


class DefinitionSchemaBasedExtractor:
    def __init__(self, cfg: Cfg):
        self.cfg = cfg.schema_based_extractor_cfg
        self.client: GenerationClient = GENERATION_CLIENT_REGISTRY[self.cfg.client_cfg.client_name]()
        self.model = self.cfg.client_cfg.model
        self.temperature = self.cfg.temperature
        self.timeout = self.cfg.timeout
        self.retry = self.cfg.retry
        self.enable_thinking = self.cfg.enable_thinking
        self.prompt_degradation_intensity = self.cfg.prompt_degradation_intensity
        if self.prompt_degradation_intensity > 0.0:
            # conditional import due to performance reasons
            from schema_based_ie.langgraphs.text_perturbation import degrade_text
            self.degrade_text = degrade_text

    def extract(self, context):

        if self.prompt_degradation_intensity > 0.0:
            context['prompt_instance'] = self.degrade_text(context['prompt_instance'], intensity=self.prompt_degradation_intensity)

        parameters = {'model': self.model, 'temperature': self.temperature, 'timeout': self.timeout, 'retry': self.retry, 'enable_thinking': self.enable_thinking}
        print("schema_extracor",self.model)
        result = self.client.generate(
            parameters=parameters,
            response_schema=context['schema_instance'],
            messages=[{"role": "system", "content": context['prompt_instance']}],
        )
        return result
@dataclass
class Services:
    icl_embedding_client: EmbeddingClient | None
    embedding_client: EmbeddingClient | None
    ephemeral_vector_store: EphemeralChromaVectorStore  | None
    query_generator: DefinitionQueryGenerator | None
    schema_based_extractor: DefinitionSchemaBasedExtractor

#############################################
############# Langgraph States ##############
#############################################


def merge_identical(a: Any, b: Any) -> Any:
    if a != b:
        print(a, type(a))
        print(b, type(b))
        print()
        raise ValueError(f"Conflicting objects detected during fan-in:\n{a}\n{b}")
    return a  # they are identical, safe to use

def merge_single(a: Any, b: Any) -> Any:
    if a is not None and b is not None:
        print(a, type(a))
        print(b, type(b))
        print()
        raise ValueError(f"Unexpected objects detected during a fan-in:\n{a}\n{b}")
    return a if a is not None else b

def update(a: Any, b: Any) -> Any:
    if b in [None, [],{}]:
        return a
    else:
        return b
    
def overwrite(a: Any, b: Any) -> Any:
    return b

class State(BaseModel):
    cfg: Annotated[Cfg | None, merge_identical] = None
    input_path: Annotated[str | Path, merge_identical]
    pdf_doc: Annotated[Any | None, update] = None
    preprocessed_text: Annotated[str | None, update] = None
    chunks: Annotated[list[str], update] = []
    embeddings: Annotated[list[list[float]], update] = []
    dictionary: Annotated[dict[str, Any] | None, update] = None
    product_metadata: Annotated[dict[str, str] | None, update] = None
    definitions: Annotated[list[dict[str, Any]] | None, update] = None
    icl: Annotated[dict[str, list[dict[str, Any]]] | None, update] = None
    prompts: Annotated[dict[str, str] | None, update] = None
    schemata: Annotated[dict[str, BaseModel | Callable] | None, update] = None
    query_batch_index: Annotated[int | None, update] = None  # deterministic counter, likely no merge needed
    query_batch_start: Annotated[int | None, update] = None
    query_batch_end: Annotated[int | None, update] = None
    queries: Annotated[list[list[str]] | None, update] = None
    extraction_batch_index: Annotated[int | None, update] = None
    extraction_batch_start: Annotated[int | None, update] = None
    extraction_batch_end: Annotated[int | None, update] = None
    retrieved_chunks: Annotated[list[list[dict[str, Any]]], update] = []
    retrieved_texts: Annotated[list[str], update] = []
    extraction_result: Annotated[dict[str, Any], update] = {}
    usage: Annotated[dict[str, Any], update] = {}


############################################
############# helper functions #############
############################################

def normalize_pdf(pdf: PdfDocument) -> str:
    text = [page.get_textpage().get_text_bounded().replace("\r\n", "\n").replace("\r", "\n")
                for page in pdf]
    return "\n".join(text) if isinstance(text, list) else text

def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    text = text or ""
    if not text:
        return []
    if chunk_size <= 0:
        return [text]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=max(0, chunk_overlap),
    )
    return [chunk for chunk in text_splitter.split_text(text) if chunk.strip()]

def aggregate_retrievals(result, top_k):
    aggregated = []

    ids_all = result["ids"]
    docs_all = result["documents"]
    dists_all = result["distances"]
    metas_all = result["metadatas"]

    # Flatten all results from all queries
    for q in range(len(ids_all)):
        ids = ids_all[q]
        docs = docs_all[q]
        dists = dists_all[q]
        metas = metas_all[q]

        for i in range(len(ids)):
            aggregated.append({
                "id": ids[i],
                "distance": dists[i],
                "document": docs[i],
                "metadata": metas[i]
            })

    # Sort globally by similarity (ascending distance)
    aggregated = sorted(aggregated, key=lambda x: x["distance"])

    # Deduplicate: keep the best (lowest distance) occurrence per chunk
    unique = []
    seen = set()
    for item in aggregated:
        if item["id"] not in seen:
            seen.add(item["id"])
            unique.append(item)

        if len(unique) == top_k:
            break

    return unique


def icl_lookup_key(property_id: str, property_name: str) -> str:
    return f"{property_id}\t{property_name}"


def definition_icl_lookup_key(definition: dict[str, Any]) -> str:
    normalized = normalize_property_definition_params([definition])[0]
    return icl_lookup_key(normalized["property_id"], normalized["property_name"])


def row_icl_lookup_key(row: dict[str, Any]) -> str:
    return icl_lookup_key(
        str(row.get("requestedPropertyId") or ""),
        str(row.get("requestedPropertyName") or ""),
    )


def build_ranked_icl_examples(
    definitions: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    cfg: ICLCfg,
    datasheet_similarity_by_source: dict[str, float] | None = None,
    *,
    target_eclass_id: str = "",
    target_manufacturer_name: str = "",
) -> dict[str, list[dict[str, Any]]]:
    """Group, rank, and format RDF-backed ICL rows by requested property."""
    if cfg.examples_per_property <= 0:
        return {}

    grouped_rows: dict[str, list[dict[str, Any]]] = {}
    for row in candidate_rows:
        grouped_rows.setdefault(row_icl_lookup_key(row), []).append(row)

    examples_by_property: dict[str, list[dict[str, Any]]] = {}
    for definition in definitions:
        key = definition_icl_lookup_key(definition)
        ranked_rows = rank_icl_candidate_rows(
            grouped_rows.get(key, []),
            cfg,
            datasheet_similarity_by_source or {},
            target_eclass_id=target_eclass_id,
            target_manufacturer_name=target_manufacturer_name,
        )
        examples = [
            format_icl_example(row, datasheet_similarity_by_source or {})
            for row in ranked_rows[: cfg.examples_per_property]
        ]
        if examples:
            examples_by_property[key] = examples
    return examples_by_property


def rank_icl_candidate_rows(
    candidate_rows: list[dict[str, Any]],
    cfg: ICLCfg,
    datasheet_similarity_by_source: dict[str, float] | None = None,
    *,
    target_eclass_id: str = "",
    target_manufacturer_name: str = "",
) -> list[dict[str, Any]]:
    """Rank candidates using the layered preference order from the ICL plan."""
    similarities = datasheet_similarity_by_source or {}

    def sort_key(index_and_row: tuple[int, dict[str, Any]]) -> tuple[Any, ...]:
        index, row = index_and_row
        source_name = str(row.get("sourceName") or "")
        return (
            int(is_same_manufacturer(row.get("manufacturerName"), target_manufacturer_name)),
            int(cfg.prefer_technical_data and is_technical_data_candidate(row)),
            int(cfg.prefer_technical_properties and is_technical_properties_candidate(row)),
            eclass_similarity(row.get("eclassId"), target_eclass_id),
            similarities.get(source_name, -1.0),
            -index,
        )

    return [
        row
        for _, row in sorted(
            enumerate(candidate_rows),
            key=sort_key,
            reverse=True,
        )
    ][: cfg.candidates_per_property]


def format_icl_example(
    row: dict[str, Any],
    datasheet_similarity_by_source: dict[str, float] | None = None,
) -> dict[str, Any]:
    values = clean_icl_values(row.get("values"))
    instructions = clean_icl_instructions(row.get("extractionInstructions") or [])
    example = {
        # "source": row.get("sourceName") or "",
        "property": row.get("propertyIdShort") or row.get("semanticId") or "",
        # "values": values[:5],
        "unit": first_instruction_value(instructions, "unit"),
        # "path": row.get("elementPath") or [],
        # "submodel": row.get("submodelIdShort") or "",
        # "manufacturer": row.get("manufacturerName") or "",
        # "eclass_id": row.get("eclassId") or "",
        "instructions": instructions[:2],
    }
    if datasheet_similarity_by_source:
        source_name = str(row.get("sourceName") or "")
        if source_name in datasheet_similarity_by_source:
            example["datasheet_similarity"] = round(datasheet_similarity_by_source[source_name], 6)
    return {key: value for key, value in example.items() if value not in ("", [], None)}


def clean_icl_values(values: Any) -> list[str]:
    if values is None:
        return []
    if not isinstance(values, list):
        values = [values]

    cleaned = []
    seen = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        cleaned.append(text)
    return cleaned


def clean_icl_instructions(instructions: list[Any]) -> list[dict[str, Any]]:
    cleaned_instructions = []
    allowed_keys = [
        "unit",
        "grounding",
        "evidence",
        "extractionRule",
        "formattingRule",
        "avoid",
    ]
    for instruction in instructions:
        if not isinstance(instruction, dict):
            continue
        cleaned = {
            key: instruction.get(key)
            for key in allowed_keys
            if has_icl_value(instruction.get(key))
        }
        if cleaned:
            cleaned_instructions.append(cleaned)
    return cleaned_instructions


def first_instruction_value(instructions: list[dict[str, Any]], key: str) -> Any:
    for instruction in instructions:
        value = instruction.get(key)
        if has_icl_value(value):
            return value
    return None


def has_icl_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, list):
        return any(str(item).strip() for item in value)
    return str(value).strip() != ""


def is_same_manufacturer(candidate: Any, target: str) -> bool:
    candidate_text = str(candidate or "").strip().lower()
    target_text = str(target or "").strip().lower()
    if not candidate_text or not target_text:
        return False
    return candidate_text in target_text or target_text in candidate_text


def is_technical_data_candidate(row: dict[str, Any]) -> bool:
    return any("technicaldata" in normalize_identifier(value) for value in candidate_path_values(row))


def is_technical_properties_candidate(row: dict[str, Any]) -> bool:
    return any("technicalproperties" in normalize_identifier(value) for value in candidate_path_values(row))


def candidate_path_values(row: dict[str, Any]) -> list[Any]:
    values: list[Any] = [
        row.get("submodelIdShort"),
        row.get("submodelUri"),
    ]
    element_path = row.get("elementPath") or []
    if isinstance(element_path, list):
        values.extend(element_path)
    else:
        values.append(element_path)
    return values


def normalize_identifier(value: Any) -> str:
    return "".join(char for char in str(value or "").lower() if char.isalnum())


def eclass_similarity(candidate: Any, target: str) -> float:
    candidate_id = normalize_eclass_id(candidate)
    target_id = normalize_eclass_id(target)
    if not candidate_id or not target_id:
        return 0.0

    for prefix_length, score in ((8, 1.0), (6, 0.75), (4, 0.5), (2, 0.25)):
        if (
            len(candidate_id) >= prefix_length
            and len(target_id) >= prefix_length
            and candidate_id[:prefix_length] == target_id[:prefix_length]
        ):
            return score
    return 0.0


def normalize_eclass_id(value: Any) -> str:
    text = str(value or "")
    contiguous_match = re.search(r"(?<!\d)(\d{8})(?!\d)", text)
    if contiguous_match:
        return contiguous_match.group(1)

    grouped_match = re.search(
        r"(?<!\d)(\d{2})[-.\s](\d{2})[-.\s](\d{2})[-.\s](\d{2})(?!\d)",
        text,
    )
    if grouped_match:
        return "".join(grouped_match.groups())

    digits = "".join(char for char in text if char.isdigit())
    return digits[:8] if len(digits) >= 8 else digits


def datasheet_embedding_config_hash(cfg: EmbeddingCfg) -> str:
    payload = {
        "provider": cfg.client_cfg.client_name,
        "model": cfg.client_cfg.model,
        "chunk_size": cfg.chunk_size,
        "chunk_overlap": cfg.chunk_overlap,
        "max_pdf_chars": cfg.max_pdf_chars,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def compute_input_datasheet_embedding(
    text: str,
    cfg: EmbeddingCfg,
    embedding_client: EmbeddingClient,
) -> dict[str, Any]:
    clipped_text = (text or "")[: cfg.max_pdf_chars]
    chunks = chunk_datasheet_text(clipped_text, cfg.chunk_size, cfg.chunk_overlap) or [""]
    result = embedding_client.embed(chunks, {"model": cfg.client_cfg.model})
    embeddings = result["embeddings"]
    averaged_embedding = [
        sum(dimension_values) / len(embeddings)
        for dimension_values in zip(*embeddings)
    ]
    return {
        "embedding": averaged_embedding,
        "usage": result.get("usage") or {},
    }


def chunk_datasheet_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    return chunk_text(text, chunk_size, chunk_overlap)


def compatible_datasheet_embedding_rows(
    embedding_rows: list[dict[str, Any]],
    expected_config_hash: str,
) -> list[dict[str, Any]]:
    return [
        row
        for row in embedding_rows
        if str(row.get("embeddingConfigHash") or "") == expected_config_hash
    ]


def datasheet_similarity_scores(
    input_embedding: list[float],
    stored_embedding_rows: list[dict[str, Any]],
) -> dict[str, float]:
    scores = {}
    for row in stored_embedding_rows:
        source_name = str(row.get("sourceName") or "")
        stored_embedding = parse_embedding_json(row.get("embeddingJson"))
        if not source_name or not stored_embedding:
            continue
        score = cosine_similarity(input_embedding, stored_embedding)
        if score is not None:
            scores[source_name] = score
    return scores


def parse_embedding_json(value: Any) -> list[float]:
    if isinstance(value, list):
        payload = value
    else:
        try:
            payload = json.loads(str(value or ""))
        except json.JSONDecodeError:
            return []
    if not isinstance(payload, list):
        return []
    try:
        return [float(item) for item in payload]
    except (TypeError, ValueError):
        return []


def cosine_similarity(left: list[float], right: list[float]) -> float | None:
    if not left or not right or len(left) != len(right):
        return None
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return None
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)

############################################
############# langgraph nodes ##############
############################################

def load_input(state: State) -> State:
    folder_path = Path(state.input_path)
    pdf_path = (folder_path / folder_path.name).with_suffix('.pdf')
    state.pdf_doc = PdfDocument(pdf_path, autoclose=True)
    return state

def normalize_input(state: State) -> State:
    state.preprocessed_text = normalize_pdf(state.pdf_doc)
    return state

def load_definitions(state: State) -> State:
    folder_path = Path(state.input_path)
    aasx_path = next(folder_path.glob("*.aasx"), None)
    if aasx_path is not None:
        object_store = load_aasx_object_store(aasx_path.read_bytes())
        dictionary, _ = extract_technical_data_properties(object_store)
        state.dictionary = dictionary
        state.product_metadata = extract_product_metadata(object_store)
        return state
    else:
        raise ValueError(f"No AASX file found in {folder_path}. Please provide a valid AASX file for definitions.")


def normalize_definitions(state: State) -> State:
    dictionary = state.dictionary
    if dictionary is None:
        raise ValueError("Definitions dictionary is missing or could not be loaded.")

    if isinstance(dictionary, list):
        definitions = dictionary
        return state

    if isinstance(dictionary, dict):
        if 'classes' in dictionary and Path(state.input_path).name in dictionary['classes']:
            property_names = dictionary['classes'][Path(state.input_path).name]['properties']
            state.definitions = [dictionary['properties'][property_name] for property_name in property_names]

        elif 'properties' in dictionary:
            props = dictionary['properties']
            if isinstance(props, list):
                state.definitions = props
            elif isinstance(props, dict):
                state.definitions = list(props.values())
            else:
                raise ValueError("Unsupported format for 'properties' in definitions dictionary.")

        elif 'definitions' in dictionary:
            definitions = dictionary['definitions']
            state.definitions = list(definitions.values()) if isinstance(definitions, dict) else definitions

        elif 'property_definitions' in dictionary:
            definitions = dictionary['property_definitions']
            state.definitions = list(definitions.values()) if isinstance(definitions, dict) else definitions

        elif all(isinstance(value, dict) for value in dictionary.values()):
            state.definitions = list(dictionary.values())
        else:
            raise ValueError("Unable to normalize definitions from the provided dictionary format.")

    for definition in state.definitions:
        definition.pop('values', None)
    return state


def retrieve_icl_examples(state: State, services: Services) -> State:
    cfg = state.cfg.icl_cfg
    definitions = state.definitions or []
    if cfg is None or not definitions or cfg.examples_per_property <= 0:
        state.icl = {}
        return state

    candidate_limit = max(1, len(definitions) * cfg.candidates_per_property)
    target_eclass_id = str((state.product_metadata or {}).get("eclass_id") or "")
    target_manufacturer_name = str((state.product_metadata or {}).get("manufacturer_name") or "")
    candidate_rows = query_property_values_with_metadata(
        definitions,
        uri=DEFAULT_NEO4J_URI,
        user=DEFAULT_NEO4J_USER,
        password=DEFAULT_NEO4J_PASSWORD,
        target_eclass_id=target_eclass_id,
        target_manufacturer_name=target_manufacturer_name,
        limit=candidate_limit,
        technical_data_only=cfg.query_technical_data_only,
        technical_properties_only=cfg.query_technical_properties_only,
        helper_generation_hash=cfg.helper_generation_hash,
        helper_artifact_id=cfg.helper_artifact_id,
        helper_provider=cfg.helper_provider,
        helper_model=cfg.helper_model,
    )
    datasheet_similarity_by_source = {}
    if cfg.use_datasheet_embedding_similarity and candidate_rows:
        expected_embedding_config_hash = datasheet_embedding_config_hash(cfg.embedding_cfg)
        embedding_rows = query_datasheet_embeddings(
            uri=DEFAULT_NEO4J_URI,
            user=DEFAULT_NEO4J_USER,
            password=DEFAULT_NEO4J_PASSWORD,
            embedding_config_hash=expected_embedding_config_hash,
            source_names={str(row.get("sourceName") or "") for row in candidate_rows if row.get("sourceName")},
        )
        embedding_rows = compatible_datasheet_embedding_rows(embedding_rows, expected_embedding_config_hash)
        if embedding_rows:
            embedding_client = services.icl_embedding_client
            if embedding_client is None:
                raise ValueError("ICL datasheet similarity is enabled, but no ICL embedding client is configured.")
            embedding_result = compute_input_datasheet_embedding(
                state.preprocessed_text or "",
                cfg.embedding_cfg,
                embedding_client,
            )
            if embedding_result.get("usage"):
                state.usage.setdefault("icl_embedding", []).append(embedding_result["usage"])
            datasheet_similarity_by_source = datasheet_similarity_scores(
                embedding_result["embedding"],
                embedding_rows,
            )

    state.icl = build_ranked_icl_examples(
        definitions,
        candidate_rows,
        cfg,
        datasheet_similarity_by_source,
        target_eclass_id=target_eclass_id,
        target_manufacturer_name=target_manufacturer_name,
    )
    return state

def load_prompts(state: State) -> State:
    BASE_DIR = Path(__file__).resolve().parent.parent
    RETRIEVE_PROMPT_REGISTRY = load_registry(
        BASE_DIR / "prompts/rag_ie/retrieve.jsonl"
    )
    SCHEMA_BASED_EXTRACTION_PROMPT_REGISTRY = load_registry(
        BASE_DIR / "prompts/rag_ie/schema_based_extraction.jsonl"
    )
    cfg = state.cfg
    state.prompts = {}
    if cfg.query_generator_cfg is not None:
        state.prompts['retrieve_prompt'] = RETRIEVE_PROMPT_REGISTRY[
            cfg.query_generator_cfg.prompt_cfg.type][cfg.query_generator_cfg.prompt_cfg.id]
    state.prompts['schema_based_extraction_prompt'] = SCHEMA_BASED_EXTRACTION_PROMPT_REGISTRY[
            cfg.schema_based_extractor_cfg.prompt.type][cfg.schema_based_extractor_cfg.prompt.id]
    return state

def load_response_schemata(state: State) -> State:
    cfg = state.cfg
    state.schemata = {}
    if cfg.query_generator_cfg is not None:
        state.schemata['retrieve_schema'] = RETRIEVE_SCHEMA_REGISTRY[cfg.query_generator_cfg.response_schema]
    state.schemata['schema_based_extraction_schema'] = AAS_SCHEMA_REGISTRY[cfg.schema_based_extractor_cfg.response_schema]

    return state

def select_query_batch(state: State) -> State:
    if state.query_batch_index is None:
        state.query_batch_index = 0
    if state.cfg.query_generator_cfg.batch_size > 0:
        state.query_batch_start = state.query_batch_index * state.cfg.query_generator_cfg.batch_size
        state.query_batch_end = state.query_batch_start + state.cfg.query_generator_cfg.batch_size
    else:
        state.query_batch_start = 0
        state.query_batch_end = len(state.definitions)
    print(f"query_batch {state.query_batch_index}", len(state.definitions))
    return state

def assemble_query_context(state: State) -> State:
    prompt_instance = "[System Instruction]\n" + state.prompts['retrieve_prompt']

    prompt_instance += "[Product Data]\n" + str(state.definitions[state.query_batch_start:state.query_batch_end])

    batch_size = len(state.definitions[state.query_batch_start:state.query_batch_end])
    schema_instance = state.schemata['retrieve_schema'](batch_size, state.cfg.query_generator_cfg.n_queries_per_property)
    state.prompts['retrieve_prompt_instance'] = prompt_instance
    state.schemata['retrieve_prompt_instance_schema'] = schema_instance
    state.query_batch_index += 1
    return state

def generate_queries(state: State, service: Services) -> State:
    # preload ollama model to avoid timeouts
    if state.cfg.query_generator_cfg.client_cfg.client_name == 'ollama' and state.query_batch_index - 1 == 0:
        service.query_generator.client.client.chat.completions.create(
            model=state.cfg.query_generator_cfg.client_cfg.model,
            messages=[{"role": "system", "content": "Warmup"}],
            max_tokens=1,
            timeout=300)
        
    context = {
        'prompt_instance': state.prompts['retrieve_prompt_instance'],
        'schema_instance': state.schemata['retrieve_prompt_instance_schema']
    }
    result = service.query_generator.generate(context)
    generated_queries = result['queries']
    usage = result['usage']
    if 'queries' in state.usage:
        state.usage['queries'] += [usage]
    else:
        state.usage['queries'] = [usage]

    if state.queries is None:
        state.queries = generated_queries
    else:
        state.queries += generated_queries
    return state

def has_more_query_batches(state: State) -> bool:
    if state.cfg.query_generator_cfg.batch_size <= 0 or state.cfg.query_generator_cfg.batch_size is None:
        return False
    total = len(state.definitions)
    return state.query_batch_end < total



def prepare_document_store(state: State, service: Services) -> State:
    # chunk input document
    chunk_size = state.cfg.query_generator_cfg.embedding_cfg.chunk_size
    chunk_overlap = state.cfg.query_generator_cfg.embedding_cfg.chunk_overlap
    state.chunks = chunk_text(state.preprocessed_text, chunk_size, chunk_overlap)
    # embed chunks
    result = service.embedding_client.embed(state.chunks, {'model': state.cfg.query_generator_cfg.embedding_cfg.client_cfg.model})
    state.embeddings = result['embeddings']
    if 'rag_embeddings' in state.usage:
        state.usage['rag_embeddings'] += [result['usage']]
    else:
        state.usage['rag_embeddings'] = [result['usage']]
    # store in ephemeral chromadb vector store for retrieval during query generation and extraction
    ids = [f"chunk_{i}" for i in range(len(state.chunks))]
    service.ephemeral_vector_store.add(state.chunks, state.embeddings, ids)
    return state

def retrieve_input(state: State, service: Services) -> State:
    top_k = state.cfg.query_generator_cfg.top_k
    n_chunks = state.cfg.query_generator_cfg.n_chunks
    for queries in state.queries:
        raw = service.ephemeral_vector_store.search(
            queries,
            top_k=top_k,
            embedder=service.embedding_client
        )
        if 'usage' in raw:
            usage = raw['usage']
            if 'rag_retrieve' in state.usage:
                state.usage['rag_retrieve'] += [usage]
            else:
                state.usage['rag_retrieve'] = [usage]


        state.retrieved_chunks.append(aggregate_retrievals(raw['docs'], top_k=n_chunks))
        state.retrieved_texts.append("\n\n".join(
            c["document"] for c in state.retrieved_chunks[-1]
        ))
    return state


def select_extraction_batch(state: State) -> State:
    if state.extraction_batch_index is None:
        state.extraction_batch_index = 0
    if state.cfg.schema_based_extractor_cfg.batch_size > 0:
        state.extraction_batch_start = state.extraction_batch_index * state.cfg.schema_based_extractor_cfg.batch_size
        state.extraction_batch_end = state.extraction_batch_start + state.cfg.schema_based_extractor_cfg.batch_size
    else:
        state.extraction_batch_start = 0
        state.extraction_batch_end = len(state.definitions)
    print(f"extraction_batch {state.extraction_batch_index}")
    return state

def assemble_ie_context(state: State) -> State:
    prompt_instance = "[System Instruction]\n" + state.prompts['schema_based_extraction_prompt']
    if state.cfg.query_generator_cfg is None:
        prompt_instance += "\n[Document Text]\n" + state.preprocessed_text

    definitions = state.definitions[state.extraction_batch_start:state.extraction_batch_end]
    definitions_for_prompt = []
    if state.cfg.query_generator_cfg is not None:
        texts = state.retrieved_texts[state.extraction_batch_start:state.extraction_batch_end]
        for i, prop in enumerate(definitions):
            prompt_definition = dict(prop)
            prompt_definition['relevant_texts'] = texts[i]
            add_icl_examples_to_prompt_definition(prompt_definition, state.icl or {})
            definitions_for_prompt.append(prompt_definition)
    else:
        for prop in definitions:
            prompt_definition = dict(prop)
            add_icl_examples_to_prompt_definition(prompt_definition, state.icl or {})
            definitions_for_prompt.append(prompt_definition)

    if any("icl_examples" in prop for prop in definitions_for_prompt):
        prompt_instance += (
            "\n[ICL Guidance Rules]\n"
            "The icl_examples entries are prior examples from other products. Use them only as extraction guidance. "
            "They may be incomplete or erroneous. Never copy an example value unless the same value is explicitly present "
            "in the current document text or retrieved text for that property.\n"
        )

    prompt_instance += "\n[Property Definitions]\n" + json.dumps(definitions_for_prompt)

    
    schema_instance = state.schemata['schema_based_extraction_schema'](property_definitions=definitions)
    state.prompts['schema_based_extraction_prompt_instance'] = prompt_instance
    state.schemata['schema_based_extraction_prompt_instance_schema'] = schema_instance
    state.extraction_batch_index += 1
    return state


def add_icl_examples_to_prompt_definition(
    prompt_definition: dict[str, Any],
    icl_examples_by_property: dict[str, list[dict[str, Any]]],
) -> None:
    examples = icl_examples_by_property.get(definition_icl_lookup_key(prompt_definition), [])
    if examples:
        prompt_definition["icl_examples"] = examples

def extract_information(state: State, service: Services) -> State:
    # preload ollama model to avoid timeouts
    if state.cfg.schema_based_extractor_cfg.client_cfg.client_name == 'ollama' and state.extraction_batch_index - 1 == 0:
        service.schema_based_extractor.client.client.chat.completions.create(
            model=state.cfg.schema_based_extractor_cfg.client_cfg.model,
            messages=[{"role": "system", "content": "Warmup"}],
            max_tokens=1,
            timeout=300)

    context = {
        'prompt_instance': state.prompts['schema_based_extraction_prompt_instance'],
        'schema_instance': state.schemata['schema_based_extraction_prompt_instance_schema']
    }
    result = service.schema_based_extractor.extract(context)
    usage = result['usage']
    if 'extraction' in state.usage:
        state.usage['extraction'] += [usage]
    else:
        state.usage['extraction'] = [usage]

    result = result['parsed'].model_dump()
    if 'properties' in result:
        result = {entry['name']: entry['item']
                for entry in result['properties']}

    state.extraction_result.update(result)
    return state

def has_more_extraction_batches(state: State) -> str:
    if state.cfg.schema_based_extractor_cfg.batch_size <= 0 or state.cfg.schema_based_extractor_cfg.batch_size is None:
        return 'no_more'
    total = len(state.definitions)
    return 'has_more' if state.extraction_batch_end < total else 'no_more'

def postprocessing(state: State) -> State:
    return state

#######################################
############# instantiation ##############
#######################################
def build_services(cfg: Cfg) -> Services:
    if cfg.icl_cfg is not None and cfg.icl_cfg.use_datasheet_embedding_similarity:
        icl_embedding_client = CACHED_EMBEDDING_CLIENT_REGISTRY[
            cfg.icl_cfg.embedding_cfg.client_cfg.client_name
        ]()
    else:
        icl_embedding_client = None

    if cfg.query_generator_cfg is not None:
        embedding_client = CACHED_EMBEDDING_CLIENT_REGISTRY[cfg.query_generator_cfg.embedding_cfg.client_cfg.client_name]()
        ephemeral_vector_store = EphemeralChromaVectorStore(cfg.query_generator_cfg.embedding_cfg.client_cfg)
        query_generator= DefinitionQueryGenerator(cfg)
    else:
        embedding_client = None
        ephemeral_vector_store = None
        query_generator = None
    schema_based_extractor = DefinitionSchemaBasedExtractor(cfg)

    return Services(
    icl_embedding_client=icl_embedding_client,
    embedding_client=embedding_client,
    ephemeral_vector_store=ephemeral_vector_store,
    query_generator=query_generator,
    schema_based_extractor=schema_based_extractor
    )



#######################################
############# langgraphs ##############
#######################################


def build_graph(cfg: Cfg, services: Services):
    prepare_input_builder = StateGraph(State)
    prepare_input_builder.add_node(load_input)
    prepare_input_builder.add_node(normalize_input)
    prepare_input_builder.add_edge(START,'load_input')
    prepare_input_builder.add_edge('load_input','normalize_input')
    prepare_input = prepare_input_builder.compile()


    prepare_definitions_builder = StateGraph(State)
    prepare_definitions_builder.add_node(load_definitions)
    prepare_definitions_builder.add_node(normalize_definitions)
    prepare_definitions_builder.add_edge(START,'load_definitions')
    prepare_definitions_builder.add_edge('load_definitions','normalize_definitions')
    prepare_definitions = prepare_definitions_builder.compile()


    prepare_context_builder = StateGraph(State)
    prepare_context_builder.add_node(load_prompts)
    prepare_context_builder.add_node(load_response_schemata)
    prepare_context_builder.add_node('prepare_input', prepare_input)
    prepare_context_builder.add_node('prepare_definitions', prepare_definitions)

    if cfg.icl_cfg is not None:
        prepare_context_builder.add_node('retrieve_icl_examples',lambda s: retrieve_icl_examples(s,services))
        prepare_context_builder.add_edge('prepare_input','retrieve_icl_examples')
        prepare_context_builder.add_edge('prepare_definitions','retrieve_icl_examples')
    prepare_context_builder.add_edge(START,'load_prompts')
    prepare_context_builder.add_edge(START,'load_response_schemata')
    prepare_context_builder.add_edge(START,'prepare_input')
    prepare_context_builder.add_edge(START,'prepare_definitions')

    prepare_context = prepare_context_builder.compile()

    if cfg.query_generator_cfg is not None:
        batched_query_generation_builder = StateGraph(State)
        batched_query_generation_builder.add_node(select_query_batch)
        batched_query_generation_builder.add_node(assemble_query_context)
        batched_query_generation_builder.add_node('generate_queries',lambda s: generate_queries(s,services))
        batched_query_generation_builder.add_edge(START,'select_query_batch')
        batched_query_generation_builder.add_edge('select_query_batch','assemble_query_context')
        batched_query_generation_builder.add_edge('assemble_query_context','generate_queries')
        batched_query_generation_builder.add_conditional_edges(
                "generate_queries",
                has_more_query_batches,
                {
                    True: "select_query_batch",
                    False: END,
                },
            )
        batched_query_generation = batched_query_generation_builder.compile()

        retrieve_input_builder = StateGraph(State)
        retrieve_input_builder.add_node('batched_query_generation', batched_query_generation)
        retrieve_input_builder.add_node('prepare_document_store',lambda s: prepare_document_store(s,services))
        retrieve_input_builder.add_node('retrieve_input',lambda s: retrieve_input(s,services))
        retrieve_input_builder.add_edge(START,'batched_query_generation')
        retrieve_input_builder.add_edge(START,'prepare_document_store')
        retrieve_input_builder.add_edge('batched_query_generation','retrieve_input')
        retrieve_input_builder.add_edge('prepare_document_store','retrieve_input')
        retrieve = retrieve_input_builder.compile()

    batched_extraction_builder = StateGraph(State)
    batched_extraction_builder.add_node(select_extraction_batch)
    batched_extraction_builder.add_node(assemble_ie_context)
    batched_extraction_builder.add_node('extract_information',lambda s: extract_information(s,services))
    batched_extraction_builder.add_edge(START,'select_extraction_batch')
    batched_extraction_builder.add_edge('select_extraction_batch','assemble_ie_context')
    batched_extraction_builder.add_edge('assemble_ie_context','extract_information')
    batched_extraction_builder.add_conditional_edges(
            "extract_information",
            has_more_extraction_batches,
            {
                'has_more': "select_extraction_batch",
                'no_more': END,
            },
        )
    batched_extraction = batched_extraction_builder.compile()


    extraction_builder = StateGraph(State)
    extraction_builder.add_node('batched_extraction', batched_extraction)
    extraction_builder.add_node(postprocessing)
    extraction_builder.add_edge(START, 'batched_extraction')
    extraction_builder.add_edge('batched_extraction', 'postprocessing')
    extract = extraction_builder.compile()

    graph_builder = StateGraph(State)
    graph_builder.add_node('prepare_context', prepare_context)
    if cfg.query_generator_cfg is not None:
        graph_builder.add_node('retrieve', retrieve)
    graph_builder.add_node('extract', extract)
    graph_builder.add_edge(START, 'prepare_context')
    if cfg.query_generator_cfg is not None:
        graph_builder.add_edge('prepare_context', 'retrieve')
        graph_builder.add_edge('retrieve', 'extract')
    else:
        graph_builder.add_edge('prepare_context', 'extract')
    return graph_builder.compile()
    


def run(input_path: str, cfg: dict | None = None) -> dict:
    if cfg is not None:
        cfg = TypeAdapter(Cfg).validate_python(cfg)
    else:
        print("Config is None. Running default Config.")
        cfg = Cfg()

    services = build_services(cfg)
    graph = build_graph(cfg, services)
    result = graph.invoke(
        {
            "input_path": input_path,
            "cfg": cfg,
        }
    )

    return result
