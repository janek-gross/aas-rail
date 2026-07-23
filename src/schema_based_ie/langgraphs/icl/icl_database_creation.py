"""
Generic helpers for building an ICL RDF database from AASX files.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence
from urllib.parse import quote

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field, TypeAdapter
from rdflib import Graph, RDF, URIRef

from schema_based_ie.model_clients.client_configs import (
    DatasheetEmbeddingCfg,
    DatasheetEmbeddingClientCfg,
    EmbeddingCfg,
)

from .aasx_io import (
    aasx_bytes_to_turtle,
    collect_property_records_from_object_store,
    filter_property_records_to_technical_data,
    filter_property_records_to_technical_properties,
    load_aasx_object_store,
)
from .extraction_helper_pipeline import (
    DEFAULT_EXTRACTION_HELPER_INSTRUCTION,
    ExtractionHelperCfg,
    load_pdf_text_from_bytes,
    run_extraction_helper_pipeline,
)
from .extraction_helper_models import ExtractionHelperRunResult, RdfPropertyRecord
from .rdf_extraction_helper import (
    SCHEMA_IE,
    add_datasheet_embedding_to_turtle,
    add_extraction_instructions_to_turtle,
    add_helper_artifact_ids_to_turtle,
    add_property_record_ids_to_turtle,
)
from .rdf_property_grouping import group_property_records
from .neo4j_connection import (
    DEFAULT_NEO4J_IMPORT_DIR,
    DEFAULT_NEO4J_IMPORT_URI_ROOT,
    DEFAULT_NEO4J_PASSWORD,
    DEFAULT_NEO4J_URI,
    DEFAULT_NEO4J_USER,
    connect_neo4j,
)
from .rdf_export import object_store_to_turtle


_connect_neo4j = connect_neo4j

DEFAULT_ICL_EXAMPLES_DIR = Path("/home/aas-rail/data/datasets/icl_examples")
ICL_CACHE_DIR_NAME = "icl_cache"


class TurtleExport(BaseModel):
    """Result of converting one AASX file to Turtle."""

    source_name: str
    ttl_name: str
    status: str
    pdf_name: Optional[str] = None
    turtle: Optional[str] = None
    triple_count: int = 0
    property_record_count: int = 0
    instruction_count: int = 0
    linked_instruction_count: int = 0
    instruction_status: Optional[str] = None
    instruction_usage: list[dict[str, Any]] = Field(default_factory=list)
    instruction_errors: list[str] = Field(default_factory=list)
    embedding_status: Optional[str] = None
    embedding_usage: dict[str, Any] = Field(default_factory=dict)
    embedding_error: Optional[str] = None
    embedding_dimension: int = 0
    embedding_chunk_count: int = 0
    helper_generation_hash: Optional[str] = None
    helper_generation_config: dict[str, Any] = Field(default_factory=dict)
    helper_artifact_ids: list[str] = Field(default_factory=list)
    embedding_config_hash: Optional[str] = None
    cache_hit: bool = False
    cache_path: Optional[str] = None
    metadata_path: Optional[str] = None
    base_turtle: Optional[str] = None
    base_cache_path: Optional[str] = None
    debug_output_path: Optional[str] = None
    error: Optional[str] = None
    output_path: Optional[str] = None


class Neo4jImportResult(BaseModel):
    """Result of importing one Turtle export into Neo4j."""

    source_name: str
    success: bool
    triples_loaded: int = 0
    mode: str = "none"
    message: str = ""


class Neo4jResetResult(BaseModel):
    """Result of clearing the Neo4j RDF database."""

    success: bool
    nodes_deleted: int = 0
    relationships_deleted: int = 0
    message: str = ""


class Neo4jResourceLabelCleanupResult(BaseModel):
    """Result of removing the generic n10s Resource label."""

    success: bool
    labels_removed: int = 0
    message: str = ""


class DatasheetEmbeddingResult(BaseModel):
    """Averaged embedding for one product datasheet."""

    embedding: list[float]
    model: str
    provider: str
    chunk_size: int
    chunk_overlap: int
    max_pdf_chars: int
    text_char_count: int
    chunk_count: int
    usage: dict[str, Any] = Field(default_factory=dict)


class CachedIclSourceMetadata(BaseModel):
    """Metadata stored next to a cached per-source ICL Turtle export."""

    source_name: str
    pdf_name: str | None = None
    aasx_sha256: str
    pdf_sha256: str | None = None
    helper_generation_hash: str | None = None
    helper_generation_config: dict[str, Any] = Field(default_factory=dict)
    embedding_config_hash: str | None = None
    generate_instructions: bool = True
    generate_datasheet_embeddings: bool = False
    technical_data_only: bool = False
    technical_properties_only: bool = False
    instruction_count: int = 0
    linked_instruction_count: int = 0
    property_record_count: int = 0
    triple_count: int = 0
    instruction_errors: list[str] = Field(default_factory=list)


def stable_json_hash(payload: Any) -> str:
    """Hash JSON-serializable metadata with deterministic key ordering."""
    raw = json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 hash of a local file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(text: str) -> str:
    """Return the SHA-256 hash of text using UTF-8 encoding."""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def normalize_extraction_helper_cfg(
    cfg: ExtractionHelperCfg | Mapping[str, Any] | None = None,
) -> ExtractionHelperCfg:
    """Validate extraction-helper configuration with defaults."""
    return TypeAdapter(ExtractionHelperCfg).validate_python(cfg or {})


def extraction_helper_generation_config(
    cfg: ExtractionHelperCfg | Mapping[str, Any] | None = None,
    instruction: str = DEFAULT_EXTRACTION_HELPER_INSTRUCTION,
    technical_data_only: bool = False,
    technical_properties_only: bool = False,
) -> dict[str, Any]:
    """Return stable, non-secret metadata that identifies helper generation settings."""
    cfg_model = normalize_extraction_helper_cfg(cfg)
    cfg_payload = cfg_model.model_dump(mode="json", exclude={"debug_output_path"})
    return {
        **cfg_payload,
        "instruction_hash": text_sha256(instruction),
        "technical_data_only": bool(technical_data_only),
        "technical_properties_only": bool(technical_properties_only),
    }


def extraction_helper_generation_hash(
    cfg: ExtractionHelperCfg | Mapping[str, Any] | None = None,
    instruction: str = DEFAULT_EXTRACTION_HELPER_INSTRUCTION,
    technical_data_only: bool = False,
    technical_properties_only: bool = False,
) -> str:
    """Return the stable hash for one helper-generation configuration."""
    config = extraction_helper_generation_config(
        cfg,
        instruction=instruction,
        technical_data_only=technical_data_only,
        technical_properties_only=technical_properties_only,
    )
    return stable_json_hash({**config, "instruction": instruction})


def extraction_helper_generation_metadata(
    cfg: ExtractionHelperCfg | Mapping[str, Any] | None = None,
    instruction: str = DEFAULT_EXTRACTION_HELPER_INSTRUCTION,
    technical_data_only: bool = False,
    technical_properties_only: bool = False,
) -> dict[str, Any]:
    """Return RDF-ready helper-generation metadata."""
    config = extraction_helper_generation_config(
        cfg,
        instruction=instruction,
        technical_data_only=technical_data_only,
        technical_properties_only=technical_properties_only,
    )
    cfg_model = normalize_extraction_helper_cfg(cfg)
    helper_hash = extraction_helper_generation_hash(
        cfg_model,
        instruction=instruction,
        technical_data_only=technical_data_only,
        technical_properties_only=technical_properties_only,
    )
    return {
        "helper_generation_hash": helper_hash,
        "helper_generation_config": config,
        "helper_generation_config_json": json.dumps(config, sort_keys=True, separators=(",", ":")),
        "helper_provider": cfg_model.client_cfg.client_name,
        "helper_model": cfg_model.client_cfg.model,
        "helper_instruction_hash": config["instruction_hash"],
        "helper_temperature": cfg_model.temperature,
        "helper_batch_size": cfg_model.batch_size,
        "helper_max_pdf_chars": cfg_model.max_pdf_chars,
        "helper_technical_data_only": bool(technical_data_only),
        "helper_technical_properties_only": bool(technical_properties_only),
    }


def datasheet_embedding_config_hash(cfg: EmbeddingCfg | Mapping[str, Any] | None = None) -> str:
    """Return the stable hash for one datasheet-embedding configuration."""
    embedding_cfg = cfg if isinstance(cfg, EmbeddingCfg) else EmbeddingCfg.model_validate(cfg or {})
    payload = {
        "provider": embedding_cfg.client_cfg.client_name,
        "model": embedding_cfg.client_cfg.model,
        "chunk_size": embedding_cfg.chunk_size,
        "chunk_overlap": embedding_cfg.chunk_overlap,
        "max_pdf_chars": embedding_cfg.max_pdf_chars,
    }
    return stable_json_hash(payload)


def convert_aasx_files_to_turtle(files: Iterable[tuple[str, bytes]]) -> list[TurtleExport]:
    """Convert a collection of named AASX byte payloads to Turtle exports."""
    results: list[TurtleExport] = []

    for source_name, file_bytes in files:
        ttl_name = f"{Path(source_name).stem}.ttl"
        try:
            turtle = aasx_bytes_to_turtle(file_bytes)
            graph = Graph()
            graph.parse(data=turtle, format="turtle")
            results.append(
                TurtleExport(
                    source_name=source_name,
                    ttl_name=ttl_name,
                    status="Converted",
                    turtle=turtle,
                    triple_count=len(graph),
                )
            )
        except Exception as exc:
            results.append(
                TurtleExport(
                    source_name=source_name,
                    ttl_name=ttl_name,
                    status="Error",
                    error=str(exc),
                )
            )

    return results


def convert_paired_aasx_pdf_files_to_turtle(
    aasx_files: Iterable[tuple[str, bytes]],
    pdf_files: Iterable[tuple[str, bytes]],
    instruction_cfg: ExtractionHelperCfg | Mapping[str, Any] | None = None,
    embedding_cfg: EmbeddingCfg | Mapping[str, Any] | None = None,
    generate_instructions: bool = True,
    generate_datasheet_embeddings: bool = False,
    technical_data_only: bool = False,
    technical_properties_only: bool = False,
    helper_artifact_id: str | None = None,
    helper_artifact_ids: Sequence[str] | None = None,
    instruction: str = DEFAULT_EXTRACTION_HELPER_INSTRUCTION,
    debug_output_dir: str | Path | None = None,
) -> list[TurtleExport]:
    """Convert AASX files to Turtle and augment them with PDF-derived extraction helpers.

    AASX and PDF files are matched by filename stem.
    """
    pdfs_by_stem = _files_by_stem(pdf_files)
    results: list[TurtleExport] = []
    helper_generation_metadata = (
        extraction_helper_generation_metadata(
            instruction_cfg,
            instruction=instruction,
            technical_data_only=technical_data_only,
            technical_properties_only=technical_properties_only,
        )
        if generate_instructions
        else None
    )
    clean_artifact_ids = _clean_artifact_ids(helper_artifact_ids)
    if helper_artifact_id:
        clean_artifact_ids = _clean_artifact_ids([*clean_artifact_ids, helper_artifact_id])
    embedding_config_hash = datasheet_embedding_config_hash(embedding_cfg) if generate_datasheet_embeddings else None

    for source_name, file_bytes in aasx_files:
        print(f"Processing AASX file: {source_name}")
        ttl_name = f"{Path(source_name).stem}.ttl"
        pdf_payload = pdfs_by_stem.get(Path(source_name).stem)
        if (generate_instructions or generate_datasheet_embeddings) and pdf_payload is None:
            results.append(
                TurtleExport(
                    source_name=source_name,
                    ttl_name=ttl_name,
                    status="Error",
                    error=f"No matching PDF found for filename stem '{Path(source_name).stem}'.",
                )
            )
            continue

        embedding_status: str | None = None
        embedding_error: str | None = None
        embedding_dimension = 0
        embedding_chunk_count = 0
        try:
            object_store = load_aasx_object_store(file_bytes)
            all_property_records = collect_property_records_from_object_store(object_store, source_name=source_name)
            base_turtle = add_property_record_ids_to_turtle(
                object_store_to_turtle(object_store),
                all_property_records,
            )
            turtle = base_turtle
            property_record_count = 0
            instruction_count = 0
            linked_instruction_count = 0
            instruction_status = "Skipped"
            instruction_usage: list[dict[str, Any]] = []
            instruction_errors: list[str] = []
            embedding_status = "Skipped"
            embedding_usage: dict[str, Any] = {}
            embedding_error = None
            embedding_dimension = 0
            embedding_chunk_count = 0
            pdf_name = pdf_payload[0] if pdf_payload else None
            pdf_text = load_pdf_text_from_bytes(pdf_payload[1]) if pdf_payload else None
            debug_output_path = _debug_output_path(debug_output_dir, source_name)

            if generate_instructions and pdf_payload is not None:
                property_records = all_property_records
                if technical_properties_only:
                    property_records = filter_property_records_to_technical_properties(property_records)
                elif technical_data_only:
                    property_records = filter_property_records_to_technical_data(property_records)
                property_record_count = len(property_records)
                if property_records:
                    instruction_result = create_extraction_helpers_for_property_records(
                        pdf_text=pdf_text or "",
                        property_records=property_records,
                        cfg=_instruction_cfg_with_debug_path(instruction_cfg, debug_output_path),
                        instruction=instruction,
                    )
                    turtle, linked_instruction_count = add_extraction_instructions_to_turtle(
                        turtle_data=turtle,
                        property_records=property_records,
                        instructions=instruction_result.instructions,
                        source_name=source_name,
                        pdf_name=pdf_name,
                        helper_generation_metadata=helper_generation_metadata,
                        helper_artifact_ids=clean_artifact_ids,
                    )
                    instruction_count = len(instruction_result.instructions)
                    instruction_usage = instruction_result.usage
                    instruction_errors = instruction_result.errors
                    debug_output_path = instruction_result.debug_output_path or debug_output_path
                    instruction_status = "Generated"
                    if instruction_errors:
                        instruction_status = f"Generated with {len(instruction_errors)} error(s)"
                else:
                    instruction_status = (
                        "No TechnicalData value-bearing properties found"
                        if technical_data_only
                        else "No value-bearing properties found"
                    )

            if generate_datasheet_embeddings and pdf_payload is not None:
                try:
                    embedding_result = create_datasheet_embedding(pdf_text or "", cfg=embedding_cfg)
                    turtle = add_datasheet_embedding_to_turtle(
                        turtle_data=turtle,
                        source_name=source_name,
                        pdf_name=pdf_name,
                        embedding=embedding_result.embedding,
                        embedding_model=embedding_result.model,
                        embedding_provider=embedding_result.provider,
                        embedding_chunk_size=embedding_result.chunk_size,
                        embedding_chunk_overlap=embedding_result.chunk_overlap,
                        embedding_max_pdf_chars=embedding_result.max_pdf_chars,
                        text_char_count=embedding_result.text_char_count,
                        chunk_count=embedding_result.chunk_count,
                    )
                    embedding_status = "Generated"
                    embedding_usage = embedding_result.usage
                    embedding_dimension = len(embedding_result.embedding)
                    embedding_chunk_count = embedding_result.chunk_count
                except Exception as exc:
                    embedding_status = "Failed"
                    embedding_error = str(exc)
                    raise RuntimeError(f"Datasheet embedding generation failed: {exc}") from exc

            graph = Graph()
            graph.parse(data=turtle, format="turtle")
            results.append(
                TurtleExport(
                    source_name=source_name,
                    ttl_name=ttl_name,
                    status="Converted",
                    pdf_name=pdf_name,
                    turtle=turtle,
                    triple_count=len(graph),
                    property_record_count=property_record_count,
                    instruction_count=instruction_count,
                    linked_instruction_count=linked_instruction_count,
                    instruction_status=instruction_status,
                    instruction_usage=instruction_usage,
                    instruction_errors=instruction_errors,
                    embedding_status=embedding_status,
                    embedding_usage=embedding_usage,
                    embedding_error=embedding_error,
                    embedding_dimension=embedding_dimension,
                    embedding_chunk_count=embedding_chunk_count,
                    helper_generation_hash=(
                        helper_generation_metadata.get("helper_generation_hash")
                        if helper_generation_metadata
                        else None
                    ),
                    helper_generation_config=(
                        helper_generation_metadata.get("helper_generation_config")
                        if helper_generation_metadata
                        else {}
                    ),
                    helper_artifact_ids=clean_artifact_ids,
                    embedding_config_hash=embedding_config_hash,
                    base_turtle=base_turtle,
                    debug_output_path=debug_output_path,
                )
            )
        except Exception as exc:
            results.append(
                TurtleExport(
                    source_name=source_name,
                    ttl_name=ttl_name,
                    status="Error",
                    pdf_name=pdf_payload[0] if pdf_payload else None,
                    embedding_status=embedding_status,
                    embedding_error=embedding_error,
                    embedding_dimension=embedding_dimension,
                    embedding_chunk_count=embedding_chunk_count,
                    helper_generation_hash=(
                        helper_generation_metadata.get("helper_generation_hash")
                        if helper_generation_metadata
                        else None
                    ),
                    helper_generation_config=(
                        helper_generation_metadata.get("helper_generation_config")
                        if helper_generation_metadata
                        else {}
                    ),
                    helper_artifact_ids=clean_artifact_ids,
                    embedding_config_hash=embedding_config_hash,
                    debug_output_path=_debug_output_path(debug_output_dir, source_name),
                    error=str(exc),
                )
            )

    return results


def write_turtle_exports(exports: Sequence[TurtleExport], output_dir: str | Path) -> list[TurtleExport]:
    """Write successful Turtle exports to an output directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    written_exports = []

    for export in exports:
        if not export.turtle:
            written_exports.append(export)
            continue
        ttl_path = output_path / export.ttl_name
        ttl_path.write_text(export.turtle, encoding="utf-8")
        written_exports.append(export.model_copy(update={"output_path": str(ttl_path)}))

    return written_exports


def discover_icl_example_sample_dirs(
    examples_root: str | Path = DEFAULT_ICL_EXAMPLES_DIR,
    sample_names: Sequence[str] | None = None,
    sample_limit: int | None = None,
) -> list[Path]:
    """Find sample folders that contain at least one AASX/PDF pair.

    When a limit is given, sample folders are selected in a deterministic
    company-balanced round robin. The company is the folder-name prefix before
    the first underscore, e.g. ``Festo_1254054`` belongs to ``Festo``.
    """
    root = Path(examples_root)
    if sample_names:
        candidates = [root / name for name in sample_names]
    else:
        candidates = sorted(path for path in root.iterdir() if path.is_dir())

    sample_dirs = [
        path
        for path in candidates
        if path.is_dir() and _sample_file(path, ".aasx") is not None and _sample_file(path, ".pdf") is not None
    ]
    if sample_limit is None:
        return sample_dirs
    return _balanced_sample_dirs_by_company(sample_dirs, sample_limit)


def build_cached_icl_turtle_exports_from_sample_dirs(
    sample_dirs: Sequence[str | Path],
    instruction_cfg: ExtractionHelperCfg | Mapping[str, Any] | None = None,
    embedding_cfg: EmbeddingCfg | Mapping[str, Any] | None = None,
    generate_instructions: bool = True,
    generate_datasheet_embeddings: bool = False,
    technical_data_only: bool = False,
    technical_properties_only: bool = False,
    helper_artifact_id: str | None = None,
    helper_artifact_ids: Sequence[str] | None = None,
    instruction: str = DEFAULT_EXTRACTION_HELPER_INSTRUCTION,
    force_rebuild: bool = False,
    debug_output_dir: str | Path | None = None,
) -> list[TurtleExport]:
    """Build or reuse cached per-source Turtle exports from existing sample folders."""
    return [
        build_cached_icl_turtle_export_from_sample_dir(
            sample_dir,
            instruction_cfg=instruction_cfg,
            embedding_cfg=embedding_cfg,
            generate_instructions=generate_instructions,
            generate_datasheet_embeddings=generate_datasheet_embeddings,
            technical_data_only=technical_data_only,
            technical_properties_only=technical_properties_only,
            helper_artifact_id=helper_artifact_id,
            helper_artifact_ids=helper_artifact_ids,
            instruction=instruction,
            force_rebuild=force_rebuild,
            debug_output_dir=debug_output_dir,
        )
        for sample_dir in sample_dirs
    ]


def build_cached_icl_turtle_exports_from_examples_dir(
    examples_root: str | Path = DEFAULT_ICL_EXAMPLES_DIR,
    sample_names: Sequence[str] | None = None,
    sample_limit: int | None = None,
    **kwargs: Any,
) -> list[TurtleExport]:
    """Discover sample folders under an examples directory and build cached exports."""
    sample_dirs = discover_icl_example_sample_dirs(
        examples_root=examples_root,
        sample_names=sample_names,
        sample_limit=sample_limit,
    )
    return build_cached_icl_turtle_exports_from_sample_dirs(sample_dirs, **kwargs)


def build_cached_icl_turtle_export_from_sample_dir(
    sample_dir: str | Path,
    instruction_cfg: ExtractionHelperCfg | Mapping[str, Any] | None = None,
    embedding_cfg: EmbeddingCfg | Mapping[str, Any] | None = None,
    generate_instructions: bool = True,
    generate_datasheet_embeddings: bool = False,
    technical_data_only: bool = False,
    technical_properties_only: bool = False,
    helper_artifact_id: str | None = None,
    helper_artifact_ids: Sequence[str] | None = None,
    instruction: str = DEFAULT_EXTRACTION_HELPER_INSTRUCTION,
    force_rebuild: bool = False,
    debug_output_dir: str | Path | None = None,
) -> TurtleExport:
    """Build or reuse the cached Turtle export for one sample folder."""
    folder = Path(sample_dir)
    aasx_path = _required_sample_file(folder, ".aasx")
    pdf_path = _required_sample_file(folder, ".pdf")
    aasx_hash = file_sha256(aasx_path)
    pdf_hash = file_sha256(pdf_path)
    helper_metadata = (
        extraction_helper_generation_metadata(
            instruction_cfg,
            instruction=instruction,
            technical_data_only=technical_data_only,
            technical_properties_only=technical_properties_only,
        )
        if generate_instructions
        else None
    )
    helper_hash = helper_metadata.get("helper_generation_hash") if helper_metadata else None
    embedding_hash = datasheet_embedding_config_hash(embedding_cfg) if generate_datasheet_embeddings else None
    source_cache_hash = stable_json_hash(
        {
            "aasx_sha256": aasx_hash,
            "pdf_sha256": pdf_hash,
            "helper_generation_hash": helper_hash,
            "embedding_config_hash": embedding_hash,
            "generate_instructions": bool(generate_instructions),
            "generate_datasheet_embeddings": bool(generate_datasheet_embeddings),
            "technical_data_only": bool(technical_data_only),
            "technical_properties_only": bool(technical_properties_only),
        }
    )
    cache_dir = folder / ICL_CACHE_DIR_NAME
    cache_dir.mkdir(parents=True, exist_ok=True)
    base_ttl_path = cache_dir / f"base.{aasx_hash}.ttl"
    ttl_path = cache_dir / f"full.{source_cache_hash}.ttl"
    metadata_path = cache_dir / f"full.{source_cache_hash}.json"
    graph_result_path = (
        _cache_graph_invoke_result_path(cache_dir, source_cache_hash)
        if generate_instructions
        else None
    )
    if force_rebuild:
        ttl_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
        if graph_result_path is not None:
            graph_result_path.unlink(missing_ok=True)
    clean_artifact_ids = _clean_artifact_ids(helper_artifact_ids)
    if helper_artifact_id:
        clean_artifact_ids = _clean_artifact_ids([*clean_artifact_ids, helper_artifact_id])

    base_turtle = (
        base_ttl_path.read_text(encoding="utf-8")
        if base_ttl_path.exists() and not force_rebuild
        else None
    )

    if not force_rebuild and ttl_path.exists() and metadata_path.exists():
        if base_turtle is None:
            base_turtle = create_base_turtle_for_aasx_bytes(aasx_path.read_bytes(), source_name=aasx_path.name)
            base_ttl_path.write_text(base_turtle, encoding="utf-8")
        metadata = CachedIclSourceMetadata.model_validate_json(metadata_path.read_text(encoding="utf-8"))
        cached_instruction_errors = _cached_instruction_errors(metadata, graph_result_path)
        if not (generate_instructions and cached_instruction_errors):
            turtle = ttl_path.read_text(encoding="utf-8")
            turtle = add_helper_artifact_ids_to_turtle(turtle, clean_artifact_ids)
            graph = Graph()
            graph.parse(data=turtle, format="turtle")
            return TurtleExport(
                source_name=aasx_path.name,
                ttl_name=ttl_path.name,
                status="Cached",
                pdf_name=pdf_path.name,
                turtle=turtle,
                triple_count=len(graph),
                property_record_count=metadata.property_record_count,
                instruction_count=metadata.instruction_count,
                linked_instruction_count=metadata.linked_instruction_count,
                instruction_status="Cached" if generate_instructions else "Skipped",
                instruction_errors=cached_instruction_errors,
                embedding_status="Cached" if generate_datasheet_embeddings else "Skipped",
                helper_generation_hash=helper_hash,
                helper_generation_config=helper_metadata.get("helper_generation_config") if helper_metadata else {},
                helper_artifact_ids=clean_artifact_ids,
                embedding_config_hash=embedding_hash,
                cache_hit=True,
                cache_path=str(ttl_path),
                metadata_path=str(metadata_path),
                base_turtle=base_turtle,
                base_cache_path=str(base_ttl_path),
                debug_output_path=(
                    str(graph_result_path)
                    if graph_result_path is not None and graph_result_path.exists()
                    else None
                ),
            )

    conversion_instruction_cfg = instruction_cfg
    if (
        graph_result_path is not None
        and not _instruction_cfg_debug_output_path(instruction_cfg)
        and debug_output_dir is None
    ):
        conversion_instruction_cfg = _instruction_cfg_with_debug_path(
            instruction_cfg,
            str(graph_result_path),
        )

    converted = convert_paired_aasx_pdf_files_to_turtle(
        [(aasx_path.name, aasx_path.read_bytes())],
        [(pdf_path.name, pdf_path.read_bytes())],
        instruction_cfg=conversion_instruction_cfg,
        embedding_cfg=embedding_cfg,
        generate_instructions=generate_instructions,
        generate_datasheet_embeddings=generate_datasheet_embeddings,
        technical_data_only=technical_data_only,
        technical_properties_only=technical_properties_only,
        helper_artifact_ids=[],
        instruction=instruction,
        debug_output_dir=debug_output_dir,
    )[0]
    if graph_result_path is not None and converted.debug_output_path:
        _copy_debug_output_to_cache(converted.debug_output_path, graph_result_path)
    cached_debug_output_path = (
        str(graph_result_path)
        if graph_result_path is not None and graph_result_path.exists()
        else converted.debug_output_path
    )
    if not converted.turtle:
        return converted.model_copy(
            update={
                "helper_generation_hash": helper_hash,
                "helper_generation_config": helper_metadata.get("helper_generation_config") if helper_metadata else {},
                "embedding_config_hash": embedding_hash,
                "cache_path": str(ttl_path),
                "metadata_path": str(metadata_path),
                "base_cache_path": str(base_ttl_path),
                "debug_output_path": cached_debug_output_path,
            }
        )

    base_turtle = converted.base_turtle or create_base_turtle_for_aasx_bytes(
        aasx_path.read_bytes(),
        source_name=aasx_path.name,
    )
    base_ttl_path.write_text(base_turtle, encoding="utf-8")
    ttl_path.write_text(converted.turtle, encoding="utf-8")
    metadata = CachedIclSourceMetadata(
        source_name=aasx_path.name,
        pdf_name=pdf_path.name,
        aasx_sha256=aasx_hash,
        pdf_sha256=pdf_hash,
        helper_generation_hash=helper_hash,
        helper_generation_config=helper_metadata.get("helper_generation_config") if helper_metadata else {},
        embedding_config_hash=embedding_hash,
        generate_instructions=generate_instructions,
        generate_datasheet_embeddings=generate_datasheet_embeddings,
        technical_data_only=technical_data_only,
        technical_properties_only=technical_properties_only,
        instruction_count=converted.instruction_count,
        linked_instruction_count=converted.linked_instruction_count,
        property_record_count=converted.property_record_count,
        triple_count=converted.triple_count,
        instruction_errors=converted.instruction_errors,
    )
    metadata_path.write_text(metadata.model_dump_json(indent=2), encoding="utf-8")

    turtle = add_helper_artifact_ids_to_turtle(converted.turtle, clean_artifact_ids)
    graph = Graph()
    graph.parse(data=turtle, format="turtle")
    return converted.model_copy(
        update={
            "ttl_name": ttl_path.name,
            "turtle": turtle,
            "triple_count": len(graph),
            "helper_generation_hash": helper_hash,
            "helper_generation_config": helper_metadata.get("helper_generation_config") if helper_metadata else {},
            "helper_artifact_ids": clean_artifact_ids,
            "embedding_config_hash": embedding_hash,
            "cache_hit": False,
            "cache_path": str(ttl_path),
            "metadata_path": str(metadata_path),
            "base_turtle": base_turtle,
            "base_cache_path": str(base_ttl_path),
            "debug_output_path": cached_debug_output_path,
        }
    )


def create_extraction_helpers_for_property_records(
    pdf_text: str,
    property_records: Sequence[RdfPropertyRecord | Mapping[str, Any]],
    cfg: ExtractionHelperCfg | Mapping[str, Any] | None = None,
    instruction: str = DEFAULT_EXTRACTION_HELPER_INSTRUCTION,
) -> ExtractionHelperRunResult:
    """Group RDF property records and generate extraction helpers for them."""
    return run_extraction_helper_pipeline(
        pdf_text=pdf_text,
        property_groups=group_property_records(property_records),
        cfg=cfg,
        instruction=instruction,
    )


def create_base_turtle_for_aasx_bytes(file_bytes: bytes, source_name: str | None = None) -> str:
    """Create base AAS Turtle annotated with stable property record IDs."""
    object_store = load_aasx_object_store(file_bytes)
    property_records = collect_property_records_from_object_store(object_store, source_name=source_name)
    return add_property_record_ids_to_turtle(object_store_to_turtle(object_store), property_records)


def create_datasheet_embedding(
    pdf_text: str,
    cfg: EmbeddingCfg | Mapping[str, Any] | None = None,
) -> DatasheetEmbeddingResult:
    """Create an averaged embedding for a product datasheet text."""
    from schema_based_ie.model_clients.llm_clients import CACHED_EMBEDDING_CLIENT_REGISTRY

    embedding_cfg = (
        cfg
        if isinstance(cfg, EmbeddingCfg)
        else EmbeddingCfg.model_validate(cfg or {})
    )
    text = (pdf_text or "")[: embedding_cfg.max_pdf_chars]
    chunks = _chunk_text(text, embedding_cfg.chunk_size, embedding_cfg.chunk_overlap)
    if not chunks:
        chunks = [""]

    client = CACHED_EMBEDDING_CLIENT_REGISTRY[embedding_cfg.client_cfg.client_name]()
    result = client.embed(chunks, {"model": embedding_cfg.client_cfg.model})
    embeddings = result["embeddings"]
    averaged_embedding = [
        sum(dimension_values) / len(embeddings)
        for dimension_values in zip(*embeddings)
    ]
    return DatasheetEmbeddingResult(
        embedding=averaged_embedding,
        model=embedding_cfg.client_cfg.model,
        provider=embedding_cfg.client_cfg.client_name,
        chunk_size=embedding_cfg.chunk_size,
        chunk_overlap=embedding_cfg.chunk_overlap,
        max_pdf_chars=embedding_cfg.max_pdf_chars,
        text_char_count=len(text),
        chunk_count=len(chunks),
        usage=result.get("usage") or {},
    )


def _sample_file(sample_dir: Path, suffix: str) -> Path | None:
    matches = sorted(path for path in sample_dir.iterdir() if path.is_file() and path.suffix.lower() == suffix)
    if not matches:
        return None
    stem_match = [path for path in matches if path.stem == sample_dir.name]
    return stem_match[0] if stem_match else matches[0]


def _balanced_sample_dirs_by_company(sample_dirs: Sequence[Path], sample_limit: int) -> list[Path]:
    if sample_limit <= 0:
        return []

    by_company: dict[str, list[Path]] = defaultdict(list)
    for sample_dir in sorted(sample_dirs):
        by_company[_sample_company_prefix(sample_dir)].append(sample_dir)

    selected: list[Path] = []
    company_names = sorted(by_company)
    while len(selected) < sample_limit:
        added_in_round = False
        for company_name in company_names:
            company_samples = by_company[company_name]
            if not company_samples:
                continue
            selected.append(company_samples.pop(0))
            added_in_round = True
            if len(selected) == sample_limit:
                return selected
        if not added_in_round:
            return selected
    return selected


def _sample_company_prefix(sample_dir: Path) -> str:
    return sample_dir.name.split("_", 1)[0]


def _required_sample_file(sample_dir: Path, suffix: str) -> Path:
    path = _sample_file(sample_dir, suffix)
    if path is None:
        raise FileNotFoundError(f"No {suffix} file found in sample folder {sample_dir}.")
    return path


def _clean_artifact_ids(values: Sequence[Any] | None) -> list[str]:
    cleaned = []
    seen = set()
    for value in values or []:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        cleaned.append(text)
    return cleaned


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, BaseModel):
        return _json_safe(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
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


def _files_by_stem(files: Iterable[tuple[str, bytes]]) -> dict[str, tuple[str, bytes]]:
    """Return named file payloads keyed by filename stem."""
    return {Path(name).stem: (name, payload) for name, payload in files}


def _instruction_cfg_with_debug_path(
    instruction_cfg: ExtractionHelperCfg | Mapping[str, Any] | None,
    debug_output_path: str | None,
) -> ExtractionHelperCfg | Mapping[str, Any] | None:
    if not debug_output_path:
        return instruction_cfg
    if isinstance(instruction_cfg, ExtractionHelperCfg):
        return instruction_cfg.model_copy(update={"debug_output_path": debug_output_path})
    return {**dict(instruction_cfg or {}), "debug_output_path": debug_output_path}


def _instruction_cfg_debug_output_path(
    instruction_cfg: ExtractionHelperCfg | Mapping[str, Any] | None,
) -> str | None:
    if isinstance(instruction_cfg, ExtractionHelperCfg):
        return instruction_cfg.debug_output_path
    if isinstance(instruction_cfg, Mapping):
        value = instruction_cfg.get("debug_output_path")
        return str(value) if value else None
    return None


def _cached_instruction_errors(
    metadata: CachedIclSourceMetadata,
    graph_result_path: Path | None,
) -> list[str]:
    """Return instruction-generation errors recorded in cache metadata or debug state."""
    errors: list[str] = []
    errors.extend(str(error) for error in metadata.instruction_errors if error)
    if graph_result_path is None or not graph_result_path.exists():
        return errors

    try:
        payload = json.loads(graph_result_path.read_text(encoding="utf-8"))
    except Exception:
        return errors

    for key in ("generation_errors", "errors"):
        value = payload.get(key)
        if isinstance(value, list):
            errors.extend(str(error) for error in value if error)

    for result in payload.get("raw_generation_results", []):
        if isinstance(result, Mapping) and result.get("error"):
            errors.append(str(result["error"]))

    return list(dict.fromkeys(errors))


def _cache_graph_invoke_result_path(cache_dir: Path, source_cache_hash: str) -> Path:
    return cache_dir / f"full.{source_cache_hash}.graph-invoke.json"


def _copy_debug_output_to_cache(source_path: str | Path, cache_path: Path) -> None:
    source = Path(source_path)
    if not source.exists() or source.resolve() == cache_path.resolve():
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def _debug_output_path(debug_output_dir: str | Path | None, source_name: str) -> str | None:
    if not debug_output_dir:
        return None
    safe_stem = "".join(
        character if character.isalnum() or character in {"-", "_", "."} else "_"
        for character in Path(source_name).stem
    )
    return str(Path(debug_output_dir) / f"{safe_stem}_extraction_instruction_state.json")


def import_turtle_exports_to_neo4j(
    exports: Sequence[TurtleExport | Mapping[str, object]],
    uri: str = DEFAULT_NEO4J_URI,
    user: str = DEFAULT_NEO4J_USER,
    password: str = DEFAULT_NEO4J_PASSWORD,
    neo4j_import_dir: str | Path | None = DEFAULT_NEO4J_IMPORT_DIR,
    neo4j_import_uri_root: str = DEFAULT_NEO4J_IMPORT_URI_ROOT,
    remove_resource_label: bool = True,
) -> list[Neo4jImportResult]:
    """Import all successful Turtle exports into Neo4j."""
    results = []
    for export in exports:
        export_model = export if isinstance(export, TurtleExport) else TurtleExport.model_validate(export)
        if not export_model.turtle:
            results.append(
                Neo4jImportResult(
                    source_name=export_model.source_name,
                    success=False,
                    message=export_model.error or "No Turtle data available.",
                )
            )
            continue

        if export_model.base_turtle:
            results.append(
                import_cached_turtle_export_to_neo4j(
                    export_model,
                    uri=uri,
                    user=user,
                    password=password,
                )
            )
            continue

        turtle_file_uri = _prepare_fetch_file_uri(
            export_model,
            neo4j_import_dir=neo4j_import_dir,
            neo4j_import_uri_root=neo4j_import_uri_root,
        )
        results.append(
            import_turtle_to_neo4j(
                turtle_data=export_model.turtle,
                graph_name=Path(export_model.source_name).stem,
                source_name=export_model.source_name,
                uri=uri,
                user=user,
                password=password,
                turtle_file_uri=turtle_file_uri,
                remove_resource_label=False,
            )
        )

    if remove_resource_label and any(result.success for result in results):
        cleanup_result = remove_neo4j_resource_labels(uri=uri, user=user, password=password)
        results = [
            result.model_copy(update={"message": f"{result.message} {cleanup_result.message}"})
            if result.success
            else result
            for result in results
        ]

    return results


def import_cached_turtle_export_to_neo4j(
    export: TurtleExport | Mapping[str, object],
    uri: str = DEFAULT_NEO4J_URI,
    user: str = DEFAULT_NEO4J_USER,
    password: str = DEFAULT_NEO4J_PASSWORD,
) -> Neo4jImportResult:
    """Import a cached export without re-importing the full AAS graph every time."""
    export_model = export if isinstance(export, TurtleExport) else TurtleExport.model_validate(export)
    if not export_model.turtle:
        return Neo4jImportResult(
            source_name=export_model.source_name,
            success=False,
            message=export_model.error or "No Turtle data available.",
        )
    if not export_model.base_turtle:
        return import_turtle_to_neo4j(
            turtle_data=export_model.turtle,
            graph_name=Path(export_model.source_name).stem,
            source_name=export_model.source_name,
            uri=uri,
            user=user,
            password=password,
            remove_resource_label=False,
        )

    try:
        from neo4j import GraphDatabase
    except ImportError as exc:
        return Neo4jImportResult(
            source_name=export_model.source_name,
            success=False,
            message=f"Neo4j Python driver is not installed: {exc}",
        )

    driver = None
    try:
        driver, connected_uri, connected_user = connect_neo4j(GraphDatabase, uri, user, password)
        with driver.session() as session:
            _ensure_n10s_ready(session)
            base_imported = False
            triples_loaded = 0
            if not _source_base_graph_exists(session, export_model.source_name):
                triples_loaded, _ = _import_with_n10s(session, export_model.base_turtle, None)
                base_imported = True
            _tag_imported_resources(
                session,
                export_model.base_turtle,
                graph_name=Path(export_model.source_name).stem,
                source_name=export_model.source_name,
            )
            helper_result = _upsert_extraction_helpers(session, export_model.turtle, export_model.source_name)
        return Neo4jImportResult(
            source_name=export_model.source_name,
            success=True,
            triples_loaded=triples_loaded,
            mode="base.once+helpers.merge",
            message=(
                f"{'Imported' if base_imported else 'Reused'} base graph on {connected_uri} as {connected_user}; "
                f"merged {helper_result['helpers']} helper node(s), linked {helper_result['links']} property node(s)."
            ),
        )
    except Exception as exc:
        return Neo4jImportResult(source_name=export_model.source_name, success=False, message=str(exc))
    finally:
        if driver is not None:
            driver.close()


def import_turtle_to_neo4j(
    turtle_data: str,
    graph_name: str,
    source_name: str,
    uri: str = DEFAULT_NEO4J_URI,
    user: str = DEFAULT_NEO4J_USER,
    password: str = DEFAULT_NEO4J_PASSWORD,
    turtle_file_uri: str | None = None,
    remove_resource_label: bool = True,
) -> Neo4jImportResult:
    """Import Turtle RDF into Neo4j with neosemantics."""
    try:
        from neo4j import GraphDatabase
    except ImportError as exc:
        return Neo4jImportResult(
            source_name=source_name,
            success=False,
            message=f"Neo4j Python driver is not installed: {exc}",
        )

    driver = None
    try:
        driver, connected_uri, connected_user = connect_neo4j(GraphDatabase, uri, user, password)
        with driver.session() as session:
            _ensure_n10s_ready(session)
            triples_loaded, import_mode = _import_with_n10s(session, turtle_data, turtle_file_uri)
            tagged_resources = _tag_imported_resources(session, turtle_data, graph_name, source_name)
            cleanup_message = ""
            if remove_resource_label:
                labels_removed = _remove_resource_labels(session)
                cleanup_message = f" Removed generic Resource label from {labels_removed} node(s)."
            return Neo4jImportResult(
                source_name=source_name,
                success=True,
                triples_loaded=triples_loaded,
                mode=import_mode,
                message=(
                    f"Imported with {import_mode} on {connected_uri} as {connected_user}; "
                    f"tagged {tagged_resources} RDF resources.{cleanup_message}"
                ),
            )
    except Exception as exc:
        return Neo4jImportResult(source_name=source_name, success=False, message=str(exc))
    finally:
        if driver is not None:
            driver.close()


def reset_neo4j_database(
    uri: str = DEFAULT_NEO4J_URI,
    user: str = DEFAULT_NEO4J_USER,
    password: str = DEFAULT_NEO4J_PASSWORD,
) -> Neo4jResetResult:
    """Delete all Neo4j nodes/relationships and re-create the n10s baseline config."""
    try:
        from neo4j import GraphDatabase
    except ImportError as exc:
        return Neo4jResetResult(success=False, message=f"Neo4j Python driver is not installed: {exc}")

    driver = None
    try:
        driver, connected_uri, connected_user = connect_neo4j(GraphDatabase, uri, user, password)
        with driver.session() as session:
            counts = session.run(
                """
                MATCH (n)
                OPTIONAL MATCH (n)-[r]-()
                RETURN count(DISTINCT n) AS nodes, count(DISTINCT r) AS relationships
                """
            ).single()
            nodes_deleted = int(counts["nodes"]) if counts else 0
            relationships_deleted = int(counts["relationships"]) if counts else 0

            session.run("MATCH (n) DETACH DELETE n").consume()
            _ensure_n10s_ready(session)

        return Neo4jResetResult(
            success=True,
            nodes_deleted=nodes_deleted,
            relationships_deleted=relationships_deleted,
            message=f"Reset Neo4j RDF database on {connected_uri} as {connected_user}.",
        )
    except Exception as exc:
        return Neo4jResetResult(success=False, message=str(exc))
    finally:
        if driver is not None:
            driver.close()


def remove_neo4j_resource_labels(
    uri: str = DEFAULT_NEO4J_URI,
    user: str = DEFAULT_NEO4J_USER,
    password: str = DEFAULT_NEO4J_PASSWORD,
) -> Neo4jResourceLabelCleanupResult:
    """Remove the generic n10s Resource label so typed labels drive Neo4j visualization."""
    try:
        from neo4j import GraphDatabase
    except ImportError as exc:
        return Neo4jResourceLabelCleanupResult(
            success=False,
            message=f"Neo4j Python driver is not installed: {exc}",
        )

    driver = None
    try:
        driver, connected_uri, connected_user = connect_neo4j(GraphDatabase, uri, user, password)
        with driver.session() as session:
            labels_removed = _remove_resource_labels(session)
        return Neo4jResourceLabelCleanupResult(
            success=True,
            labels_removed=labels_removed,
            message=(
                f"Removed generic Resource label from {labels_removed} node(s) "
                f"on {connected_uri} as {connected_user}."
            ),
        )
    except Exception as exc:
        return Neo4jResourceLabelCleanupResult(success=False, message=str(exc))
    finally:
        if driver is not None:
            driver.close()


def _source_base_graph_exists(session, source_name: str) -> bool:
    result = session.run(
        """
        MATCH (resource)
        WHERE $source_name IN coalesce(resource.iclSourceNames, [])
        RETURN count(resource) > 0 AS exists
        """,
        source_name=source_name,
    ).single()
    return bool(result and result["exists"])


def _upsert_extraction_helpers(session, turtle_data: str, source_name: str) -> dict[str, int]:
    helper_payloads = _extraction_helper_payloads_from_turtle(turtle_data)
    helpers = 0
    links = 0
    for payload in helper_payloads:
        result = session.run(
            """
            MERGE (instruction:schemaie__ExtractionInstruction {uri: $uri})
            SET instruction += $properties
            WITH instruction
            WITH instruction,
                 CASE
                     WHEN instruction.`schemaie__helperArtifactId` IS NULL THEN []
                     WHEN instruction.`schemaie__helperArtifactId` IS :: LIST<ANY> THEN toStringList(instruction.`schemaie__helperArtifactId`)
                     ELSE [toString(instruction.`schemaie__helperArtifactId`)]
                 END AS rawExistingArtifactIds
            WITH instruction, [artifactId IN rawExistingArtifactIds WHERE artifactId IS NOT NULL AND trim(artifactId) <> "" | trim(artifactId)] AS existingArtifactIds
            SET instruction.`schemaie__helperArtifactId` = reduce(
                ids = existingArtifactIds,
                artifactId IN $artifact_ids |
                CASE WHEN artifactId IN ids THEN ids ELSE ids + artifactId END
            )
            WITH instruction
            OPTIONAL MATCH (instruction)-[selfRel:schemaie__hasExtractionInstruction]->(instruction)
            FOREACH (rel IN CASE WHEN selfRel IS NULL THEN [] ELSE [selfRel] END | DELETE rel)
            WITH instruction
            UNWIND $record_ids AS recordId
            MATCH (element)
            WHERE $source_name IN coalesce(element.iclSourceNames, [])
              AND any(label IN labels(element)
                  WHERE label ENDS WITH "__Property"
                     OR label ENDS WITH "__MultiLanguageProperty"
                     OR label ENDS WITH "__Range")
              AND any(key IN keys(element)
                  WHERE key ENDS WITH "__propertyRecordId"
                    AND any(value IN CASE
                        WHEN element[key] IS NULL THEN []
                        WHEN element[key] IS :: LIST<ANY> THEN toStringList(element[key])
                        ELSE [toString(element[key])]
                    END WHERE value IS NOT NULL AND value CONTAINS recordId)
              )
            MERGE (element)-[:schemaie__hasExtractionInstruction]->(instruction)
            RETURN count(DISTINCT element) AS linked
            """,
            uri=payload["uri"],
            properties=payload["properties"],
            artifact_ids=payload["artifact_ids"],
            record_ids=payload["record_ids"],
            source_name=source_name,
        ).single()
        helpers += 1
        links += int(result["linked"]) if result else 0
    return {"helpers": helpers, "links": links}


def _extraction_helper_payloads_from_turtle(turtle_data: str) -> list[dict[str, Any]]:
    graph = Graph()
    graph.parse(data=turtle_data, format="turtle")
    payloads = []
    for subject in graph.subjects(RDF.type, SCHEMA_IE.ExtractionInstruction):
        grouped_values: dict[str, list[str]] = {}
        for predicate, value in graph.predicate_objects(subject):
            key = _schemaie_property_key(predicate)
            if key is None:
                continue
            text = str(value).strip()
            if text and text not in grouped_values.setdefault(key, []):
                grouped_values[key].append(text)

        properties = {
            key: values[0] if len(values) == 1 else values
            for key, values in grouped_values.items()
        }
        artifact_ids = _as_string_list(properties.pop("schemaie__helperArtifactId", []))
        payloads.append(
            {
                "uri": str(subject),
                "properties": properties,
                "artifact_ids": artifact_ids,
                "record_ids": _as_string_list(properties.get("schemaie__propertyRecordId", [])),
            }
        )
    return payloads


def _schemaie_property_key(predicate: Any) -> str | None:
    predicate_text = str(predicate)
    namespace = str(SCHEMA_IE)
    if not predicate_text.startswith(namespace):
        return None
    return f"schemaie__{predicate_text[len(namespace):]}"


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        values = value
    else:
        values = [value]
    return [str(item).strip() for item in values if str(item).strip()]


def _ensure_n10s_ready(session) -> None:
    """Create the required n10s uniqueness constraint and graph config."""
    session.run("CREATE CONSTRAINT n10s_unique_uri IF NOT EXISTS FOR (r:Resource) REQUIRE r.uri IS UNIQUE").consume()

    config = session.run("CALL n10s.graphconfig.show() YIELD param RETURN count(param) AS count").single()
    if config and int(config["count"]) > 0:
        return

    session.run("CALL n10s.graphconfig.init({handleVocabUris: 'SHORTEN'})").consume()


def _remove_resource_labels(session) -> int:
    result = session.run(
        """
        MATCH (resource:Resource)
        WITH collect(resource) AS resources
        FOREACH (resource IN resources | REMOVE resource:Resource)
        RETURN size(resources) AS labelsRemoved
        """
    ).single()
    return int(result["labelsRemoved"]) if result else 0


def _import_with_n10s(session, turtle_data: str, turtle_file_uri: str | None) -> tuple[int, str]:
    """Run either n10s fetch or inline import, both yielding the standard n10s graph shape."""
    if turtle_file_uri:
        result = session.run(
            """
            CALL n10s.rdf.import.fetch($file_uri, 'Turtle', {commitSize: 5000})
            YIELD triplesLoaded
            RETURN triplesLoaded
            """,
            file_uri=turtle_file_uri,
        ).single()
        return (int(result["triplesLoaded"]) if result else 0, "n10s.fetch")

    result = session.run(
        """
        CALL n10s.rdf.import.inline($turtle, 'Turtle', {commitSize: 5000})
        YIELD triplesLoaded
        RETURN triplesLoaded
        """,
        turtle=turtle_data,
    ).single()
    return (int(result["triplesLoaded"]) if result else 0, "n10s.inline")


def _prepare_fetch_file_uri(
    export: TurtleExport,
    neo4j_import_dir: str | Path | None,
    neo4j_import_uri_root: str,
) -> str | None:
    """Write Turtle to a Neo4j-readable import directory and return its server file URI."""
    if not neo4j_import_dir or not export.turtle:
        return None

    import_dir = Path(neo4j_import_dir).expanduser()
    if not import_dir.exists() or not import_dir.is_dir():
        return None

    import_dir.mkdir(parents=True, exist_ok=True)
    ttl_path = import_dir / export.ttl_name
    ttl_path.write_text(export.turtle, encoding="utf-8")
    relative_name = quote(ttl_path.name)
    return f"{neo4j_import_uri_root.rstrip('/')}/{relative_name}"


def _tag_imported_resources(session, turtle_data: str, graph_name: str, source_name: str) -> int:
    """Attach source metadata to resources from the imported Turtle document."""
    resource_uris = sorted(_root_resource_uris_from_turtle(turtle_data))
    tagged = 0
    for start in range(0, len(resource_uris), 5000):
        chunk = resource_uris[start : start + 5000]
        summary = session.run(
            """
            UNWIND $uris AS uri
            MATCH (resource {uri: uri})
            SET resource.iclSourceNames =
                CASE
                    WHEN resource.iclSourceNames IS NULL THEN [$source_name]
                    WHEN NOT $source_name IN resource.iclSourceNames THEN resource.iclSourceNames + $source_name
                    ELSE resource.iclSourceNames
                END,
                resource.iclGraphNames =
                CASE
                    WHEN resource.iclGraphNames IS NULL THEN [$graph_name]
                    WHEN NOT $graph_name IN resource.iclGraphNames THEN resource.iclGraphNames + $graph_name
                    ELSE resource.iclGraphNames
                END
            RETURN count(resource) AS tagged
            """,
            uris=chunk,
            source_name=source_name,
            graph_name=graph_name,
        ).single()
        tagged += int(summary["tagged"]) if summary else 0

    propagated = session.run(
        """
        MATCH (root)
        WHERE root.uri IN $uris
        MATCH (root)-[*1..16]->(resource)
        SET resource.iclSourceNames =
            CASE
                WHEN resource.iclSourceNames IS NULL THEN [$source_name]
                WHEN NOT $source_name IN resource.iclSourceNames THEN resource.iclSourceNames + $source_name
                ELSE resource.iclSourceNames
            END,
            resource.iclGraphNames =
            CASE
                WHEN resource.iclGraphNames IS NULL THEN [$graph_name]
                WHEN NOT $graph_name IN resource.iclGraphNames THEN resource.iclGraphNames + $graph_name
                ELSE resource.iclGraphNames
            END
        RETURN count(DISTINCT resource) AS tagged
        """,
        uris=resource_uris,
        source_name=source_name,
        graph_name=graph_name,
    ).single()

    return tagged + (int(propagated["tagged"]) if propagated else 0)


def _root_resource_uris_from_turtle(turtle_data: str) -> set[str]:
    graph = Graph()
    graph.parse(data=turtle_data, format="turtle")
    return {str(subject) for subject in graph.subjects() if isinstance(subject, URIRef)}
