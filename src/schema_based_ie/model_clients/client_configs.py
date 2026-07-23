"""Shared configuration models for generation and embedding clients."""

from __future__ import annotations

from abc import ABC
from typing import Literal

from pydantic import BaseModel, Field
from typing import Annotated


DEFAULT_LOCAL_LLAMA_MODEL = "ggml-org/gpt-oss-20b-GGUF:MXFP4"


class BaseClientCfg(BaseModel, ABC):
    client_name: str
    model: str


class OllamaClientCfg(BaseClientCfg):
    client_name: Literal["ollama"] = "ollama"
    model: Literal[
        "gpt-oss:20b",
        "gpt-oss:120b",
        "qwen3:0.6b",
        "qwen3:1.7b",
        "qwen3:4b",
        "qwen3:8b",
        "qwen3:30b",
        "qwen3:32b",
        "deepseek-r1:1.5b",
        "deepseek-r1:7b",
        "deepseek-r1:8b",
        "deepseek-r1:14b",
        "deepseek-r1:32b",
        "deepseek-r1:70b",
    ] = Field(default="gpt-oss:20b", json_schema_extra={"group": "perturbation"})


class LlamaClientCfg(BaseClientCfg):
    client_name: Literal["llama"] = "llama"
    model: Literal[
    'Qwen/Qwen3-0.6B-GGUF:Q8_0',
    'Qwen/Qwen3-32B-GGUF:Q4_K_M',
    'Qwen/Qwen3-32B-GGUF:Q5_0',
    'Qwen/Qwen3-32B-GGUF:Q5_K_M',
    'Qwen/Qwen3-32B-GGUF:Q6_K',
    'Qwen/Qwen3-32B-GGUF:Q8_0',
    "unsloth/Qwen3.6-35B-A3B-GGUF:Q8_0",
    'Qwen/Qwen3-Embedding-0.6B-GGUF:F16',
    'Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0',
    'Qwen/Qwen3-Embedding-4B-GGUF:F16',
    'Qwen/Qwen3-Embedding-4B-GGUF:Q4_K_M',
    'Qwen/Qwen3-Embedding-4B-GGUF:Q5_0',
    'Qwen/Qwen3-Embedding-4B-GGUF:Q5_K_M',
    'Qwen/Qwen3-Embedding-4B-GGUF:Q6_K',
    'Qwen/Qwen3-Embedding-4B-GGUF:Q8_0',
    'Qwen/Qwen3-Embedding-8B-GGUF:F16',
    'Qwen/Qwen3-Embedding-8B-GGUF:Q4_K_M',
    'Qwen/Qwen3-Embedding-8B-GGUF:Q5_0',
    'Qwen/Qwen3-Embedding-8B-GGUF:Q5_K_M',
    'Qwen/Qwen3-Embedding-8B-GGUF:Q6_K',
    'Qwen/Qwen3-Embedding-8B-GGUF:Q8_0',
    'ggml-org/gemma-4-26B-A4B-it-GGUF:Q4_K_M',
    'ggml-org/gemma-4-26B-A4B-it-GGUF:Q8_0',
    'ggml-org/gemma-4-31B-it-GGUF:Q4_K_M',
    'ggml-org/gemma-4-31B-it-GGUF:Q8_0',
    'ggml-org/gemma-4-E2B-it-GGUF:BF16',
    'ggml-org/gemma-4-E2B-it-GGUF:Q8_0',
    'ggml-org/gemma-4-E4B-it-GGUF:BF16',
    'ggml-org/gemma-4-E4B-it-GGUF:Q4_K_M',
    'ggml-org/gemma-4-E4B-it-GGUF:Q8_0',
    'ggml-org/gpt-oss-120b-GGUF:MXFP4',
    'ggml-org/gpt-oss-20b-GGUF:MXFP4',
    'lmstudio-community/DeepSeek-R1-Distill-Llama-8B-GGUF:Q3_K_L',
    'lmstudio-community/DeepSeek-R1-Distill-Qwen-1.5B-GGUF:Q3_K_L',
    'lmstudio-community/DeepSeek-R1-Distill-Qwen-1.5B-GGUF:Q4_K_M',
    'lmstudio-community/DeepSeek-R1-Distill-Qwen-14B-GGUF:Q4_K_M',
    ] = (
        DEFAULT_LOCAL_LLAMA_MODEL
    )


class OpenAIClientCfg(BaseClientCfg):
    client_name: Literal["openai"] = "openai"
    model: Literal["gpt-4o-mini", "gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.6-terra"] = "gpt-4o-mini"


class GoogleClientCfg(BaseClientCfg):
    client_name: Literal["google"] = "google"
    model: Literal["gemini-3.5-flash-lite", "gemini-3.5-flash"] = "gemini-3.5-flash-lite"


class AnthropicClientCfg(BaseClientCfg):
    client_name: Literal["anthropic"] = "anthropic"
    model: Literal['claude-opus-4-8', 'claude-sonnet-4-6', 'claude-haiku-4-5','claude-sonnet-5' ] = "claude-sonnet-4-6"

class ExtractionHelperClientCfg(BaseClientCfg):
    client_name: str = "llama"
    model: str = DEFAULT_LOCAL_LLAMA_MODEL

ClientCfg = OllamaClientCfg | LlamaClientCfg | OpenAIClientCfg | GoogleClientCfg | AnthropicClientCfg | ExtractionHelperClientCfg






class OpenAIEmbeddingClientCfg(BaseClientCfg):
    client_name: Literal["openai"] = "openai"
    model: Literal["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"] = Field(
        default="text-embedding-3-small"
    )

class LlamaEmbeddingClientCfg(BaseClientCfg):
    client_name: Literal["llama"] = "llama"
    model: Literal["Qwen/Qwen3-0.6B-GGUF:Q8_0", "ggml-org/gpt-oss-20b-GGUF:MXFP4"] = (
        DEFAULT_LOCAL_LLAMA_MODEL
    )

class GoogleEmbeddingClientCfg(BaseClientCfg):
    client_name: Literal["google"] = "google"
    model: Literal["gemini-embedding-001"] = "gemini-embedding-001"


class DatasheetEmbeddingClientCfg(BaseClientCfg):
    client_name: str = "llama"
    model: str = DEFAULT_LOCAL_LLAMA_MODEL


EmbeddingClientCfg = OpenAIEmbeddingClientCfg | GoogleEmbeddingClientCfg | LlamaEmbeddingClientCfg | DatasheetEmbeddingClientCfg

class EmbeddingCfg(BaseModel):
    client_cfg: EmbeddingClientCfg = Field(
        default_factory=LlamaEmbeddingClientCfg,
        description="Configuration for the model client to use for embedding generation.",
    )
    chunk_size: int = Field(default=512, description="The maximum number of characters in each chunk.")
    chunk_overlap: int = Field(default=100, description="The number of overlapping characters between chunks.")
    max_pdf_chars: int = Field(default=200000, ge=1000)


DatasheetEmbeddingCfg = EmbeddingCfg
