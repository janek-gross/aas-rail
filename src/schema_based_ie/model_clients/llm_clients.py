from abc import ABC, abstractmethod
from openai import OpenAI
from google import genai
from anthropic import Anthropic
import voyageai
from sentence_transformers import SentenceTransformer
import os
from pathlib import Path
from pydantic import BaseModel
from typing import Any
import sqlite3
import json
import hashlib
import numpy as np
SQLITE_DB_PATH = "/home/aas-rail/data/embedding_cache.sqlite"

# Conventions:
# Clients and api keys are named after the company, e.g.
# AnthropicClient and ANTHROPIC_API_KEY not ClaudeClient and CLAUDE_API_KEY
# TODO: client input, output conventions
# Generation settings (parameters) currently supported by all clients:
# temperature
# top_p
# max_tokens

class GenerationClient(ABC):
    @abstractmethod
    def generate(self, messages: list, parameters: dict, response_schema: BaseModel | None = None) -> dict:
        ...
    

class EmbeddingClient(ABC):
    @property
    @abstractmethod
    def provider(self) -> str:
        """Stable provider name, e.g. 'openai', 'cohere', 'sentence-transformers'"""
        ...

    @abstractmethod
    def embed(self, texts: list[str], parameters: dict) -> dict:
        ...

class SqliteEmbeddingCache:
    def __init__(self, db_path: str = SQLITE_DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _connect(self):
        # isolation_level=None enables autocommit
        return sqlite3.connect(self.db_path, isolation_level=None)

    def _init_db(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    key TEXT PRIMARY KEY,
                    embedding BLOB NOT NULL
                )
                """
            )

    def get_many(self, keys: list[str]) -> dict[str, list[float]]:
        if not keys:
            return {}

        placeholders = ",".join("?" for _ in keys)
        query = f"""
            SELECT key, embedding
            FROM embeddings
            WHERE key IN ({placeholders})
        """

        result: dict[str, list[float]] = {}

        with self._connect() as conn:
            cursor = conn.execute(query, keys)
            for key, blob in cursor.fetchall():
                emb = np.frombuffer(blob, dtype=np.float32)
                result[key] = emb.tolist()

        return result

    def set_many(self, items: dict[str, list[float]]) -> None:
        if not items:
            return

        rows = [
            (key, np.asarray(embedding, dtype=np.float32).tobytes())
            for key, embedding in items.items()
        ]

        with self._connect() as conn:
            conn.executemany(
                """
                INSERT OR IGNORE INTO embeddings (key, embedding)
                VALUES (?, ?)
                """,
                rows,
            )

def embedding_cache_key(
    *,
    provider: str,
    model: str,
    dimensions: int | None,
    text: str,
    parameters: dict,
    ) -> str:

    payload = {
        "provider": provider,
        "model": model,
        "dimensions": dimensions,
        "text": text,
        "parameters": parameters,
    }
    raw = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()

class CachedEmbeddingClient(EmbeddingClient):
    def __init__(
        self,
        client: EmbeddingClient,
        cache: SqliteEmbeddingCache = SqliteEmbeddingCache(),
    ):
        self.client = client
        self.cache = cache

    @property
    def provider(self) -> str:
        return self.client.provider


    def embed(self, texts: list[str], parameters: dict) -> dict:
        model = parameters["model"]
        dimensions = parameters.get("dimensions")
        keys = {
            text: embedding_cache_key(
                provider=self.provider,
                model=model,
                dimensions=dimensions,
                text=text,
                parameters=parameters,
            )
            for text in texts
        }

        cached = self.cache.get_many(list(keys.values()))

        missing_texts = [
            text for text, key in keys.items() if key not in cached
        ]

        new_embeddings = {}
        usage_info = {
            "input_characters": _embedding_input_characters(texts),
            "model": model,
            "endpoint_type": "embed",
        }
        if missing_texts:
            result = self.client.embed(missing_texts, parameters)
            usage_info.update(result.get("usage", {}))
            usage_info["input_characters"] = _embedding_input_characters(texts)
            for text, emb in zip(missing_texts, result["embeddings"]):
                key = keys[text]
                new_embeddings[key] = emb

            self.cache.set_many(new_embeddings)

        merged = {
            text: cached.get(keys[text]) or new_embeddings[keys[text]]
            for text in texts
        }

        return {
            "embeddings": [merged[text] for text in texts],
            "cached": list(cached.keys()),
            "usage": usage_info,
        }




def clean_kwargs(**kwargs):
    """
    Filter None arguments to prevent overriding defaults
    """
    return {k: v for k, v in kwargs.items() if v is not None}


def _content_character_count(content: Any) -> int:
    if content is None:
        return 0
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        total = 0
        for item in content:
            if isinstance(item, str):
                total += len(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    total += len(text)
            else:
                total += len(str(item))
        return total
    return len(str(content))


def _messages_input_characters(messages: list) -> int:
    return sum(_content_character_count(message.get("content")) for message in messages)


def _embedding_input_characters(texts: list[str]) -> list[int]:
    return [len(text) for text in texts]


def _get_nested_value(obj: Any, *path: str, default: Any = None) -> Any:
    value = obj
    for key in path:
        if value is None:
            return default
        if isinstance(value, dict):
            value = value.get(key)
        else:
            value = getattr(value, key, None)
    return default if value is None else value


def _usage_int(obj: Any, *path: str, default: int = 0) -> int:
    value = _get_nested_value(obj, *path, default=default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _first_usage_int(obj: Any, *paths: tuple[str, ...], default: int = 0) -> int:
    for path in paths:
        value = _get_nested_value(obj, *path)
        if value is not None:
            return _usage_int(obj, *path, default=default)
    return default


def _openai_generation_usage(resp: Any) -> dict[str, Any]:
    usage = _get_nested_value(resp, "usage")
    return {
        "output_tokens": _first_usage_int(usage, ("completion_tokens",), ("output_tokens",)),
        "input_tokens": _first_usage_int(usage, ("prompt_tokens",), ("input_tokens",)),
        "cached_tokens": _first_usage_int(
            usage,
            ("prompt_tokens_details", "cached_tokens"),
            ("input_tokens_details", "cached_tokens"),
        ),
        "model": _get_nested_value(resp, "model"),
        "endpoint_type": "generate",
    }


def _anthropic_generation_usage(resp: Any) -> dict[str, Any]:
    usage = _get_nested_value(resp, "usage")
    cache_creation_tokens = _usage_int(usage, "cache_creation_input_tokens")
    cache_read_tokens = _usage_int(usage, "cache_read_input_tokens")
    non_cached_input_tokens = _usage_int(usage, "input_tokens")
    return {
        "output_tokens": _usage_int(usage, "output_tokens"),
        "input_tokens": non_cached_input_tokens + cache_creation_tokens + cache_read_tokens,
        "cached_tokens": cache_read_tokens,
        "cache_creation_input_tokens": cache_creation_tokens,
        "cache_read_input_tokens": cache_read_tokens,
        "model": _get_nested_value(resp, "model"),
        "endpoint_type": "generate",
    }


def _google_generation_usage(resp: Any) -> dict[str, Any]:
    usage = _get_nested_value(resp, "usage_metadata", default=_get_nested_value(resp, "usageMetadata"))
    return {
        "output_tokens": _first_usage_int(
            usage,
            ("candidates_token_count",),
            ("candidatesTokenCount",),
        ),
        "input_tokens": _first_usage_int(
            usage,
            ("prompt_token_count",),
            ("promptTokenCount",),
        ),
        "cached_tokens": _first_usage_int(
            usage,
            ("cached_content_token_count",),
            ("cachedContentTokenCount",),
        ),
        "model": _get_nested_value(resp, "model_version", default=_get_nested_value(resp, "modelVersion")),
        "endpoint_type": "generate",
    }


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_project_env() -> None:
    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[3] / ".env",
    ]
    for path in candidates:
        _load_env_file(path)


load_project_env()


def _openai_compatible_client(
    endpoint_env_var: str,
    api_key_env_var: str,
    provider: str,
):
    endpoint = os.getenv(endpoint_env_var)
    if not endpoint:
        raise ValueError(
            f"{endpoint_env_var} is not set. Refusing to fall back to the OpenAI API "
            f"for provider '{provider}'. Set {endpoint_env_var} in the environment "
            "or project .env."
        )

    return OpenAIClient(
        end_point=endpoint,
        api_key=os.getenv(api_key_env_var) or "not-needed",
        provider=provider,
    )

import time
class OpenAIClient(GenerationClient, EmbeddingClient):
    def __init__(self, end_point=None, api_key: str | None = None, provider: str = "openai"):
        self._provider = provider
        if end_point in ["openai", None, ""]:
            self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        else:
            self.client = OpenAI(base_url=end_point, api_key=api_key or "not-needed")

    @property
    def provider(self):
        return self._provider

    def generate(self, messages: list, parameters: dict, response_schema: BaseModel | None = None) -> dict:
        kwargs = clean_kwargs(
            model=parameters['model'],
            messages=messages,
            temperature=parameters.get('temperature', None),         # randomness, 0 = deterministic
            top_p=parameters.get('top_p', None),                   # nucleus sampling probability
            #top_k=parameters.get('top_k', None),                   # top-K sampling, likely not supported
            max_completion_tokens=parameters.get('max_tokens', None),  # max tokens to generate
            presence_penalty=parameters.get('presence_penalty', None),  # penalize new topic repeats
            frequency_penalty=parameters.get('frequency_penalty', None),  # penalize repeated tokens
            n=parameters.get('candidate_count', None),      # number of completions to generate
            seed=parameters.get('seed', None),                    # deterministic randomness seed
            response_format=response_schema
        )

        enable_thinking = parameters.get('enable_thinking', None)
        if enable_thinking is not None:
            kwargs['extra_body'] = {"chat_template_kwargs": 
                                    {"enable_thinking": enable_thinking}}
        resp = self.client.chat.completions.parse(**kwargs)

        text = resp.choices[0].message.content
        parsed = getattr(resp.choices[0].message, "parsed", None)
        usage_info = _openai_generation_usage(resp)
        usage_info["input_characters"] = _messages_input_characters(messages)
        return {"raw": resp, "parsed": parsed, "text": text, "usage": usage_info}
    
    def embed(self, texts: list[str], parameters: dict) -> dict:
        kwargs = clean_kwargs(
            model=parameters["model"],   # e.g. "text-embedding-3-large"
            dimensions=parameters.get("dimensions", None),
            input=texts,
        )

        resp = self.client.embeddings.create(**kwargs)

        embeddings = [item.embedding for item in resp.data]
        usage_info = {'input_tokens': resp.usage.prompt_tokens, 'model': resp.model,
                      'endpoint_type': 'embed',
                      'input_characters': _embedding_input_characters(texts)}
        return {
            "raw": resp,
            "embeddings": embeddings,
            "usage": usage_info
        }


class AnthropicClient(GenerationClient):
    def __init__(self):
        self.client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

    @property
    def provider(self):
        return 'anthropic'

    def generate(self, messages: list, parameters: dict, response_schema: BaseModel | None = None) -> dict:
        system = next((m['content'] for m in messages if m['role'] == "system"), None)
        content_messages = [
            {"role": m['role'], "content": m['content']}
            for m in messages
            if m['role'] != "system"
        ]
        if not content_messages and system is not None:
            content_messages = [{"role": "user", "content": system}]
            system = None

        temperature = parameters.get('temperature', None)
        if temperature:
            assert temperature <= 1.0 # Anthropic temperature capped at 1.0
        # presence_penalty, frequency_penalty, candidate_count, seed likely not supported

        kwargs = clean_kwargs(
            model=parameters['model'],
            system=system,
            messages=content_messages,
            max_tokens=parameters.get('max_tokens', 16384),          # max tokens for output
            temperature=temperature,                                # randomness
            top_p=parameters.get('top_p', None),                   # nucleus sampling
            top_k=parameters.get('top_k', None),                   # top-K sampling
            # presence_penalty=parameters.get('presence_penalty', None),  # discourage new topic repetition
            # frequency_penalty=parameters.get('frequency_penalty', None), # discourage repeated tokens
            # candidate_count=parameters.get('candidate_count', None),      # number of completions
            # seed=parameters.get('seed', None),                     # random seed for deterministic output
            output_format=response_schema,
        )

        resp = self.client.messages.parse(**kwargs)

        text_block = next(
            (block for block in resp.content if block.type == "text"),
            None,
        )
        if text_block is None:
            raise RuntimeError("Anthropic response contained no text block")
        text = text_block.text
        parsed = text_block.parsed_output if response_schema else None
        usage_info = _anthropic_generation_usage(resp)
        usage_info["input_characters"] = _messages_input_characters(messages)
        return {'raw': resp, 'parsed': parsed, 'text': text, 'usage': usage_info}


class GoogleClient(GenerationClient, EmbeddingClient):
    def __init__(self):
        self.client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))

    @property
    def provider(self):
        return 'google'

    def generate(self, messages: list, parameters: dict, response_schema: BaseModel | None = None) -> dict:
        # Flatten messages → single prompt
        prompt = "\n".join(
            [f"{m['role'].upper()}: {m['content']}"
            for m in messages]
        )

        config = clean_kwargs(
            temperature = parameters.get('temperature', None),           # randomness
            top_p = parameters.get('top_p', None),                       # nucleus sampling
            top_k = parameters.get('top_k', None),                       # top-K sampling
            max_output_tokens = parameters.get('max_tokens', None),      # max tokens to generate
            presence_penalty = parameters.get('presence_penalty', None), # discourage new topics
            frequency_penalty = parameters.get('frequency_penalty', None), # discourage repeated tokens
            candidate_count = parameters.get('candidate_count', None),   # number of completions
            seed = parameters.get('seed', None),                         # deterministic randomness
        )

        if response_schema:
            config.update({
                "response_mime_type": "application/json",
                "response_json_schema": response_schema.model_json_schema(),
            })

        resp = self.client.models.generate_content(
            model=parameters['model'],
            contents=prompt,
            config=config,
        )

        # TEMPORARY BATCH API ALTERNATIVE
        # To enable it, comment out the generate_content call above and uncomment
        # this block. `resp` remains a GenerateContentResponse, so the code below
        # and all downstream callers keep the same contract.
        # batch_config = config
        # if response_schema:
        #     # The Batch API's structured-output path uses responseSchema rather
        #     # than the responseJsonSchema field used by direct generation.
        #     batch_config = {
        #         **config,
        #         "response_schema": response_schema,
        #     }
        #     batch_config.pop("response_json_schema", None)

        # batch_job = self.client.batches.create(
        #     model=parameters['model'],
        #     src=[{
        #         "contents": prompt,
        #         "config": batch_config,
        #     }],
        # )
        # terminal_states = {
        #     "JOB_STATE_SUCCEEDED",
        #     "JOB_STATE_FAILED",
        #     "JOB_STATE_CANCELLED",
        #     "JOB_STATE_EXPIRED",
        #     "JOB_STATE_PARTIALLY_SUCCEEDED",
        # }
        # while True:
        #     batch_job = self.client.batches.get(name=batch_job.name)
        #     state = batch_job.state.name
        #     if state in terminal_states:
        #         break
        #     time.sleep(10)

        # if state not in {"JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED"}:
        #     raise RuntimeError(
        #         f"Google batch generation ended in {state}: {batch_job.error}"
        #     )
        # if not batch_job.dest or not batch_job.dest.inlined_responses:
        #     raise RuntimeError("Google batch generation returned no inline response")
        # inline_result = batch_job.dest.inlined_responses[0]
        # if inline_result.error:
        #     raise RuntimeError(f"Google batch generation failed: {inline_result.error}")
        # resp = inline_result.response
        ##

        text = resp.text
        parsed = None

        if response_schema:
            parsed = response_schema.model_validate_json(text)
        usage_info = _google_generation_usage(resp)
        usage_info["input_characters"] = _messages_input_characters(messages)
        return {'raw': resp, 'parsed': parsed, 'text': text, 'usage': usage_info}

    def embed(self, texts: list[str], parameters: dict) -> dict:
        config = clean_kwargs(
            model=parameters['model'],
            task_type = parameters.get('task_type',None),
            output_title = parameters.get('output_title',None),
            output_dimensionality = parameters.get('dimensions', None),

        )
        tokens = self.client.models.count_tokens(
                    model=parameters['model'],
                    contents=texts,
                )

        resp = self.client.models.embed_content(
            model=parameters["model"],  # e.g. "text-embedding-004"
            contents=texts,
            config = config
        )

        # Google returns a list of embedding objects
        embeddings = [e.values for e in resp.embeddings]
        
        usage_info = {'input_tokens': tokens.total_tokens, 'model': parameters['model'],
                      'endpoint_type': 'embed',
                      'input_characters': _embedding_input_characters(texts)}
        return {
            "raw": resp,
            "embeddings": embeddings,
            "usage": usage_info
        }


class VoyageClient(EmbeddingClient):
    def __init__(self):
        self.embedding_client = voyageai.Client()

    @property
    def provider(self):
        return 'voyage'

    def embed(self, texts: list[str], parameters: dict) -> dict:
        kwargs = clean_kwargs(
            texts = texts,
            model = parameters['model'],
            input_type = parameters.get('input_type', None),
            output_dimensions = parameters.get('dimensions', None),
            # output_dtype = None,
            # truncation = True
        )

        resp = self.embedding_client(**kwargs)
        embeddings = [e for e in resp.embeddings]
        usage_info = {'input_tokens': resp.total_tokens,
                      'model': parameters['model'], 'endpoint_type': 'embed',
                      'input_characters': _embedding_input_characters(texts)}
        return {
            "raw": resp,
            "embeddings": embeddings,
            "usage": usage_info
        }

class SentenceTransformerClient(EmbeddingClient):
    _models = {}

    @property
    def provider(self):
        return 'sentencetransformer'

    @classmethod
    def _get_model(cls, model_name: str):
        if model_name not in cls._models:
            cls._models[model_name] = SentenceTransformer(model_name)
        return cls._models[model_name]

    def embed(self, texts: list[str], parameters: dict) -> dict:
        model = self._get_model(parameters['model'])
        resp = model.encode(texts, normalize_embeddings=True)
        embeddings = [e.tolist() for e in resp]
        tokenizer = model.tokenizer
        tokens = tokenizer("Hello world", return_length=True)
        usage_info = {'input_tokens': tokens['length'],
                      'model': parameters['model'], 'endpoint_type': 'embed',
                      'input_characters': _embedding_input_characters(texts)}
        return {
            "raw": resp,
            "embeddings": embeddings,
            "usage": usage_info
        }






GENERATION_CLIENT_REGISTRY = {
    'openai': OpenAIClient,
    'ollama': lambda: _openai_compatible_client('OLLAMA_ENDPOINT', 'OLLAMA_API_KEY', 'ollama'),
    'llama': lambda: _openai_compatible_client('LLAMA_ENDPOINT', 'LLAMA_API_KEY', 'llama'),
    'google': GoogleClient,
    'anthropic': AnthropicClient
}


EMBEDDING_CLIENT_REGISTRY = {
    'openai': OpenAIClient,
    'ollama': lambda: _openai_compatible_client('OLLAMA_ENDPOINT', 'OLLAMA_API_KEY', 'ollama'),
    'llama': lambda: _openai_compatible_client('LLAMA_ENDPOINT', 'LLAMA_API_KEY', 'llama'),
    'google': GoogleClient,
    'voyage': VoyageClient,
    'sentence_transformer': SentenceTransformerClient
}

def cached(client_factory):
    cache = SqliteEmbeddingCache(SQLITE_DB_PATH)
    def wrapper():
        return CachedEmbeddingClient(client_factory(), cache)
    return wrapper

CACHED_EMBEDDING_CLIENT_REGISTRY = {
    name: cached(factory) for name, factory in EMBEDDING_CLIENT_REGISTRY.items()}
