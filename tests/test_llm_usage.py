from types import SimpleNamespace

from pydantic import BaseModel, ConfigDict

from aas_rail.model_clients.llm_clients import (
    AnthropicClient,
    GoogleClient,
    _anthropic_generation_usage,
    _google_generation_usage,
    _openai_generation_usage,
)


def test_anthropic_generate_supports_a_system_only_prompt_with_stable_parser():
    class Payload(BaseModel):
        value: str

    captured = {}
    response = SimpleNamespace(
        content=[
            SimpleNamespace(
                type="thinking",
                thinking="Working on the structured response.",
            ),
            SimpleNamespace(
                type="text",
                text='{"value":"ok"}',
                parsed_output=Payload(value="ok"),
            )
        ],
        model="claude-test",
        usage=SimpleNamespace(
            output_tokens=1,
            input_tokens=2,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
    )

    class FakeMessages:
        def parse(self, **kwargs):
            captured.update(kwargs)
            return response

    client = AnthropicClient.__new__(AnthropicClient)
    client.client = SimpleNamespace(messages=FakeMessages())

    result = client.generate(
        messages=[{"role": "system", "content": "Return JSON."}],
        parameters={"model": "claude-test"},
        response_schema=Payload,
    )

    assert "system" not in captured
    assert captured["messages"] == [
        {"role": "user", "content": "Return JSON."}
    ]
    assert captured["output_format"] is Payload
    assert captured["max_tokens"] == 16384
    assert "betas" not in captured
    assert result["text"] == '{"value":"ok"}'
    assert result["parsed"] == Payload(value="ok")


def test_openai_generation_usage_reports_cached_tokens_from_chat_details():
    resp = SimpleNamespace(
        model="gpt-4o-mini",
        usage=SimpleNamespace(
            completion_tokens=17,
            prompt_tokens=128,
            prompt_tokens_details=SimpleNamespace(cached_tokens=64),
        ),
    )

    assert _openai_generation_usage(resp) == {
        "output_tokens": 17,
        "input_tokens": 128,
        "cached_tokens": 64,
        "model": "gpt-4o-mini",
        "endpoint_type": "generate",
    }


def test_openai_generation_usage_tolerates_missing_cache_details():
    resp = {
        "model": "local-openai-compatible",
        "usage": {
            "completion_tokens": 9,
            "prompt_tokens": 45,
        },
    }

    assert _openai_generation_usage(resp) == {
        "output_tokens": 9,
        "input_tokens": 45,
        "cached_tokens": 0,
        "model": "local-openai-compatible",
        "endpoint_type": "generate",
    }


def test_anthropic_generation_usage_reports_cache_reads_and_writes():
    resp = SimpleNamespace(
        model="claude-sonnet-4-5",
        usage=SimpleNamespace(
            output_tokens=23,
            input_tokens=100,
            cache_creation_input_tokens=40,
            cache_read_input_tokens=60,
        ),
    )

    assert _anthropic_generation_usage(resp) == {
        "output_tokens": 23,
        "input_tokens": 200,
        "cached_tokens": 60,
        "cache_creation_input_tokens": 40,
        "cache_read_input_tokens": 60,
        "model": "claude-sonnet-4-5",
        "endpoint_type": "generate",
    }


def test_google_generation_usage_reports_cached_content_tokens():
    resp = SimpleNamespace(
        model_version="gemini-2.5-flash",
        usage_metadata=SimpleNamespace(
            candidates_token_count=31,
            prompt_token_count=256,
            cached_content_token_count=128,
        ),
    )

    assert _google_generation_usage(resp) == {
        "output_tokens": 31,
        "input_tokens": 256,
        "cached_tokens": 128,
        "model": "gemini-2.5-flash",
        "endpoint_type": "generate",
    }


def test_google_generate_uses_json_schema_for_pydantic_structured_output():
    class StrictPayload(BaseModel):
        model_config = ConfigDict(extra="forbid")

        value: str

    captured = {}

    class FakeModels:
        def generate_content(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                text='{"value": "ok"}',
                model_version="gemini-test",
                usage_metadata=SimpleNamespace(
                    candidates_token_count=1,
                    prompt_token_count=2,
                    cached_content_token_count=0,
                ),
            )

    client = GoogleClient.__new__(GoogleClient)
    client.client = SimpleNamespace(models=FakeModels())

    result = client.generate(
        messages=[{"role": "system", "content": "Return JSON."}],
        parameters={"model": "gemini-test"},
        response_schema=StrictPayload,
    )

    assert captured["config"]["response_mime_type"] == "application/json"
    assert "response_json_schema" in captured["config"]
    assert "response_schema" not in captured["config"]
    assert captured["config"]["response_json_schema"]["additionalProperties"] is False
    assert result["parsed"] == StrictPayload(value="ok")
