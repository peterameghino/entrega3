"""CLI de punta a punta (main.main) contra el servidor HTTP falso local: sin red externa ni API keys reales.

Los SDKs leen OPENAI_BASE_URL / ANTHROPIC_BASE_URL, así que la CLI real se ejecuta sin tocar el código.
"""

import sys

import pytest

import main as cli

from .conftest import FAKE_KEY
from .test_local_http import WORDS, FakeServer, sse


@pytest.fixture
async def server():
    srv = FakeServer()
    await srv.start()
    yield srv
    await srv.stop()


@pytest.fixture
def cli_env(monkeypatch, server):
    monkeypatch.setenv("OPENAI_BASE_URL", server.url + "/v1")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", server.url)
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_MAX_RETRIES", "0")


def run_cli(monkeypatch, *args: str):
    monkeypatch.setattr(sys, "argv", ["main.py", *args])
    return cli.main()


@pytest.mark.parametrize(
    ("env_provider", "flag_provider", "expected_path"),
    [
        ("anthropic", "openai", "/v1/chat/completions"),
        ("openai", "anthropic", "/v1/messages"),
    ],
)
async def test_provider_flag_overrides_llm_provider(
    monkeypatch, cli_env, server, capsys, env_provider, flag_provider, expected_path
):
    monkeypatch.setenv("LLM_PROVIDER", env_provider)
    code = await run_cli(monkeypatch, "--provider", flag_provider)
    out = capsys.readouterr().out
    assert code == 0
    assert server.requests == [expected_path]
    assert f"[{flag_provider} |" in out
    assert "respuesta normal" in out


async def test_env_provider_is_used_without_flag(monkeypatch, cli_env, server, capsys):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    assert await run_cli(monkeypatch) == 0
    assert server.requests == ["/v1/messages"]
    assert "[anthropic |" in capsys.readouterr().out


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
async def test_stream_mode_prints_chunks_in_order(monkeypatch, cli_env, capsys, provider):
    assert await run_cli(monkeypatch, "--provider", provider, "--mode", "stream") == 0
    assert capsys.readouterr().out == "".join(WORDS) + "\n"


async def test_cli_exit_codes_for_config_and_provider_errors(monkeypatch, cli_env, server, capsys):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_BASE_URL", server.url + "/auth/v1")  # el servidor responde 401
    assert await run_cli(monkeypatch) == 1
    assert "LLMAuthenticationError" in capsys.readouterr().err

    monkeypatch.delenv("OPENAI_API_KEY")
    assert await run_cli(monkeypatch) == 2
    err = capsys.readouterr().err
    assert "OPENAI_API_KEY" in err
    assert FAKE_KEY not in err


async def test_stream_error_after_first_chunk_surfaces_as_controlled_error(monkeypatch, cli_env, server, capsys):
    """Una conexión cortada a mitad de SSE no debe reintentarse ni perder los fragmentos ya emitidos."""
    server.truncated_sse = sse(
        [
            (
                None,
                {
                    "id": "c",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "m",
                    "choices": [{"index": 0, "delta": {"content": "parcial"}, "finish_reason": None}],
                },
            )
        ]
    )
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_BASE_URL", server.url + "/trunc/v1")
    code = await run_cli(monkeypatch, "--mode", "stream")
    captured = capsys.readouterr()
    assert code == 1
    assert captured.out.startswith("parcial")
    assert "Error del proveedor" in captured.err
    assert server.requests.count("/trunc/v1/chat/completions") == 1
    assert FAKE_KEY not in captured.err
