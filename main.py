import argparse
import asyncio
import io
import sys

from errors import ConfigurationError, LLMError
from llm_manager import AsyncLLMManager
from schemas import ChatMessage

DEFAULT_PROMPT = "¿Qué es la entropía? Explicalo brevemente."


async def run_normal(manager: AsyncLLMManager, messages: list[ChatMessage]) -> None:
    response = await manager.generate(messages)
    print(response.content)
    print(f"\n[{response.provider} | {response.model} | finish={response.finish_reason} | {response.usage}]")


async def run_stream(manager: AsyncLLMManager, messages: list[ChatMessage]) -> None:
    async for chunk in manager.stream(messages):
        print(chunk, end="", flush=True)
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified Async LLM Client")
    parser.add_argument("--mode", choices=["normal", "stream"], default="normal")
    parser.add_argument("--provider", choices=["openai", "anthropic"], help="sobrescribe LLM_PROVIDER")
    parser.add_argument("--model", help="sobrescribe LLM_MODEL")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    return parser.parse_args()


async def main() -> int:
    args = parse_args()
    try:
        manager = AsyncLLMManager.from_env(args.provider, args.model)
        messages = [ChatMessage(role="user", content=args.prompt)]
        if args.mode == "stream":
            await run_stream(manager, messages)
        else:
            await run_normal(manager, messages)
    except ConfigurationError as exc:
        print(f"Error de configuración: {exc}", file=sys.stderr)
        return 2
    except LLMError as exc:
        print(f"Error del proveedor ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8")
    sys.exit(asyncio.run(main()))
