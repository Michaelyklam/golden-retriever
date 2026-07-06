from __future__ import annotations

import argparse
import json
import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

Provider = Literal["upstream", "codex-cli"]

UPSTREAM_REQUIREMENTS = {
    "web": ["ANTHROPIC_API_KEY", "SERPER_API_KEY", "JINA_API_KEY", "OPENAI_API_KEY", "CHROMA_API_KEY", "CHROMA_DATABASE"],
    "sec": ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "CHROMA_API_KEY", "CHROMA_DATABASE", "BASETEN_API_KEY"],
    "patents": ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "CHROMA_API_KEY", "CHROMA_DATABASE", "USPTO_API_KEY", "SEARCH_API_KEY", "DATALAB_API_KEY"],
    "epstein": ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "CHROMA_API_KEY", "CHROMA_DATABASE"],
}

# Michael does not want Anthropic or OpenAI API-key based generation. The
# Codex-subscription path treats Codex CLI/OAuth as the LLM/teacher and uses our
# local corpus/indexing stack instead of Chroma Cloud/OpenAI embeddings. Domain
# fetch credentials are therefore handled by concrete materializers later, not
# by this LLM-provider readiness check.
CODEX_CLI_REQUIREMENTS = {domain: [] for domain in UPSTREAM_REQUIREMENTS}


def codex_cli_available() -> bool:
    return shutil.which("codex") is not None


def codex_auth_available() -> bool:
    return (Path.home() / ".codex" / "auth.json").exists() or (Path.home() / ".hermes" / "auth.json").exists()


def domain_requirements(provider: Provider = "codex-cli") -> dict[str, list[str]]:
    """Required environment variables for Context-1 data generation.

    `upstream` mirrors Chroma's public repo exactly. `codex-cli` is our project
    path: no Anthropic API key and no OpenAI API key for LLM calls.
    """

    if provider == "upstream":
        return {domain: list(reqs) for domain, reqs in UPSTREAM_REQUIREMENTS.items()}
    if provider == "codex-cli":
        return {domain: list(reqs) for domain, reqs in CODEX_CLI_REQUIREMENTS.items()}
    raise ValueError(f"unsupported provider: {provider}")


def missing_requirements(domain: str, env: Mapping[str, str] | None = None, provider: Provider = "codex-cli") -> list[str]:
    values = env if env is not None else os.environ
    return [name for name in domain_requirements(provider=provider)[domain] if not values.get(name)]


def render_dry_run_commands(output_root: str = "data/context1-generated/raw") -> dict[str, str]:
    return {
        "web": f"uv run python -m agentic_search_data_gen.domains.web --output {output_root}/web --collection context1-web",
        "sec": f"uv run python -m agentic_search_data_gen.domains.sec -o {output_root}/sec -c context1-sec --identity 'Michael Lam <email@example.com>'",
        "patents": f"uv run python -m agentic_search_data_gen.domains.patents --output {output_root}/patents --collection context1-patents",
        "epstein": f"uv run python -m agentic_search_data_gen.domains.epstein -o {output_root}/epstein -c context1-epstein",
    }


def render_codex_local_plan(output_root: str = "data/context1-generated/raw") -> dict[str, str]:
    return {
        "teacher": "Use Codex CLI OAuth/subscription via `codex exec` for teacher trajectory generation; do not use Anthropic or OpenAI API keys.",
        "corpus": "Use public/static corpora plus local lexical/dense indexing; do not require Chroma Cloud for first full fine-tune corpus.",
        "output_root": output_root,
    }


def build_status(
    output_root: str = "data/context1-generated/raw",
    provider: Provider = "codex-cli",
    env: Mapping[str, str] | None = None,
    codex_cli_available: bool | None = None,
    codex_auth_available: bool | None = None,
) -> dict[str, object]:
    commands = render_dry_run_commands(output_root)
    cli_ok = globals()["codex_cli_available"]() if codex_cli_available is None else codex_cli_available
    auth_ok = globals()["codex_auth_available"]() if codex_auth_available is None else codex_auth_available
    status: dict[str, object] = {
        "output_root": output_root,
        "llm_provider": provider,
        "domains": {
            domain: {"missing_env": missing_requirements(domain, env=env, provider=provider), "upstream_command": commands[domain]}
            for domain in domain_requirements(provider=provider)
        },
    }
    if provider == "codex-cli":
        status.update(
            {
                "codex_cli_available": cli_ok,
                "codex_auth_available": auth_ok,
                "missing_tools": ([] if cli_ok else ["codex"]),
                "missing_auth": ([] if auth_ok else ["~/.codex/auth.json or ~/.hermes/auth.json"]),
                "codex_local_plan": render_codex_local_plan(output_root),
            }
        )
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description="Dry-run/status helper for Context-1 corpus generation.")
    parser.add_argument("--output-root", default="data/context1-generated/raw")
    parser.add_argument("--provider", choices=["upstream", "codex-cli"], default="codex-cli")
    args = parser.parse_args()
    print(json.dumps(build_status(args.output_root, provider=args.provider), indent=2))


if __name__ == "__main__":
    main()
