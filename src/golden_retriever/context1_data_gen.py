from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping


def domain_requirements() -> dict[str, list[str]]:
    """Required environment variables for Chroma's public context-1-data-gen domains."""

    return {
        "web": ["ANTHROPIC_API_KEY", "SERPER_API_KEY", "JINA_API_KEY", "OPENAI_API_KEY", "CHROMA_API_KEY", "CHROMA_DATABASE"],
        "sec": ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "CHROMA_API_KEY", "CHROMA_DATABASE", "BASETEN_API_KEY"],
        "patents": ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "CHROMA_API_KEY", "CHROMA_DATABASE", "USPTO_API_KEY", "SEARCH_API_KEY", "DATALAB_API_KEY"],
        "epstein": ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "CHROMA_API_KEY", "CHROMA_DATABASE"],
    }


def missing_requirements(domain: str, env: Mapping[str, str] | None = None) -> list[str]:
    values = env if env is not None else os.environ
    return [name for name in domain_requirements()[domain] if not values.get(name)]


def render_dry_run_commands(output_root: str = "data/context1-generated/raw") -> dict[str, str]:
    return {
        "web": f"uv run python -m agentic_search_data_gen.domains.web --output {output_root}/web --collection context1-web",
        "sec": f"uv run python -m agentic_search_data_gen.domains.sec -o {output_root}/sec -c context1-sec --identity 'Michael Lam <email@example.com>'",
        "patents": f"uv run python -m agentic_search_data_gen.domains.patents --output {output_root}/patents --collection context1-patents",
        "epstein": f"uv run python -m agentic_search_data_gen.domains.epstein -o {output_root}/epstein -c context1-epstein",
    }


def build_status(output_root: str = "data/context1-generated/raw") -> dict[str, object]:
    commands = render_dry_run_commands(output_root)
    return {
        "output_root": output_root,
        "domains": {
            domain: {"missing_env": missing_requirements(domain), "command": commands[domain]}
            for domain in domain_requirements()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Dry-run/status helper for Chroma context-1-data-gen corpus generation.")
    parser.add_argument("--output-root", default="data/context1-generated/raw")
    args = parser.parse_args()
    print(json.dumps(build_status(args.output_root), indent=2))


if __name__ == "__main__":
    main()
