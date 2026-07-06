from golden_retriever.context1_data_gen import (
    build_status,
    domain_requirements,
    missing_requirements,
    render_dry_run_commands,
)


def test_upstream_domain_requirements_cover_public_context1_domains():
    reqs = domain_requirements(provider="upstream")

    assert {"web", "sec", "patents", "epstein"}.issubset(reqs)
    assert "ANTHROPIC_API_KEY" in reqs["web"]
    assert "CHROMA_API_KEY" in reqs["sec"]


def test_codex_cli_requirements_do_not_require_anthropic_or_openai_api_keys():
    reqs = domain_requirements(provider="codex-cli")

    for domain_reqs in reqs.values():
        assert "ANTHROPIC_API_KEY" not in domain_reqs
        assert "OPENAI_API_KEY" not in domain_reqs
    assert reqs["web"] == []


def test_missing_requirements_reports_absent_environment_for_upstream():
    missing = missing_requirements("web", env={"ANTHROPIC_API_KEY": "x"}, provider="upstream")

    assert "ANTHROPIC_API_KEY" not in missing
    assert "SERPER_API_KEY" in missing
    assert "OPENAI_API_KEY" in missing


def test_build_status_reports_codex_cli_auth_without_anthropic_requirement():
    status = build_status(
        output_root="data/context1-generated/raw",
        provider="codex-cli",
        env={},
        codex_cli_available=True,
        codex_auth_available=True,
    )

    assert status["llm_provider"] == "codex-cli"
    assert status["codex_cli_available"] is True
    assert status["codex_auth_available"] is True
    assert status["domains"]["web"]["missing_env"] == []
    assert "ANTHROPIC_API_KEY" not in str(status)
    assert "OPENAI_API_KEY" not in str(status)


def test_render_dry_run_commands_includes_all_domains():
    commands = render_dry_run_commands(output_root="data/context1-generated/raw")

    assert "agentic_search_data_gen.domains.web" in commands["web"]
    assert "agentic_search_data_gen.domains.sec" in commands["sec"]
    assert "agentic_search_data_gen.domains.patents" in commands["patents"]
    assert "agentic_search_data_gen.domains.epstein" in commands["epstein"]
