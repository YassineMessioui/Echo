import importlib.util
import os
from pathlib import Path

LIVE_TEST = Path(__file__).parent / "test_llm_live.py"


def test_live_test_module_reads_dotenv_without_leaking_it(tmp_path, monkeypatch):
    """pytest imports test_llm_live.py while collecting *every* test, so it must read
    .env without putting its values into os.environ (real keys would leak everywhere)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ECHO_DOTENV_LEAK_CHECK", raising=False)
    monkeypatch.setenv("ECHO_SHELL_WINS", "shell")
    (tmp_path / ".env").write_text("ECHO_DOTENV_LEAK_CHECK=dotenv\nECHO_SHELL_WINS=dotenv\n")

    spec = importlib.util.spec_from_file_location("live_under_test", LIVE_TEST)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert "ECHO_DOTENV_LEAK_CHECK" not in os.environ
    assert module.ENV["ECHO_DOTENV_LEAK_CHECK"] == "dotenv"
    assert module.ENV["ECHO_SHELL_WINS"] == "shell"
