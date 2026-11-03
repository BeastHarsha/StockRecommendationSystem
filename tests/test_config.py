import pytest

from config import ROOT_DIR, load_settings

ENV_VARS = [
    "GROWW_API_KEY",
    "GROWW_API_SECRET",
    "GROWW_ACCESS_TOKEN",
    "OPENAI_API_KEY",
    "LLM_MODEL",
    "USE_MOCK_GROWW",
    "PRICE_DROP_THRESHOLD_PCT",
    "POLL_INTERVAL_MINUTES",
    "MAX_SELF_CHECK_RETRIES",
    "CHROMA_PERSIST_DIR",
    "EMBEDDING_MODEL",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_defaults():
    settings = load_settings(env_file=None)
    assert settings.use_mock_groww is True
    assert settings.price_drop_threshold_pct == 5.0
    assert settings.poll_interval_minutes == 15
    assert settings.max_self_check_retries == 3
    assert settings.chroma_persist_dir == ROOT_DIR / "storage" / "chroma"
    assert settings.llm_model == "gpt-6-luna"


def test_mock_mode_does_not_require_groww_credentials():
    settings = load_settings(env_file=None)
    assert settings.missing_credentials() == ["OPENAI_API_KEY"]


def test_real_mode_requires_groww_credentials(monkeypatch):
    monkeypatch.setenv("USE_MOCK_GROWW", "false")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    settings = load_settings(env_file=None)
    assert settings.missing_credentials() == ["GROWW_API_KEY", "GROWW_API_SECRET"]


def test_direct_access_token_replaces_key_and_secret(monkeypatch):
    monkeypatch.setenv("USE_MOCK_GROWW", "false")
    monkeypatch.setenv("GROWW_ACCESS_TOKEN", "test-token")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    settings = load_settings(env_file=None)
    assert settings.groww_access_token == "test-token"
    assert settings.missing_credentials() == []


def test_env_values_override_defaults(monkeypatch):
    monkeypatch.setenv("PRICE_DROP_THRESHOLD_PCT", "2.5")
    monkeypatch.setenv("POLL_INTERVAL_MINUTES", "5")
    settings = load_settings(env_file=None)
    assert settings.price_drop_threshold_pct == 2.5
    assert settings.poll_interval_minutes == 5


@pytest.mark.parametrize(
    "name, value",
    [
        ("PRICE_DROP_THRESHOLD_PCT", "abc"),
        ("PRICE_DROP_THRESHOLD_PCT", "0"),
        ("POLL_INTERVAL_MINUTES", "0"),
        ("MAX_SELF_CHECK_RETRIES", "-1"),
    ],
)
def test_invalid_values_raise(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        load_settings(env_file=None)
