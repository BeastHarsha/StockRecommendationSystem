import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_number(name: str, default: float, cast: type, minimum: float):
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return cast(default)
    try:
        value = cast(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


@dataclass(frozen=True)
class Settings:
    groww_api_key: str
    groww_api_secret: str
    anthropic_api_key: str
    llm_model: str
    use_mock_groww: bool
    price_drop_threshold_pct: float
    poll_interval_minutes: int
    max_self_check_retries: int
    chroma_persist_dir: Path
    embedding_model: str

    def missing_credentials(self) -> list[str]:
        missing = []
        if not self.use_mock_groww:
            if not self.groww_api_key:
                missing.append("GROWW_API_KEY")
            if not self.groww_api_secret:
                missing.append("GROWW_API_SECRET")
        if not self.anthropic_api_key:
            missing.append("ANTHROPIC_API_KEY")
        return missing


def load_settings(env_file: Path | None = ROOT_DIR / ".env") -> Settings:
    if env_file is not None and env_file.exists():
        load_dotenv(env_file, override=False)

    chroma_dir = Path(os.getenv("CHROMA_PERSIST_DIR") or "storage/chroma")
    if not chroma_dir.is_absolute():
        chroma_dir = ROOT_DIR / chroma_dir

    return Settings(
        groww_api_key=os.getenv("GROWW_API_KEY", "").strip(),
        groww_api_secret=os.getenv("GROWW_API_SECRET", "").strip(),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", "").strip(),
        llm_model=os.getenv("LLM_MODEL") or "claude-sonnet-5-5",
        use_mock_groww=_get_bool("USE_MOCK_GROWW", default=True),
        price_drop_threshold_pct=_get_number("PRICE_DROP_THRESHOLD_PCT", 5, float, minimum=0.01),
        poll_interval_minutes=_get_number("POLL_INTERVAL_MINUTES", 15, int, minimum=1),
        max_self_check_retries=_get_number("MAX_SELF_CHECK_RETRIES", 3, int, minimum=0),
        chroma_persist_dir=chroma_dir,
        embedding_model=os.getenv("EMBEDDING_MODEL") or "all-MiniLM-L6-v2",
    )
