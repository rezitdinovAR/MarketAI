"""Configuration management for MarketMind."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Project root
ROOT_DIR = Path(__file__).resolve().parent.parent

load_dotenv(ROOT_DIR / ".env")


class LLMConfig(BaseModel):
    provider: str = "deepseek"
    model: str = "deepseek-chat"
    base_url: str = "https://api.deepseek.com"
    api_key: str = Field(default="")
    timeout: int = 30
    max_retries: int = 2
    max_calls_per_request: int = 10
    max_tokens_per_request: int = 50000
    max_cost_per_request: float = 0.10


class SourceConfig(BaseModel):
    enabled: bool = True
    use_mock: bool = True
    timeout: int = 10


class SearchConfig(BaseModel):
    sources: dict[str, SourceConfig] = Field(default_factory=lambda: {
        "ozon": SourceConfig(),
        "wildberries": SourceConfig(),
        "yandex": SourceConfig(),
    })
    max_results_per_source: int = 20
    max_results_total: int = 10
    min_rating: float = 3.5


class AnalysisConfig(BaseModel):
    max_reviews_per_product: int = 15
    review_token_limit: int = 3000
    batch_size: int = 3


class AppConfig(BaseModel):
    name: str = "MarketMind"
    version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"


class PathsConfig(BaseModel):
    mock_data: str = "data/mock/"
    logs: str = "logs/"
    prompts: str = "prompts/"


class Settings(BaseModel):
    app: AppConfig = Field(default_factory=AppConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)

    def get_mock_data_path(self) -> Path:
        return ROOT_DIR / self.paths.mock_data

    def get_logs_path(self) -> Path:
        return ROOT_DIR / self.paths.logs

    def get_prompts_path(self) -> Path:
        return ROOT_DIR / self.paths.prompts


def load_settings() -> Settings:
    """Load settings from YAML + env vars."""
    config_path = ROOT_DIR / "config" / "settings.yaml"

    data = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

    # Build nested config from YAML
    settings = Settings(
        app=AppConfig(**data.get("app", {})),
        llm=LLMConfig(**data.get("llm", {})),
        search=SearchConfig(
            sources={
                k: SourceConfig(**v)
                for k, v in data.get("search", {}).get("sources", {}).items()
            },
            max_results_per_source=data.get("search", {}).get("max_results_per_source", 20),
            max_results_total=data.get("search", {}).get("max_results_total", 10),
            min_rating=data.get("search", {}).get("min_rating", 3.5),
        ),
        analysis=AnalysisConfig(**data.get("analysis", {})),
        paths=PathsConfig(**data.get("paths", {})),
    )

    # Env overrides
    settings.llm.api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if os.getenv("MARKETMIND_DEBUG"):
        settings.app.debug = os.getenv("MARKETMIND_DEBUG", "").lower() in ("true", "1", "yes")
    if os.getenv("MARKETMIND_LOG_LEVEL"):
        settings.app.log_level = os.getenv("MARKETMIND_LOG_LEVEL", "INFO")
    if os.getenv("MARKETMIND_LLM_MODEL"):
        settings.llm.model = os.getenv("MARKETMIND_LLM_MODEL", "deepseek-chat")

    return settings
