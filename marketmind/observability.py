"""Structured logging and request tracing for MarketMind."""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class JSONFormatter(logging.Formatter):
    """Structured JSON log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
        }
        if hasattr(record, "request_id"):
            log_entry["request_id"] = record.request_id
        if hasattr(record, "stage"):
            log_entry["stage"] = record.stage
        if hasattr(record, "data"):
            log_entry["data"] = record.data
        return json.dumps(log_entry, ensure_ascii=False, default=str)


def setup_logger(name: str = "marketmind", level: str = "INFO", log_dir: Optional[Path] = None) -> logging.Logger:
    """Configure application logger with JSON formatting."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if logger.handlers:
        return logger

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(JSONFormatter())
    logger.addHandler(console)

    # File handler
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / "app.log", encoding="utf-8")
        fh.setFormatter(JSONFormatter())
        logger.addHandler(fh)

    return logger


@dataclass
class StageTrace:
    name: str
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    status: str = "running"
    llm_calls: int = 0
    metadata: dict = field(default_factory=dict)

    def finish(self, status: str = "success") -> None:
        self.end_time = time.time()
        self.status = status

    @property
    def duration_seconds(self) -> float:
        end = self.end_time or time.time()
        return round(end - self.start_time, 3)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "duration_seconds": self.duration_seconds,
            "status": self.status,
            "llm_calls": self.llm_calls,
            "metadata": self.metadata,
        }


@dataclass
class RequestTrace:
    request_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    stages: list[StageTrace] = field(default_factory=list)
    total_llm_calls: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    final_status: str = "running"
    error_message: Optional[str] = None

    def start_stage(self, name: str) -> StageTrace:
        stage = StageTrace(name=name)
        self.stages.append(stage)
        return stage

    def finish(self, status: str = "success", error: Optional[str] = None) -> None:
        self.end_time = time.time()
        self.final_status = status
        self.error_message = error

    @property
    def duration_seconds(self) -> float:
        end = self.end_time or time.time()
        return round(end - self.start_time, 3)

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "duration_seconds": self.duration_seconds,
            "stages": [s.to_dict() for s in self.stages],
            "total_llm_calls": self.total_llm_calls,
            "total_tokens": self.total_tokens,
            "total_cost": round(self.total_cost, 6),
            "final_status": self.final_status,
            "error_message": self.error_message,
        }

    def save(self, log_dir: Path) -> None:
        traces_dir = log_dir / "traces"
        traces_dir.mkdir(parents=True, exist_ok=True)
        path = traces_dir / f"{self.request_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
