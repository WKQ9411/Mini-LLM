from __future__ import annotations

import json
import hashlib
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


class ExperimentStore:
    def __init__(self, path: Path):
        self.path = path
        self._lock = Lock()
        self._loss_history_dir = self.path.parent / "experiment_loss_history"

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            experiments = self._read_all()
            return [self._hydrate_experiment(item) for item in experiments]

    def save(self, experiment: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            serialized = self._serialize_experiment(experiment)
            experiments = self._read_all()
            filtered = [item for item in experiments if item.get("id") != serialized.get("id")]
            filtered.append(serialized)
            filtered.sort(key=lambda item: item.get("timestamp", 0), reverse=True)
            self._write_all(filtered)
            return self._hydrate_experiment(serialized)

    def update(self, experiment_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
        with self._lock:
            experiments = self._read_all()
            for item in experiments:
                if item.get("id") == experiment_id:
                    item.update(updates)
                    self._write_all(experiments)
                    return self._hydrate_experiment(item)
            return None

    def delete(self, experiment_id: str) -> bool:
        with self._lock:
            experiments = self._read_all()
            removed_item = next((item for item in experiments if item.get("id") == experiment_id), None)
            filtered = [item for item in experiments if item.get("id") != experiment_id]
            if len(filtered) == len(experiments):
                return False
            self._write_all(filtered)
            if isinstance(removed_item, dict):
                self._delete_loss_history_file(removed_item)
            return True

    def _serialize_experiment(self, experiment: dict[str, Any]) -> dict[str, Any]:
        serialized = dict(experiment)
        loss_history = serialized.pop("loss_history", [])
        if not isinstance(loss_history, list):
            raise ValueError("loss_history must be a list")

        experiment_id = str(serialized.get("id", ""))
        if not experiment_id:
            raise ValueError("Experiment id is required to store loss history")

        filename = self._loss_history_filename(experiment_id)
        self._write_loss_history(filename, loss_history)
        serialized["loss_history_file"] = filename
        serialized["loss_history_points"] = len(loss_history)
        return serialized

    def _hydrate_experiment(self, item: dict[str, Any]) -> dict[str, Any]:
        hydrated = dict(item)
        filename = hydrated.get("loss_history_file")
        if not isinstance(filename, str):
            raise ValueError("Invalid experiment record: missing loss_history_file")

        hydrated["loss_history"] = self._read_loss_history(filename)
        return hydrated

    def _loss_history_filename(self, experiment_id: str) -> str:
        digest = hashlib.sha1(experiment_id.encode("utf-8")).hexdigest()[:16]
        return f"{digest}.parquet"

    def _write_loss_history(self, filename: str, loss_history: list[Any]) -> None:
        self._loss_history_dir.mkdir(parents=True, exist_ok=True)
        target = self._loss_history_dir / filename
        table = pa.Table.from_pylist(loss_history)
        with NamedTemporaryFile("wb", delete=False, dir=str(self._loss_history_dir), suffix=".tmp") as handle:
            pq.write_table(table, handle.name, compression="zstd")
            temp_path = Path(handle.name)
        temp_path.replace(target)

    def _read_loss_history(self, filename: str) -> list[dict[str, Any]]:
        target = self._loss_history_dir / filename
        if not target.exists():
            raise FileNotFoundError(f"Loss history file not found: {target}")
        table = pq.read_table(target)
        return table.to_pylist()

    def _delete_loss_history_file(self, item: dict[str, Any]) -> None:
        filename = item.get("loss_history_file")
        if not isinstance(filename, str):
            return
        target = self._loss_history_dir / filename
        if target.exists():
            target.unlink()

    def _read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        data = json.loads(self.path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []

    def _write_all(self, experiments: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(self.path.parent), suffix=".tmp") as handle:
            json.dump(experiments, handle, ensure_ascii=True, indent=2)
            temp_path = Path(handle.name)
        temp_path.replace(self.path)
