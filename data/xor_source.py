"""Text-file data source for the classic two-input XOR example."""

from __future__ import annotations

import csv
from pathlib import Path

from data.base import DataSource
from data.dataset import Dataset


class XORDataSource(DataSource):
    """Read a small, editable XOR dataset from a comma-separated text file."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else Path(__file__).parent / "datasets" / "xor.txt"

    def load(self) -> Dataset:
        if not self.path.is_file():
            raise FileNotFoundError(f"XOR dataset file was not found: {self.path}")

        inputs: list[list[float]] = []
        targets: list[list[float]] = []
        try:
            with self.path.open("r", encoding="utf-8", newline="") as file:
                for line_number, raw_line in enumerate(file, start=1):
                    stripped = raw_line.strip()
                    if not stripped or stripped.startswith("#"):
                        continue
                    try:
                        row = next(csv.reader([raw_line], skipinitialspace=True))
                        if len(row) != 3:
                            raise ValueError("expected exactly 3 comma-separated values")
                        values = [float(value.strip()) for value in row]
                    except (ValueError, csv.Error) as exc:
                        raise ValueError(
                            f"Invalid XOR dataset row at {self.path}:{line_number}: {exc}"
                        ) from exc
                    inputs.append(values[:2])
                    targets.append([values[2]])
        except OSError as exc:
            raise OSError(f"Could not read dataset file {self.path}: {exc}") from exc

        return Dataset(
            inputs=inputs,
            targets=targets,
            input_names=("input_1", "input_2"),
            target_names=("xor",),
        )
