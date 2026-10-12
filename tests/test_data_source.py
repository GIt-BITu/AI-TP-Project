from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from data import XORDataSource


class XORDataSourceTests(unittest.TestCase):
    def test_loads_xor_file(self) -> None:
        dataset = XORDataSource().load()
        self.assertEqual(dataset.inputs, [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        self.assertEqual(dataset.targets, [[0.0], [1.0], [1.0], [0.0]])
        self.assertEqual(dataset.input_names, ("input_1", "input_2"))

    def test_reports_malformed_row_with_line_number(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad_xor.txt"
            path.write_text("# header\n0,1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"bad_xor.txt:2"):
                XORDataSource(path).load()

    def test_reports_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.txt"
            with self.assertRaises(FileNotFoundError):
                XORDataSource(path).load()


if __name__ == "__main__":
    unittest.main()
