import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from analysis import CROSS_3, X_3, analyze_case, analyze_file, load_filters, performance_rows
import main
from matrix import Matrix
from npu import benchmark, decide, mac, normalize_label


class NpuTests(unittest.TestCase):
    def setUp(self):
        self.output = io.StringIO()
        capture = contextlib.redirect_stdout(self.output)
        capture.__enter__()
        self.addCleanup(capture.__exit__, None, None, None)
        self.data = json.loads((Path(__file__).resolve().parents[1] / "data.json").read_text(encoding="utf-8"))

    def test_cross_x_scores(self):
        cross, x = Matrix(CROSS_3), Matrix(X_3)
        self.assertEqual(mac(cross, cross), 5)
        self.assertEqual(mac(cross, x), 1)
        self.assertEqual(mac(x, x), 5)

    def test_fractional_negative_and_zero(self):
        self.assertAlmostEqual(mac(Matrix([[0.5, -2], [0, 1]]), Matrix([[2, 3], [4, -1]])), -6)
        self.assertEqual(mac(Matrix([[0]]), Matrix([[99]])), 0)

    def test_matrix_shape_and_types(self):
        for values in ([], [[1, 2]], [[1, 2], [3]], [[True]], [["1"]], [[float("inf")]], [[float("nan")]]):
            with self.assertRaises(ValueError):
                Matrix(values)
        with self.assertRaises(ValueError):
            Matrix([[1]], 3)

    def test_indexed_read_and_write(self):
        matrix = Matrix([[1, 2], [3, 4]])
        matrix.set(0, 1, 7)
        self.assertEqual(matrix.get(0, 1), 7)
        with self.assertRaises(ValueError):
            matrix.set(0, 0, float("nan"))

    def test_mac_size_and_overflow(self):
        with self.assertRaises(ValueError):
            mac(Matrix([[1]]), Matrix(CROSS_3))
        with self.assertRaises(ValueError):
            mac(Matrix([[1e308]]), Matrix([[1e308]]))

    def test_normalization(self):
        for label in ("+", "cross", "Cross", " CROSS "):
            self.assertEqual(normalize_label(label), "Cross")
        for label in ("x", "X", " X "):
            self.assertEqual(normalize_label(label), "X")
        for label in ("unknown", None, 1):
            with self.assertRaises(ValueError):
                normalize_label(label)

    def test_epsilon_tie_and_real_difference(self):
        self.assertEqual(decide(0.9, 0.8999999999999999), "UNDECIDED")
        self.assertEqual(decide(0.0, 0.0), "UNDECIDED")
        self.assertEqual(decide(1.0, 2.0), "X")
        self.assertEqual(decide(1.0, 1.0 - 2e-9), "Cross")
        self.assertEqual(decide(1, 5, "A", "B"), "B")

    def test_official_data_results_and_counts(self):
        filters, errors = load_filters(self.data)
        results = [analyze_case(key, item, filters, errors) for key, item in self.data["patterns"].items()]
        self.assertEqual(len(results), 6)
        self.assertEqual(sum(result["passed"] for result in results), 3)
        self.assertEqual({result["id"] for result in results if not result["passed"]},
                         {"size_5_1", "size_13_2", "size_25_1"})
        self.assertTrue(all(result["prediction"] == "UNDECIDED" for result in results if not result["passed"]))
        self.assertEqual({item["size"] for item in results}, {5, 13, 25})

    def test_bad_case_does_not_hide_other_cases(self):
        filters, errors = load_filters(self.data)
        item = copy.deepcopy(self.data["patterns"]["size_5_1"])
        item["input"] = [[1]]
        self.assertFalse(analyze_case("size_5_1", item, filters, errors)["passed"])
        self.assertTrue(analyze_case("size_5_2", self.data["patterns"]["size_5_2"], filters, errors)["passed"])
        self.assertFalse(analyze_case("bad-key", item, filters, errors)["passed"])

    def test_missing_filter_and_unknown_label(self):
        data = copy.deepcopy(self.data)
        del data["filters"]["size_5"]
        filters, errors = load_filters(data)
        self.assertIn(5, errors)
        self.assertFalse(analyze_case("size_5_1", data["patterns"]["size_5_1"], filters, errors)["passed"])
        item = copy.deepcopy(data["patterns"]["size_13_1"])
        item["expected"] = "unknown"
        self.assertFalse(analyze_case("size_13_1", item, filters, errors)["passed"])

    def test_tie_is_fail_against_x(self):
        filters = {3: {"Cross": Matrix(CROSS_3), "X": Matrix(X_3)}}
        result = analyze_case("size_3_1", {"input": [[0] * 3 for _ in range(3)], "expected": "x"}, filters, {})
        self.assertEqual(result["prediction"], "UNDECIDED")
        self.assertFalse(result["passed"])

    def test_corrupt_or_missing_json(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.json"
            self.assertIsNone(analyze_file(path, 10))
            path.write_text("not JSON", encoding="utf-8")
            self.assertIsNone(analyze_file(path, 10))

    def test_schema_failures_counted_and_continue(self):
        data = copy.deepcopy(self.data)
        data["patterns"]["size_5_2"]["input"] = [[1]]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad-case.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            result = analyze_file(path, 10)
        self.assertEqual((result["total"], result["passed"], len(result["failed"])), (6, 2, 4))

    def test_benchmark_repeats_and_operation_counts(self):
        matrix = Matrix(CROSS_3)
        self.assertGreaterEqual(benchmark(matrix, matrix, 10), 0)
        with self.assertRaises(ValueError):
            benchmark(matrix, matrix, 9)
        filters, _ = load_filters(self.data)
        rows = performance_rows(filters, 10)
        self.assertEqual([row["operations"] for row in rows], [9, 25, 169, 625])

    def test_interactive_retries_bad_rows(self):
        with patch("builtins.input", side_effect=["", "a b c", "1 2", "nan 0 0", "0 1 0", "1 1 1", "0 1 0"]):
            matrix = main.read_matrix("시험")
        self.assertEqual(matrix.values, [[float(item) for item in row] for row in CROSS_3])
        self.assertGreaterEqual(self.output.getvalue().count("입력 형식 오류"), 4)

    def test_interactive_mode_result(self):
        rows = [" ".join(map(str, row)) for matrix in (CROSS_3, X_3, X_3) for row in matrix]
        with patch("builtins.input", side_effect=rows):
            main.interactive(10)
        self.assertIn("판정: B", self.output.getvalue())
        self.assertIn("A 점수: 1", self.output.getvalue())


if __name__ == "__main__":
    unittest.main()
