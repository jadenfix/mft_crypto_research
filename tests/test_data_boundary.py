"""Execute actual notebook generator with explicit synthetic-data provenance."""
import ast
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parents[1]


def generator():
    notebook = json.loads((ROOT / "gpu_optimized_ml_trading_script.ipynb").read_text())
    source = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    tree = ast.parse("\n".join(line for line in source.splitlines() if not line.startswith(("!", "%"))))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name == "create_dummy_csv_if_not_exists"]
    scope = {"os": os, "np": np, "pd": pd, "PREFIX": {"ada": "ada"},
             "TARGET_COL": {"ada": "target_next_close"}}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ROOT), "exec"), scope)
    return scope["create_dummy_csv_if_not_exists"]


class DataBoundaryTests(unittest.TestCase):
    def test_missing_real_data_fails_closed(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {"MFT_ALLOW_SYNTHETIC": "0"}):
            with self.assertRaisesRegex(FileNotFoundError, "Required market data"):
                generator()("ada", d + "/missing.csv", True, d)

    def test_existing_real_path_is_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "real.csv"
            path.write_text("supplied data")
            self.assertEqual(generator()("ada", str(path), True, d), str(path))
            self.assertFalse(Path(str(path) + ".synthetic.json").exists())

    def test_explicit_fixture_generator_csv_model_flow(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {"MFT_ALLOW_SYNTHETIC": "1"}):
            path = generator()("ada", d + "/missing.csv", True, d)
            frame = pd.read_csv(path)
            receipt = json.loads(Path(path + ".synthetic.json").read_text())
            self.assertEqual(receipt["source"], "synthetic")
            self.assertEqual(len(frame), 500)
            features = ["ada_open", "ada_high", "ada_low", "ada_close", "ada_volume"]
            model = RandomForestRegressor(n_estimators=3, max_depth=3, random_state=7, n_jobs=1)
            model.fit(frame[features].iloc[:400], frame["target_next_close"].iloc[:400])
            predictions = model.predict(frame[features].iloc[400:])
            self.assertEqual(predictions.shape, (100,))
            self.assertTrue(np.isfinite(predictions).all())
