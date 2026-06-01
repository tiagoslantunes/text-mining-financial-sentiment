import os
import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ReproducibilityTests(unittest.TestCase):
    def _read_csv_rows(self, relative_path):
        with (PROJECT_ROOT / relative_path).open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def test_preprocessing_imports_with_empty_nltk_data_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = os.environ.copy()
            env["NLTK_DATA"] = tmp
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import sys; "
                        "sys.path.insert(0, '.'); "
                        "from src.preprocessing import full_pipeline; "
                        "print(full_pipeline('Stocks are not falling today'))"
                    ),
                ],
                cwd=PROJECT_ROOT,
                env=env,
                text=True,
                capture_output=True,
                timeout=120,
            )

        self.assertEqual(
            result.returncode,
            0,
            msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )
        self.assertIn("not", result.stdout)

    def test_submission_predictions_are_synchronized_and_valid(self):
        test_rows = self._read_csv_rows("data/raw/test.csv")
        pred_rows = self._read_csv_rows("pred_33.csv")
        pred_final_rows = self._read_csv_rows("results/predictions/pred_final.csv")
        pred_roberta_rows = self._read_csv_rows("results/predictions/pred_robertalarge.csv")

        self.assertEqual(["id", "label"], list(pred_rows[0].keys()))
        self.assertEqual(len(test_rows), len(pred_rows))
        self.assertEqual([row["id"] for row in test_rows], [row["id"] for row in pred_rows])
        self.assertTrue({row["label"] for row in pred_rows}.issubset({"0", "1", "2"}))
        self.assertEqual(pred_rows, pred_roberta_rows)
        self.assertEqual(pred_rows, pred_final_rows)

    def test_lgbm_agent_respects_model_path_and_feature_names(self):
        sys.path.insert(0, str(PROJECT_ROOT))
        import numpy as np
        import src.agent as agent

        calls = {}

        class FakeSbert:
            def encode(self, texts, batch_size, normalize_embeddings, show_progress_bar):
                calls["texts"] = texts
                return np.array([[0.1, 0.2, 0.3]])

        class FakeModel:
            feature_names_in_ = np.array(["Column_0", "Column_1", "Column_2"])

            def predict_proba(self, x):
                calls["model_input_columns"] = list(getattr(x, "columns", []))
                return np.array([[0.05, 0.9, 0.05]])

        old_get_sbert = agent._get_sbert
        old_get_lgbm = agent._get_lgbm
        try:
            agent._get_sbert = lambda: FakeSbert()

            def fake_get_lgbm(model_path="results/models/final_model.pkl"):
                calls["model_path"] = model_path
                return FakeModel()

            agent._get_lgbm = fake_get_lgbm
            output = agent.classify_with_lgbm("great earnings", model_path="custom.pkl")
        finally:
            agent._get_sbert = old_get_sbert
            agent._get_lgbm = old_get_lgbm

        self.assertIn("Bullish (1)", output)
        self.assertEqual("custom.pkl", calls["model_path"])
        self.assertEqual(["Column_0", "Column_1", "Column_2"], calls["model_input_columns"])


if __name__ == "__main__":
    unittest.main()
