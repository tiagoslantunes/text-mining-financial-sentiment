import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DataQualityAnalysesTests(unittest.TestCase):
    def test_script_reproduces_cached_data_quality_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/data_quality_analyses.py",
                    "--output-dir",
                    tmp,
                    "--quiet",
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                timeout=120,
            )

            self.assertEqual(
                result.returncode,
                0,
                msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
            )

            for name in [
                "duplicates_cashtag_analysis.json",
                "shift_and_noise_analysis.json",
            ]:
                expected = json.loads(
                    (PROJECT_ROOT / "results" / "tables" / name).read_text(encoding="utf-8")
                )
                actual = json.loads((Path(tmp) / name).read_text(encoding="utf-8"))
                self.assertEqual(expected, actual, msg=name)


if __name__ == "__main__":
    unittest.main()
