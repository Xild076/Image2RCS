import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset import AircraftRCSDataset


class TestDatasetResilience(unittest.TestCase):
    def test_max_load_retries_raises_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            missing_image = tmp_path / "missing.png"
            samples = [(missing_image, 1.0)]

            dataset = AircraftRCSDataset(
                samples=samples,
                image_transform=lambda tensor: tensor,
                target_mode="none",
                cache_mode="off",
                max_load_retries=3,
            )

            with self.assertRaises(RuntimeError) as ctx:
                _ = dataset[0]

            message = str(ctx.exception)
            self.assertIn("after 3 attempts", message)
            self.assertIn("initial_index=0", message)


if __name__ == "__main__":
    unittest.main()
