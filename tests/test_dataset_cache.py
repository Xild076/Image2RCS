import sys
import tempfile
import time
import unittest
from pathlib import Path

import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset import ImageTensorCache, inverse_transform_target, transform_target


class TestDatasetCache(unittest.TestCase):
    def _write_image(self, path: Path, color: tuple[int, int, int]) -> None:
        image = Image.new("RGB", (20, 20), color=color)
        image.save(path)

    def test_target_transform_round_trip(self):
        values = [0.0, 0.01, 1.0, 50.0, 100.0]
        for value in values:
            transformed = transform_target(value, mode="log1p")
            restored = inverse_transform_target(transformed, mode="log1p")
            self.assertAlmostEqual(restored, value, places=6)

    def test_disk_cache_hit_and_mtime_invalidation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            image_path = tmp_path / "sample.png"
            cache_dir = tmp_path / "cache"

            self._write_image(image_path, (255, 0, 0))
            cache = ImageTensorCache(mode="disk", cache_dir=cache_dir)

            first = cache.get(image_path)
            self.assertIsInstance(first, torch.Tensor)
            self.assertEqual(len(list(cache_dir.glob("*.pt"))), 1)

            second = cache.get(image_path)
            self.assertTrue(torch.equal(first, second))
            self.assertGreaterEqual(cache.hits, 1)

            time.sleep(0.01)
            self._write_image(image_path, (0, 255, 0))

            third = cache.get(image_path)
            self.assertFalse(torch.equal(first, third))
            self.assertGreaterEqual(len(list(cache_dir.glob("*.pt"))), 2)

    def test_memory_cache_lru_eviction(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            image_a = tmp_path / "a.png"
            image_b = tmp_path / "b.png"
            image_c = tmp_path / "c.png"
            self._write_image(image_a, (255, 0, 0))
            self._write_image(image_b, (0, 255, 0))
            self._write_image(image_c, (0, 0, 255))

            cache = ImageTensorCache(mode="memory", max_memory_items=2)
            cache.get(image_a)
            cache.get(image_b)
            self.assertEqual(cache.stats()["memory_entries"], 2)

            # Accessing a third image should evict the least-recently used entry.
            cache.get(image_c)
            stats = cache.stats()
            self.assertEqual(stats["memory_entries"], 2)
            self.assertGreaterEqual(stats["evictions"], 1)


if __name__ == "__main__":
    unittest.main()
