import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from train_compression import build_or_load_train_compression


class TestTrainCompression(unittest.TestCase):
    def _write_image(self, path: Path, value: int) -> None:
        image = Image.new("RGB", (24, 24), color=(value, value, value))
        image.save(path)

    def _build_samples(self, root: Path, counts: dict[str, int]) -> list[tuple[Path, float]]:
        samples: list[tuple[Path, float]] = []
        for folder_idx, (folder_name, count) in enumerate(counts.items()):
            folder = root / folder_name
            folder.mkdir(parents=True, exist_ok=True)
            target = float(folder_idx + 1)
            for image_idx in range(count):
                path = folder / f"img_{image_idx:03d}.png"
                self._write_image(path, value=(image_idx * 17 + folder_idx * 13) % 255)
                samples.append((path, target))
        return samples

    def test_deterministic_for_fixed_split(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            images_root = tmp_path / "images"
            csv_path = tmp_path / "labels.csv"
            cache_dir = tmp_path / "compression_cache"

            csv_path.write_text("aircraft,rcs\nA,1.0\n", encoding="utf-8")
            samples = self._build_samples(images_root, {"f1": 12, "f2": 10})

            first = build_or_load_train_compression(
                samples,
                csv_path=str(csv_path),
                images_root=str(images_root),
                hdf5_path=None,
                split_seed=42,
                val_split=0.2,
                group_size=4,
                cache_dir=cache_dir,
                rebuild=True,
            )
            second = build_or_load_train_compression(
                samples,
                csv_path=str(csv_path),
                images_root=str(images_root),
                hdf5_path=None,
                split_seed=42,
                val_split=0.2,
                group_size=4,
                cache_dir=cache_dir,
                rebuild=True,
            )

            self.assertEqual(first.fingerprint, second.fingerprint)
            self.assertEqual(first.sample_weights, second.sample_weights)
            self.assertEqual([str(p) for p, _ in first.samples], [str(p) for p, _ in second.samples])

    def test_expected_ratio_and_weight_sum(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            images_root = tmp_path / "images"
            csv_path = tmp_path / "labels.csv"
            cache_dir = tmp_path / "compression_cache"

            csv_path.write_text("aircraft,rcs\nA,1.0\n", encoding="utf-8")
            samples = self._build_samples(images_root, {"f1": 10, "f2": 6})

            result = build_or_load_train_compression(
                samples,
                csv_path=str(csv_path),
                images_root=str(images_root),
                hdf5_path=None,
                split_seed=7,
                val_split=0.3,
                group_size=4,
                cache_dir=cache_dir,
                rebuild=True,
            )

            expected_compressed = 5  # ceil(10/4) + ceil(6/4)
            self.assertEqual(result.original_count, 16)
            self.assertEqual(result.compressed_count, expected_compressed)
            self.assertEqual(int(result.effective_weight_sum), 16)
            self.assertAlmostEqual(result.ratio, 16 / expected_compressed, places=6)

    def test_cache_hit_and_invalidation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            images_root = tmp_path / "images"
            csv_path = tmp_path / "labels.csv"
            cache_dir = tmp_path / "compression_cache"

            csv_path.write_text("aircraft,rcs\nA,1.0\n", encoding="utf-8")
            samples = self._build_samples(images_root, {"f1": 8, "f2": 8})

            first = build_or_load_train_compression(
                samples,
                csv_path=str(csv_path),
                images_root=str(images_root),
                hdf5_path=None,
                split_seed=1,
                val_split=0.2,
                group_size=4,
                cache_dir=cache_dir,
                rebuild=False,
            )
            second = build_or_load_train_compression(
                samples,
                csv_path=str(csv_path),
                images_root=str(images_root),
                hdf5_path=None,
                split_seed=1,
                val_split=0.2,
                group_size=4,
                cache_dir=cache_dir,
                rebuild=False,
            )

            self.assertFalse(first.cache_hit)
            self.assertTrue(second.cache_hit)

            csv_path.write_text("aircraft,rcs\nA,1.1\n", encoding="utf-8")
            third = build_or_load_train_compression(
                samples,
                csv_path=str(csv_path),
                images_root=str(images_root),
                hdf5_path=None,
                split_seed=1,
                val_split=0.2,
                group_size=4,
                cache_dir=cache_dir,
                rebuild=False,
            )

            self.assertFalse(third.cache_hit)
            self.assertNotEqual(second.fingerprint, third.fingerprint)


if __name__ == "__main__":
    unittest.main()
