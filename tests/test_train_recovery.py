import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from train import run_epoch_with_loader_recovery


class TestTrainRecovery(unittest.TestCase):
    def test_retries_once_on_worker_failure(self):
        call_count = {"step": 0, "rebuild": 0}

        def train_step():
            call_count["step"] += 1
            if call_count["step"] == 1:
                raise RuntimeError("DataLoader worker (pid 63723) is killed by signal: Killed")
            return (0.1, {"data_load": 0.0, "forward_backward": 0.0})

        def rebuild_safe_loaders(_error):
            call_count["rebuild"] += 1

        (result, recovered) = run_epoch_with_loader_recovery(
            train_step,
            rebuild_safe_loaders,
            loader_recovery="auto",
        )

        self.assertTrue(recovered)
        self.assertEqual(call_count["step"], 2)
        self.assertEqual(call_count["rebuild"], 1)
        self.assertEqual(result[0], 0.1)

    def test_no_retry_when_recovery_disabled(self):
        def train_step():
            raise RuntimeError("DataLoader worker (pid 1) is killed by signal: Killed")

        with self.assertRaises(RuntimeError):
            run_epoch_with_loader_recovery(train_step, lambda _error: None, loader_recovery="off")


if __name__ == "__main__":
    unittest.main()
