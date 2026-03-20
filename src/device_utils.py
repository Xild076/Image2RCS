import torch
import torch.nn.functional as F
from functools import lru_cache


def mps_is_available() -> bool:
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def _probe_single_cuda_device(index: int) -> tuple[bool, str]:
    try:
        device = torch.device(f"cuda:{index}")
        x = torch.randn((2, 3, 16, 16), device=device, dtype=torch.float32)
        w = torch.randn((4, 3, 3, 3), device=device, dtype=torch.float32)
        y = F.conv2d(x, w, padding=1)
        _ = float(y.mean().item())
        torch.cuda.synchronize(device)
        return True, ""
    except Exception as exc:
        return False, str(exc)


@lru_cache(maxsize=1)
def cuda_probe_results() -> dict[int, tuple[bool, str]]:
    results: dict[int, tuple[bool, str]] = {}
    if not torch.cuda.is_available():
        return results
    for index in range(torch.cuda.device_count()):
        results[index] = _probe_single_cuda_device(index)
    return results


def usable_cuda_device_indices() -> list[int]:
    return [index for index, (ok, _) in cuda_probe_results().items() if ok]


def cuda_probe_failure_summary() -> str:
    failures: list[str] = []
    for index, (ok, reason) in cuda_probe_results().items():
        if ok:
            continue
        name = torch.cuda.get_device_name(index)
        short_reason = reason.splitlines()[0] if reason else "unknown error"
        failures.append(f"cuda:{index} ({name}): {short_reason}")
    return "; ".join(failures)


def resolve_device(requested: str) -> torch.device:
    value = requested.strip().lower()
    if value in {"auto", "gpu"}:
        cuda_devices = usable_cuda_device_indices()
        if cuda_devices:
            return torch.device(f"cuda:{cuda_devices[0]}")
        if mps_is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if value == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("--device=cuda requested, but CUDA is not available on this machine")
        cuda_devices = usable_cuda_device_indices()
        if not cuda_devices:
            details = cuda_probe_failure_summary()
            raise ValueError(
                "--device=cuda requested, but no CUDA device passed a kernel probe. "
                f"Details: {details}"
            )
        return torch.device(f"cuda:{cuda_devices[0]}")

    if value == "mps":
        if not mps_is_available():
            raise ValueError("--device=mps requested, but MPS is not available on this machine")
        return torch.device("mps")

    if value == "cpu":
        return torch.device("cpu")

    try:
        device = torch.device(value)
    except (RuntimeError, TypeError) as exc:
        raise ValueError(
            f"Unsupported --device value '{requested}'. Use auto, gpu, cuda, cuda:N, mps, or cpu"
        ) from exc

    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device requested, but CUDA is not available on this machine")
    if device.type == "cuda":
        cuda_devices = usable_cuda_device_indices()
        if device.index is None:
            if not cuda_devices:
                details = cuda_probe_failure_summary()
                raise ValueError(
                    "CUDA requested, but no CUDA device passed a kernel probe. "
                    f"Details: {details}"
                )
            return torch.device(f"cuda:{cuda_devices[0]}")
        if device.index not in cuda_devices:
            details = cuda_probe_failure_summary()
            raise ValueError(
                f"Requested {device} is not usable with this Torch/CUDA build. "
                f"Details: {details}"
            )
    if device.type == "mps" and not mps_is_available():
        raise ValueError("MPS device requested, but MPS is not available on this machine")
    return device


def configure_torch_for_device(device: torch.device):
    if device.type != "cuda":
        return

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        if device.index is None:
            index = torch.cuda.current_device()
        else:
            index = device.index
        gpu_name = torch.cuda.get_device_name(index)
        capability = torch.cuda.get_device_capability(index)
        return f"cuda:{index} ({gpu_name}, capability={capability[0]}.{capability[1]})"
    if device.type == "mps":
        return "mps (Apple Metal)"
    return str(device)
