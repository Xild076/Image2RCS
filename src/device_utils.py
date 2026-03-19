import torch


def mps_is_available() -> bool:
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def resolve_device(requested: str) -> torch.device:
    value = requested.strip().lower()
    if value in {"auto", "gpu"}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if mps_is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if value == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("--device=cuda requested, but CUDA is not available on this machine")
        return torch.device("cuda")

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
