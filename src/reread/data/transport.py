import torch


def stack_and_transfer(values, device):
    device = torch.device(device)
    if device.type == "cuda" and values[0].device.type == "cpu":
        first = values[0]
        if any(
            value.device.type != "cpu"
            or value.shape != first.shape
            or value.dtype != first.dtype
            for value in values
        ):
            raise ValueError(
                "Cached CUDA transfer requires matching CPU row shapes and dtypes"
            )
        batch = torch.empty(
            (len(values), *first.shape),
            dtype=first.dtype,
            device="cpu",
            pin_memory=True,
        )
        torch.stack(values, out=batch)
    else:
        batch = torch.stack(values)
    return batch.to(device=device, non_blocking=True)
