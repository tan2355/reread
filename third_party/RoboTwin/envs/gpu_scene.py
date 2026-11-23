import os


def create_scene(engine, config):
    import torch
    import sapien
    from reread.evaluation.runtime_setup import cuda_pci_bus_id, normalize_pci
    index = torch.cuda.current_device()
    expected_index = int(os.environ["NEWM_EVAL_CUDA_INDEX"])
    if index != expected_index:
        raise RuntimeError(f"CUDA device changed inside worker: {index} != {expected_index}")
    pci = normalize_pci(os.environ["FSWM_RENDER_PCI"])
    if cuda_pci_bus_id(index) != pci:
        raise RuntimeError("Renderer PCI no longer matches worker CUDA device")
    sapien.physx.set_scene_config(config)
    system = sapien.render.RenderSystem(sapien.Device("pci:" + pci))
    actual = system.device
    if actual.is_cpu() or not actual.can_render() or normalize_pci(actual.pci_string) != pci:
        raise RuntimeError(f"Simulator renderer GPU mismatch: requested {pci}, got {actual}")
    if actual.cuda_id != index:
        raise RuntimeError(f"Renderer CUDA ordinal mismatch: expected {index}, got {actual.cuda_id}")
    return sapien.Scene([sapien.physx.PhysxCpuSystem(), system])
