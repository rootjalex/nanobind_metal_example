import nanobind_gpu_example
import torch
import time

def test_metal_add_helper(n: int):
    x = torch.randn(n, dtype=torch.float32, device="mps")
    y = torch.randn(n, dtype=torch.float32, device="mps")

    # Pre-allocate output buffers
    r_torch = torch.empty_like(x)  # PyTorch output
    r_metal = torch.empty_like(x)  # Placeholder for Metal output

    iters = 10

    with torch.no_grad():
        # Warmup
        for i in range(iters):
            torch.add(x, y, out=r_torch)
            torch.mps.synchronize()

        torch_times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            torch.add(x, y, out=r_torch)
            torch.mps.synchronize()
            t1 = time.perf_counter()
            torch_times.append((t1 - t0) * 1000.0)  # ms

        # Drop low and high outliers, take average of remaining
        torch_times.sort()
        avg_ms = sum(torch_times[1:-1]) / (len(torch_times) - 2)
        print(f"Torch MPS: {avg_ms:.3f} ms")

    torch.mps.synchronize()
    # Warmup (important to trigger Metal pipeline compilation)
    # _ = nanobind_gpu_example.vecf_add(x, y)
    for i in range(iters):
        nanobind_gpu_example.vecf_add_out(x, y, r_metal)
        nanobind_gpu_example.synchronize()

    # --- nanobind Metal add timing ---

    metal_times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        nanobind_gpu_example.vecf_add_out(x, y, r_metal)
        nanobind_gpu_example.synchronize()
        t1 = time.perf_counter()
        metal_times.append((t1 - t0) * 1000.0)  # ms

    # Drop low and high outliers, take average of remaining
    metal_times.sort()
    avg_ms = sum(metal_times[1:-1]) / (len(metal_times) - 2)
    print(f"nanobind Metal: {avg_ms:.3f} ms")

    # --- Validation ---
    # assert torch.allclose(r_torch, r_metal, atol=1e-6), "Mismatch between Torch and Metal results"
    assert torch.equal(r_torch, r_metal), "Mismatch between Torch and Metal results"

    # --- CPU timing ---
    x_cpu = x.to("cpu")
    y_cpu = y.to("cpu")
    r_cpu = torch.empty_like(x_cpu)

    # Warmup
    for i in range(iters):
        torch.add(x_cpu, y_cpu, out=r_cpu)

    cpu_times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        torch.add(x_cpu, y_cpu, out=r_cpu)
        t1 = time.perf_counter()
        cpu_times.append((t1 - t0) * 1000.0)

    cpu_times.sort()
    avg_ms = sum(cpu_times[1:-1]) / (len(cpu_times) - 2)
    print(f"Torch CPU: {avg_ms:.3f} ms")

    torch_cpu = r_torch.to("cpu")

    assert torch.equal(r_cpu, torch_cpu), "Mismatch between Torch (CPU) and Torch (GPU) results"

if __name__ == "__main__":
    for i in range(2, 10):
        print(f"Input size: {10 ** i}")
        test_metal_add_helper(10 ** i)
