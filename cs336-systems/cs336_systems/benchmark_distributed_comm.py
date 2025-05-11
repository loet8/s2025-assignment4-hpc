import os
import time
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank, world_size, backend, device, tensor_size_bytes,
           warmup_iters, timed_iters):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=world_size)

    if device == "cuda":
        torch.cuda.set_device(rank)
        dev = torch.device(f"cuda:{rank}")
    else:
        dev = torch.device("cpu")

    numel = tensor_size_bytes // 4
    data = torch.rand(numel, dtype=torch.float32, device=dev)

    for _ in range(warmup_iters):
        dist.all_reduce(data, async_op=False)
        if device == "cuda":
            torch.cuda.synchronize()

    times = []
    for _ in range(timed_iters):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        dist.all_reduce(data, async_op=False)

        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    dist.barrier()
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, times)

    if rank == 0:
        all_times = [t for sub in gathered for t in sub]
        avg_ms = (sum(all_times) / len(all_times)) * 1e3
        print(f"{backend},{device},{world_size},"
              f"{tensor_size_bytes/1e6:.1f},{avg_ms:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark all-reduce latency (single-node)")
    parser.add_argument("--backends", nargs="+",
                        default=["gloo", "nccl"],
                        help="Which dist backends to test")
    parser.add_argument("--devices", nargs="+",
                        default=["cpu", "cuda"],
                        help="cpu or cuda")
    parser.add_argument("--sizes", nargs="+", type=int,
                        default=[512*1024, 1*1024*1024, 10*1024*1024,
                                 50*1024*1024, 100*1024*1024,
                                 500*1024*1024, 1*1024*1024*1024],
                        help="Tensor sizes in bytes")
    parser.add_argument("--world-sizes", nargs="+", type=int,
                        default=[2, 4, 6],
                        help="Number of processes / GPUs")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Number of warm-up iterations")
    parser.add_argument("--iters", type=int, default=20,
                        help="Number of timed iterations")
    args = parser.parse_args()

    for backend in args.backends:
        for device in args.devices:
            if backend == "nccl" and device == "cpu":
                continue
            for world_size in args.world_sizes:
                if device == "cuda" and world_size > torch.cuda.device_count():
                    continue
                for size in args.sizes:
                    mp.spawn(
                        worker,
                        args=(world_size, backend, device, size,
                              args.warmup, args.iters),
                        nprocs=world_size,
                        join=True
                    )


if __name__ == "__main__":
    main()
