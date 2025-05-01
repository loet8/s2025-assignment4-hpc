import sys
sys.path.insert(0, "/content/s2025-assignment4-hpc/cs336-basics")

from torch.profiler import profile, record_function, ProfilerActivity
from torch.optim import AdamW
from cs336_basics.model import BasicsTransformerLM as Transformer
import argparse, timeit, torch, statistics
from typing import Callable, List

def parse_args():
    p = argparse.ArgumentParser(description="End-to-end benchmarking of Transformer")
    p.add_argument("--d_model",    type=int,   default=768,  help="Hidden size")
    p.add_argument("--d_ff",       type=int,   default=3072, help="Feed-forward inner size")
    p.add_argument("--num_layers", type=int,   default=12,   help="Number of Transformer layers")
    p.add_argument("--num_heads",  type=int,   default=12,   help="Number of attention heads")
    p.add_argument("--vocab_size", type=int,   default=10000,help="Vocabulary size")
    p.add_argument("--seq_len",    type=int,   default=128,  help="Context length")
    p.add_argument("--batch_size", type=int,   default=16,   help="Batch size")
    p.add_argument("--warmup_steps",   "-w", type=int, default=1, help="Number of warm-up iterations")
    p.add_argument("--measure_steps", "-n", type=int, default=5, help="Number of measured iterations")
    p.add_argument("--forward_only",  action="store_true")
    p.add_argument("--backward_only", action="store_true")
    p.add_argument("--profile", action="store_true")

    return p.parse_args()



def get_device(index: int = 0) -> torch.device:
    """Try to use the GPU if possible, otherwise, use CPU."""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{index}")
    else:
        return torch.device("cpu")

def make_run(args) -> Callable:
    device = get_device()
    # Define a model 
    model = Transformer(
        vocab_size=args.vocab_size,
        context_length = args.seq_len,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
    ).to(device)
    model.train()

    # Define an input (random)
    inputs = torch.randint(low=0, high=args.vocab_size,size=(args.batch_size, args.seq_len), device=device, dtype=torch.long)

    #forward-only
    if args.forward_only:
        def run_fwd():
            _ = model(inputs)
        return run_fwd

    #backward-only
    if args.backward_only:
        # build the graph once
        out  = model(inputs)
        loss = out.sum()
        def run_bwd():
            loss.backward(retain_graph=True)
            model.zero_grad(set_to_none=True)
        return run_bwd

    #forward+backward
    def run_fb():
        out = model(inputs)
        loss = out.sum()
        loss.backward()
        model.zero_grad(set_to_none=True)
    return run_fb
  
def mean(x: List[float]) -> float:
    return sum(x) / len(x)

def round1(x: float) -> float:
    """Round to 1 decimal place."""
    return round(x, 1)
  
def benchmark(description: str, run: Callable, num_warmups, num_trials):
    """Benchmark `func` by running it `num_trials`, and return all the times."""
    # Warmup: first times might be slower due to compilation, things not cached.
    # Since we will run the kernel multiple times, the timing that matters is steady state.
    for _ in range(num_warmups):
        run()
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)

    times: List[float] = []
    for trial in range(num_trials):  # Do it multiple times to capture variance
        start_time = timeit.default_timer()

        run()  # Actually perform computation
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Wait for CUDA threads to finish (important!)

        end_time = timeit.default_timer()
        times.append((end_time - start_time) * 1000)

    mean_time = mean(times)
    print(f"{description}: {list(map(round1, sorted(times)))} (mean {round1(mean_time)} ms)")

    mean_time_stats = statistics.mean(times)
    std_time  = statistics.stdev(times) if len(times)>1 else 0.0

    print(f"{description} over {num_trials} steps: "
    f"{mean_time_stats:.2f}±{std_time:.2f} ms per step")



def profile_xl(args):
  device = get_device()
  xl_model = Transformer(
        vocab_size=args.vocab_size,
        context_length = args.seq_len,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
    ).to(device)
  optimizer = AdamW(xl_model.parameters())
  inputs = torch.randint(low=args.vocab_size, high=args.vocab_size,size=(args.batch_size, args.seq_len), device=device, dtype=torch.long)
  def run_step():
        with record_function("forward_pass"):
            out = model(inputs)
        with record_function("backward_pass"):
            loss = out.sum()
            loss.backward()
        with record_function("optimizer"):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    for _ in range(args.warmup_steps):
        run_step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
        record_shapes=True, profile_memory=False, with_stack=True
    ) as prof:
        for _ in range(args.measure_steps):
            run_step()
            prof.step()

    prof.export_stacks("xl_profiler_stacks.txt", "self_cuda_time_total")
    print(prof.key_averages().table(
        sort_by="cuda_time_total", row_limit=50
    ))
  


def main():
    args = parse_args()
    if args.profile:
      profile_xl(args)
      return
      
    runner = make_run(args)
    if args.forward_only:
        desc = "FWD only"
    elif args.backward_only:
        desc = "BWD only"
    else:
        desc = "FWD+BWD"

    benchmark(desc, runner, args.warmup_steps, args.measure_steps)
  
  

if __name__ == "__main__":
    main()
