from cs336_basics.model import Transformer
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
        d_model=args.d_model,
        n_heads=args.num_heads,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        max_seq_len=args.seq_len
    ).to(device)
    model.train()

    # Define an input (random)
    inputs = torch.randint(low=0, high=args.vocab_size,size=(args.batch_size, args.seq_len), device=device, dtype=torch.long)

    def run():
        # Run the model `num_steps` times (note: no optimizer updates)
        out = model(inputs)
        if not args.forward_only:
            loss = out.sum()
            loss.backward()
            model.zero_grad(set_to_none=True)
    return run
  
  def mean(x: List[float]) -> float:
    return sum(x) / len(x)

def round1(x: float) -> float:
    """Round to 1 decimal place."""
    return round(x, 1)

def benchmark(description: str, run: Callable, num_warmups, num_trials):
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


def main():
    args = parse_args()
    runner = make_run(args)
    desc   = "FWD" if args.forward_only else "FWD+BWD"
    benchmark(desc, runner, args.warmup_steps, args.measure_steps)
  

if __name__ == "__main__":
    main()
