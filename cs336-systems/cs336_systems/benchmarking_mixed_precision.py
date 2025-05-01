import sys, os
sys.path.insert(0, "/content/s2025-assignment4-hpc/cs336-basics")

from torch.profiler import profile, record_function, ProfilerActivity
from torch.optim import AdamW
from contextlib import nullcontext
from torch.cuda.amp    import autocast
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
    p.add_argument("--mixed_precision", action="store_true")

    return p.parse_args()

def create_flame_graph(in_path: str, out_path: str):
    if not os.path.exists("FlameGraph"):
        os.system("git clone https://github.com/brendangregg/FlameGraph")
    os.system(
        f"FlameGraph/flamegraph.pl "
        f"--title \"CUDA time\" --countname \"us\" {in_path} > {out_path}"
    )

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
    amp_ctx = autocast() if args.mixed_precision else nullcontext()

    #forward-only
    if args.forward_only:
        def run_fwd():
            with amp_ctx:
                _ = model(inputs)
        return run_fwd

    #backward-only
    if args.backward_only:
        # build the graph once
        with amp_ctx:
            out  = model(inputs)
            loss = out.sum()
        def run_bwd():
            with amp_ctx:
                loss.backward(retain_graph=True)
            model.zero_grad(set_to_none=True)
        return run_bwd

    def run_fb():
        with amp_ctx:
            out  = model(inputs)
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
    for _ in range(num_warmups):
        run()
    if torch.cuda.is_available():
        torch.cuda.synchronize()  

    times: List[float] = []
    for _ in range(num_trials):  
        start_time = timeit.default_timer()

        run()  
        if torch.cuda.is_available():
            torch.cuda.synchronize()  

        end_time = timeit.default_timer()
        times.append((end_time - start_time) * 1000)

    m = statistics.mean(times)
    s = statistics.stdev(times) if len(times)>1 else 0.0
    print(f"{description} over {num_trials} steps: {m:.1f}±{s:.1f} ms per step")




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
    inputs = torch.randint(low=0, high=args.vocab_size,size=(args.batch_size, args.seq_len), device=device, dtype=torch.long)
    amp_ctx = autocast() if args.mixed_precision else nullcontext()

    def run_step():
        with record_function("forward_pass"):
            with amp_ctx:
                out = xl_model(inputs)
        if not args.forward_only:
            with record_function("backward_pass"):
                with amp_ctx:
                    loss = out.sum()
                    loss.backward()
            with record_function("optimizer"):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
  
    for _ in range(args.warmup_steps):
        run_step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    stacks_file = "xl_profiler_stacks.txt"
    svg_file    = "xl-flame-graph.svg"
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
        record_shapes=True, profile_memory=False, with_stack=True
    ) as prof:
        for _ in range(args.measure_steps):
            run_step()
            prof.step()
            if device.type=="cuda": torch.cuda.synchronize()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
    prof.export_stacks(stacks_file, "self_cuda_time_total")

    create_flame_graph(stacks_file, svg_file)
    print(f"Flame graph written to {svg_file}")



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
