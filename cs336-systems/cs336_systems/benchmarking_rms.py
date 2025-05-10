import sys, os


repo_root = os.path.abspath(os.path.join(__file__, "..", ".."))
sys.path.insert(0, repo_root)
sys.path.insert(0, os.path.join(repo_root, "cs336-basics"))
sys.path.insert(0, os.path.join(repo_root, "cs336-systems"))

from torch.profiler import profile, record_function, ProfilerActivity, schedule
import torch.cuda.memory as _mem
from torch.optim import AdamW
from contextlib import nullcontext
from torch.cuda.amp    import autocast
from cs336_basics.model import BasicsTransformerLM as Transformer
from cs336_basics.model import RMSNorm
import argparse, timeit, torch, statistics
from typing import Callable, List
from torch.utils.checkpoint import checkpoint  
from torch.nn import LayerNorm
from tests import adapters
from functools import partial
import torch


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
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--norm_benchmark", action="store_true")
    p.add_argument("--norm_type", choices=["rms","layer", "triton"])
    p.add_argument("--norm_fb", action="store_true")
    p.add_argument("--compile_model", action="store_true")
    p.add_argument("--mem-profile", action="store_true")

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
    
def forward_blocks(x: torch.Tensor, model: Transformer, args) -> torch.Tensor:
    pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
    x = model.token_embeddings(x) + model.position_embeddings(pos)
    for block in model.layers:
        if args.gradient_checkpointing:
            x = checkpoint(block, x)
        else:
            x = block(x)
    x = model.ln_final(x)
    return model.lm_head(x)

def make_run(args) -> Callable:
    device = get_device()
    model = Transformer(
        vocab_size=args.vocab_size,
        context_length = args.seq_len,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        norm_type=args.norm_type,
    ).to(device)
    
    if args.compile_model:
        model = torch.compile(model)
    model.train()

    inputs = torch.randint(low=0, high=args.vocab_size,size=(args.batch_size, args.seq_len), device=device, dtype=torch.long)
    amp_ctx = autocast() if args.mixed_precision else nullcontext()

    if args.forward_only:
        def run_fwd():
            with amp_ctx:
                _ = forward_blocks(inputs, model, args)
        return run_fwd

    if args.backward_only:
        with amp_ctx:
            out  = forward_blocks(inputs, model, args)
            loss = out.sum()
        def run_bwd():
            with amp_ctx:
                loss.backward(retain_graph=True)
            model.zero_grad(set_to_none=True)
        return run_bwd

    def run_fb():
        with amp_ctx:
            out  = forward_blocks(inputs, model, args)
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
        norm_type=args.norm_type,
    ).to(device)
    optimizer = AdamW(xl_model.parameters())
    inputs = torch.randint(low=0, high=args.vocab_size,size=(args.batch_size, args.seq_len), device=device, dtype=torch.long)
    amp_ctx = autocast() if args.mixed_precision else nullcontext()

    def run_step():
        with record_function("forward_pass"):
            with amp_ctx:
                out = forward_blocks(inputs, xl_model, args)
        if not args.forward_only:
            with record_function("backward_pass"):
                with amp_ctx:
                    loss = out.sum(); loss.backward()
            with record_function("optimizer"):
                optimizer.step(); optimizer.zero_grad(set_to_none=True)

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

class RMSNormPyFunctionWrapper(torch.nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        self.func = adapters.get_rmsnorm_autograd_function_pytorch()

    def forward(self, x: torch.Tensor):
        return self.func.apply(x, self.weight)

class RMSNormTritonWrapper(torch.nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        self.func = adapters.get_rmsnorm_autograd_function_triton()

    def forward(self, x: torch.Tensor):
        return self.func.apply(x, self.weight, self.eps)


def benchmark_norms():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N_ROWS = 50_000
    DIMS   = [1024, 2048, 4096, 8192]
    N_ITERS = 1000

    print("| hidden_dim | RMSNorm (ms) | RMSNorm_py (ms) | RMSNorm_py Compiled (ms) | TritonRMS (ms) | LayerNorm (ms) |")
    print("|-----------:|-------------:|----------------:|-------------------------:|---------------:|---------------:|")

    for dim in DIMS:
        x = torch.randn(N_ROWS, dim, device=device, dtype=torch.float32)

        rms_norm = RMSNorm(hidden_size=dim).to(device).eval()
        rms_py = RMSNormPyFunctionWrapper(hidden_size=dim).to(device).eval()
        rms_py_c = torch.compile(rms_py)
        rms_tr = RMSNormTritonWrapper(hidden_size=dim).to(device).eval()

        ln  = LayerNorm(dim).to(device).eval()

        for _ in range(10):
            _ = rms_norm(x)
            _ = rms_py(x) 
            _ = rms_py_c(x)
            _ = rms_tr(x)
            _ = ln(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

        def time_it(m: torch.nn.Module):
            t = statistics.mean(
                timeit.repeat(lambda: m(x), repeat=3, number=N_ITERS)
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            return (t / N_ITERS) * 1000

        t_norm = time_it(rms_norm)
        t_py   = time_it(rms_py)
        t_py_c = time_it(rms_py_c)
        t_tr = time_it(rms_tr)
        t_ln     = time_it(ln)


        print(f"| {dim:4d} | {t_norm:15.3f} | {t_py:15.3f} | {t_py_c:15.3f} | {t_tr:15.3f} | {t_ln:15.3f} |")

def benchmark_norms_fb():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N_ROWS = 50_000
    DIMS   = [1024, 2048, 4096, 8192]
    N_ITERS = 1000

    print("| hidden_dim | RMSNorm-FB (ms) | RMSNorm_py-FB (ms) | RMSNorm_py Compiled (ms) | TritonRMS-FB (ms) | LayerNorm-FB (ms) |")
    print("|-----------:|----------------:|-------------------:|-------------------------:|------------------:|------------------:|")

    for dim in DIMS:
        x = torch.randn(N_ROWS, dim, device=device).requires_grad_(True)
        dy = torch.randn_like(x)

        mods = {
            "RMSNorm": RMSNorm(hidden_size=dim).to(device).eval(),
            "RMSNorm_py": RMSNormPyFunctionWrapper(dim).to(device).eval(),
            "Compiled_RMSNorm_py": torch.compile("RMSNorm_py"),
            "Triton": RMSNormTritonWrapper(dim).to(device).eval(),
            "LayerNorm":     LayerNorm(dim).to(device).eval()
        }

        for _ in range(10):
            for m in mods.values():
                out = m(x)
                out.backward(dy, retain_graph=True)
                x.grad         = None
                m.weight.grad  = None
        if device.type == "cuda":
            torch.cuda.synchronize()

        def time_mod_fb(m):
            def run_once():
                out = m(x)
                out.backward(dy, retain_graph=True)
                x.grad        = None
                m.weight.grad = None
            t = statistics.mean(timeit.repeat(run_once, repeat=3, number=N_ITERS))
            if device.type == "cuda":
                torch.cuda.synchronize()
            return (t / N_ITERS) * 1000

        times_fb = {k: time_mod_fb(m) for k,m in mods.items()}

        print(f"| {dim:4d} | {times_fb['RMSNorm']:17.3f} | {times_fb['RMSNorm_py']:17.3f} | {times_fb['Compiled_RMSNorm_py']:17.3f} | {times_fb['Triton']:17.3f} | {times_fb['LayerNorm']:17.3f} |")

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

    if args.norm_benchmark:
        if args.norm_fb:
            benchmark_norms_fb()
        else:
            benchmark_norms()
        return
    
    if args.mem_profile:
        _mem._record_memory_history(max_entries=1_000_000)

        n_steps = args.measure_steps
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=0, warmup=0, active=1, repeat=n_steps),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as mem_prof:
            for _ in range(n_steps):
                runner()            
                mem_prof.step()     
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

        mem_prof.export_memory_timeline("timeline.html", device=get_device())
        _mem._dump_snapshot("memory_snapshot.pickle")
        _mem._record_memory_history(enabled=None)

        print("Wrote timeline.html & memory_snapshot.pickle")
        return

    benchmark(desc, runner, args.warmup_steps, args.measure_steps)
  
  

if __name__ == "__main__":
    main()
