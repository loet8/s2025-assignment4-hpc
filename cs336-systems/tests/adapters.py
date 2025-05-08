#!/usr/bin/env python3
from __future__ import annotations
from typing import Type
import torch
from torch.autograd import Function
from torch import Tensor
import triton
import triton.language as tl


def get_rmsnorm_autograd_function_pytorch() -> Type:
    """
    Returns a torch.autograd.Function subclass that implements RMSNorm.
    The expectation is that this class will implement RMSNorm
    using standard PyTorch operations.

    Returns:
        A class object (not an instance of the class)
    """
    class MyRMSNormAutogradFunctionClass(Function):
        @staticmethod
        def forward(ctx, x: Tensor, weight: Tensor) -> Tensor:
            eps = 1e-5
            mu2 = x.pow(2).mean(dim=-1, keepdim=True)
            inv_rms = torch.rsqrt(mu2 + eps)       
            out = x * inv_rms * weight
            ctx.save_for_backward(x, weight)
            ctx.eps = eps
            return out

        @staticmethod
        def backward(ctx, grad_output):
            x, g = ctx.saved_tensors
            eps = ctx.eps
            dg = rmsnorm_backward_g_pytorch(grad_output, x, g)
            dx = rmsnorm_backward_x_pytorch(grad_output, x, g)
            return dx, dg

            raise NotImplementedError("RMSNormForwardOnly only implements the forward pass")

    
    
    return MyRMSNormAutogradFunctionClass
    raise NotImplementedError
    
@triton.jit
def _rmsnorm_fwd_kernel(
        X_ptr, W_ptr, Out_ptr,
        stride_xm, stride_xn, stride_wn,
        M, N, eps,
        BLOCK: tl.constexpr
    ):
        row = tl.program_id(0)
        col = tl.arange(0, BLOCK)
        mask = col < N

        x_ptrs = X_ptr + row * stride_xm + col * stride_xn
        w_ptrs = W_ptr + col * stride_wn

        x = tl.load(x_ptrs, mask=mask, other=0.0)
        w = tl.load(w_ptrs, mask=mask, other=0.0)

        sumsq = tl.sum(x * x, axis=0)
        inv_rms = 1.0 / tl.sqrt(sumsq / N + eps)
        y = x * inv_rms * w

        tl.store(Out_ptr + row * stride_xm + col * stride_xn, y, mask=mask)

@triton.jit       
def _rmsnorm_bwd_fused_kernel(
    X_ptr, W_ptr, GY_ptr, DX_ptr, PDG_ptr,
    stride_xm, stride_xn, stride_wn,
    stride_pgm, stride_pgn,
    M, N, eps, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    mask = col < N

    x  = tl.load(X_ptr  + row*stride_xm + col*stride_xn, mask=mask)
    w  = tl.load(W_ptr  +        col*stride_wn,   mask=mask)
    gy = tl.load(GY_ptr + row*stride_xm + col*stride_xn, mask=mask)

    sumsq   = tl.sum(x * x, axis=0)
    inv_rms = 1.0 / tl.sqrt(sumsq / N + eps)
    inv_rms3 = inv_rms * inv_rms * inv_rms

    dot = tl.sum(w * gy * x, axis=0)

    dx = w * gy * inv_rms \
         - (x * (inv_rms3 / N)) * dot
    tl.store(DX_ptr + row*stride_xm + col*stride_xn, dx, mask=mask)

    pdg = gy * x * inv_rms
    tl.store(PDG_ptr + row*stride_pgm + col*stride_pgn, pdg, mask=mask)

def get_rmsnorm_autograd_function_triton() -> Type:
    """
    Returns a torch.autograd.Function subclass that implements RMSNorm
    using Triton kernels.
    The expectation is that this class will implement the same operations
    as the class you return in get_rmsnorm_autograd_function_pytorch(),
    but it should do so by invoking custom Triton kernels in the forward
    and backward passes.

    Returns:
        A class object (not an instance of the class)
    """

    class RMSNormTriton(Function):
        @staticmethod
        def forward(ctx, x: Tensor, weight: Tensor, eps: float ) -> Tensor:
            orig_shape = x.shape
            H = orig_shape[-1]
            M = x.numel() // H

            x_flat = x.contiguous().view(M, H)
            out_flat = torch.empty_like(x_flat)

            ctx.save_for_backward(x, weight)
            ctx.H = H
            ctx.eps = eps


            stride_xm = H
            stride_xn = 1
            stride_wn = weight.stride(0)

            grid = (M,)
            _rmsnorm_fwd_kernel[grid](
                x_flat, weight, out_flat,
                stride_xm, stride_xn, stride_wn,
                M, H, eps,
                BLOCK=1024, num_warps=4
            )

            return out_flat.view(orig_shape)

        @staticmethod
        def backward(ctx, grad_output):
            x, weight = ctx.saved_tensors
            H    = ctx.H
            eps  = ctx.eps
            orig_shape = grad_output.shape

            M = grad_output.numel() // H
            x_flat  = x.contiguous().view(M, H)
            gy_flat = grad_output.contiguous().view(M, H)

            dx_flat  = torch.empty_like(x_flat)
            pdg_flat = torch.empty_like(x_flat)   

            stride_xm, stride_xn = H, 1
            stride_wn = weight.stride(0)
            stride_pgm, stride_pgn = H, 1

            grid = (M,)
            _rmsnorm_bwd_fused_kernel[grid](
                x_flat, weight, gy_flat,
                dx_flat, pdg_flat,
                stride_xm, stride_xn, stride_wn,
                stride_pgm, stride_pgn,
                M, H, eps,
                BLOCK=1024, num_warps=4
            )

            dx = dx_flat.view(orig_shape)
            dg = pdg_flat.sum(dim=0)

            return dx, dg, None
            raise NotImplementedError("TritonForwardOnly only implements the forward pass")


    return RMSNormTriton
    raise NotImplementedError


def rmsnorm_backward_g_pytorch(
    grad_output: torch.Tensor, x: torch.Tensor, g: torch.Tensor
) -> torch.Tensor:
    """
    Compute the gradient of the RMSNorm operation pass with respect to g.

    Args:
        grad_output: torch.Tensor
            Gradient of the loss with respect to the output of the RMSNorm operation.
            This has the same shape as x.
        x: torch.Tensor
            Input to the RMSNorm operation. Shape: (*, H)
        g: torch.Tensor
            The g learnable parameter of the RMSNorm layer. Shape: (H,)

    Returns:
        Gradient of the loss with respect to g. Shape: (H,)
    """
    eps = 1e-5
    m2 = x.pow(2).mean(dim=-1, keepdim=True)           
    inv_rms = torch.rsqrt(m2 + eps)                    

    contrib = grad_output * x * inv_rms                 

    reduce_dims = tuple(range(x.ndim - 1))
    grad_g = contrib.sum(dim=reduce_dims)               

    return grad_g
    raise NotImplementedError


def rmsnorm_backward_x_pytorch(
    grad_output: torch.Tensor, x: torch.Tensor, g: torch.Tensor
) -> torch.Tensor:
    """
    Compute the gradient of the RMSNorm operation pass with respect to x.

    Args:
        grad_output: torch.Tensor
            Gradient of the loss with respect to the output of the RMSNorm operation.
            This has the same shape as x.
        x: torch.Tensor
            Input to the RMSNorm operation. Shape: (*, H)
        g: torch.Tensor
            The g learnable parameter of the RMSNorm layer. Shape: (H,)

    Returns:
        Gradient of the loss with respect to x. Shape: (*, H)
    """
    H = x.shape[-1]
    eps = 1e-5

    mu2 = x.pow(2).mean(dim=-1, keepdim=True)         
    r   = (mu2 + eps).rsqrt()                         

    inner = (g * grad_output * x).sum(dim=-1, keepdim=True)  

    dx = g * grad_output * r \
       - (x * (r**3 / H)) * inner

    return dx

    raise NotImplementedError


def get_ddp_individual_parameters(module: torch.nn.Module) -> torch.nn.Module:
    """
    Returns a torch.nn.Module container that handles
    parameter broadcasting and gradient synchronization for
    distributed data parallel training.

    This container should overlaps communication with backprop computation
    by asynchronously communicating gradients as they are ready
    in the backward pass. The gradient for each parameter tensor
    is individually communicated.

    Args:
        module: torch.nn.Module
            Underlying model to wrap with DDP.
    Returns:
        Instance of a DDP class.
    """
    # For example: return DDPIndividualParameters(module)
    raise NotImplementedError


def ddp_individual_parameters_on_after_backward(
    ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer
):
    """
    Code to run after the backward pass is completed, but before we take
    an optimizer step.

    Args:
        ddp_model: torch.nn.Module
            DDP-wrapped model.
        optimizer: torch.optim.Optimizer
            Optimizer being used with the DDP-wrapped model.
    """
    # For example: ddp_model.finish_gradient_synchronization()
    raise NotImplementedError


def get_ddp_bucketed(module: torch.nn.Module, bucket_size_mb: float) -> torch.nn.Module:
    """
    Returns a torch.nn.Module container that handles
    parameter broadcasting and gradient synchronization for
    distributed data parallel training.

    This container should overlaps communication with backprop computation
    by asynchronously communicating buckets of gradients as they are ready
    in the backward pass.

    Args:
        module: torch.nn.Module
            Underlying model to wrap with DDP.
        bucket_size_mb: The bucket size, in megabytes. If None, use a single
            bucket of unbounded size.
    Returns:
        Instance of a DDP class.
    """
    raise NotImplementedError


def ddp_bucketed_on_after_backward(
    ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer
):
    """
    Code to run after the backward pass is completed, but before we take
    an optimizer step.

    Args:
        ddp_model: torch.nn.Module
            DDP-wrapped model.
        optimizer: torch.optim.Optimizer
            Optimizer being used with the DDP-wrapped model.
    """
    # For example: ddp_model.finish_gradient_synchronization()
    raise NotImplementedError


def ddp_bucketed_on_train_batch_start(
    ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer
):
    """
    Code to run at the very start of the training step.

    Args:
        ddp_model: torch.nn.Module
            DDP-wrapped model.
        optimizer: torch.optim.Optimizer
            Optimizer being used with the DDP-wrapped model.
    """
    raise NotImplementedError


def get_sharded_optimizer(
    params, optimizer_cls: Type[torch.optim.Optimizer], **kwargs
) -> torch.optim.Optimizer:
    """
    Returns a torch.optim.Optimizer that handles optimizer state sharding
    of the given optimizer_cls on the provided parameters.

    Arguments:
        params (``Iterable``): an ``Iterable`` of :class:`torch.Tensor` s
            or :class:`dict` s giving all parameters, which will be sharded
            across ranks.
        optimizer_class (:class:`torch.nn.Optimizer`): the class of the local
            optimizer.
    Keyword arguments:
        kwargs: keyword arguments to be forwarded to the optimizer constructor.
    Returns:
        Instance of sharded optimizer.
    """
    raise NotImplementedError
