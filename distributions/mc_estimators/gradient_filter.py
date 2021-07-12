import torch.autograd


class __GradFilter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args, **kwargs):
        input_, filter_ = args
        ctx.save_for_backward(filter_)
        return input_

    @staticmethod
    def backward(ctx, *args):
        grad_input = args[0].clone()
        filter_, = ctx.saved_tensors
        return filter_ * grad_input, None


grad_filter = __GradFilter.apply
