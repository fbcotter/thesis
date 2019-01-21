""" This script tests that parameterizing in the fourier domain does
not affect optimization """
import torch
import numpy as np
import torch.nn.functional as F
import math
from torch.optim import Optimizer


class CplxAdam(Optimizer):
    r"""Implements Adam algorithm.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    #  denom = exp_avg_sq.sqrt().add_(group['eps'])
                    denom = exp_avg_sq.sum(dim=-1, keepdim=True).sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss


def main():
    N = 3
    pad = (N-1)//2
    index = np.outer(np.arange(N), np.arange(N))
    U = 1/np.sqrt(N) * np.exp(-1j*2*np.pi*index/N)
    Us = np.conj(U)
    #  U = np.exp(-1j*2*np.pi*index/N)
    #  Us = 1/N * np.conj(U)

    w = np.random.randn(5,5,N,N).astype('float32')
    w_hat = np.fft.fft2(w, norm='ortho')
    w_hat = np.stack((w_hat.real, w_hat.imag), axis=-1)

    W = torch.tensor(w, requires_grad=True)
    W_hat = torch.tensor(w_hat, requires_grad=True)

    optim1 = torch.optim.SGD([W,], lr=0.1, momentum=0.9, nesterov=True)
    optim2 = torch.optim.SGD([W_hat,], lr=0.1, momentum=0.9, nesterov=True)
    #  optim1 = torch.optim.Adam([W,], lr=0.01)
    #  optim2 = CplxAdam([W_hat,], lr=0.01)
    #  optim1 = torch.optim.Adagrad([W,], lr=0.1)
    #  optim2 = torch.optim.Adagrad([W_hat,], lr=0.1)
    loss1 = torch.nn.MSELoss()
    loss2 = torch.nn.MSELoss()
    for i in range(10):
        print('Testing step {}'.format(i))
        optim1.zero_grad()
        optim2.zero_grad()

        W2 = torch.irfft(W_hat, signal_ndim=2, onesided=False, normalized=True)
        W2.retain_grad()
        np.testing.assert_array_almost_equal(W2.detach().numpy(), W.detach().numpy(), decimal=4)

        x = torch.randn(10, 5, 32, 32)
        y1 = F.conv2d(x, W, padding=pad)
        y2 = F.conv2d(x, W2, padding=pad)
        np.testing.assert_array_almost_equal(y1.detach().numpy(), y2.detach().numpy(), decimal=4)
        y_target = torch.randn(10, 5, 32, 32)

        output1 = loss1(y1, y_target)
        output2 = loss2(y2, y_target)
        output1.backward()
        optim1.step()
        output2.backward()
        optim2.step()
        np.testing.assert_array_almost_equal(W2.grad.numpy(), W.grad.numpy(), decimal=4)

        # Do some sanity checks on gradients
        a = np.matmul(U, np.matmul(W2.grad.numpy(), U))
        ar, ai = a.real, a.imag
        b = W_hat.grad.numpy()

        np.testing.assert_array_almost_equal(ar, b[...,0], decimal=4)
        np.testing.assert_array_almost_equal(ai, b[...,1], decimal=4)

    print('Done! They matched')

if __name__=='__main__':
    main()

