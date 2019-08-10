""" This script tests that parameterizing in the fourier domain does
not affect optimization """
import torch
import numpy as np
import torch.nn.functional as F


def dft_reg_loss(w, type='l2'):
    if type == 'l2':
        w2 = torch.rfft(w, signal_ndim=2, onesided=False, normalized=True)
        return 0.5*torch.sum(w2[...,0]**2 + w2[...,1]**2)
    else:
        w2 = torch.rfft(w, signal_ndim=2, onesided=False, normalized=True)
        return torch.sum(torch.sqrt(w2[...,0]**2 + w2[...,1]**2))

def reg_loss(w, type='l2'):
    if type == 'l2':
        return 0.5*torch.sum(w**2)
    else:
        return torch.sum(torch.abs(w))

def main():
    wd = 1e-2
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

    optim1 = torch.optim.SGD([W,], lr=0.1, momentum=0.9, weight_decay=0)
    optim2 = torch.optim.SGD([W_hat,], lr=0.1, momentum=0.9, weight_decay=0)
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
        output2.backward()
        # Check the gradients are the same before regularization
        np.testing.assert_array_almost_equal(W2.grad.numpy(), W.grad.numpy(), decimal=4)
        #  reg1 = wd*dft_reg_loss(W)
        reg1 = wd*reg_loss(W, 'l2')
        reg2 = wd*reg_loss(W_hat, 'l2')
        # Do some sanity checks on gradients
        a = np.matmul(U, np.matmul(W2.grad.numpy(), U))
        ar, ai = a.real, a.imag
        b = W_hat.grad.numpy()

        np.testing.assert_array_almost_equal(ar, b[...,0], decimal=4)
        np.testing.assert_array_almost_equal(ai, b[...,1], decimal=4)

        # Add in regularization
        reg1.backward()
        reg2.backward()
        optim1.step()
        optim2.step()


    print('Done! They matched')

if __name__=='__main__':
    main()

