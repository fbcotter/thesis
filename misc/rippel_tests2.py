""" This script tests that parameterizing in the dtcwt domain does
not affect optimization """
import torch
import numpy as np
import torch.nn.functional as F
from pytorch_wavelets import DTCWTForward, DTCWTInverse


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
    xfm = DTCWTForward(J=1)
    #  xfm.h0o = xfm.h0a
    #  xfm.h1o = xfm.h1a
    ifm = DTCWTInverse()
    #  ifm.g0o = ifm.g0a
    #  ifm.g1o = ifm.g1a
    b1 = (ifm.g0o.data.numpy().ravel()[::-1], ifm.g1o.data.numpy().ravel()[::-1])
    xfm2 = DTCWTForward(J=1, biort=b1)
    b1 = (np.copy(xfm.h0o.data.numpy().ravel()[::-1]), np.copy(xfm.h1o.data.numpy().ravel()[::-1]))
    ifm2 = DTCWTInverse(biort=b1)
    #  xfm2 = xfm
    #  ifm2 = ifm
    wd = 1e-2
    N = 8
    pad = (N-1)//2
    #  U = np.exp(-1j*2*np.pi*index/N)
    #  Us = 1/N * np.conj(U)

    w = np.random.randn(8,5,N,N).astype('float32')
    W = torch.randn(8, 5, N, N, requires_grad=True)
    W_hat_lp, (W_hat_bp,) = xfm(W.data)
    W_hat_lp.requires_grad = True
    W_hat_bp.requires_grad = True

    optim1 = torch.optim.SGD([W,], lr=0.1, momentum=0.0, weight_decay=0)
    optim2 = torch.optim.SGD([W_hat_lp, W_hat_bp], lr=0.1, momentum=0.0, weight_decay=0)
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

        W2 = ifm((W_hat_lp, (W_hat_bp,)))
        W2.retain_grad()
        np.testing.assert_array_almost_equal(W2.detach().numpy(), W.detach().numpy(), decimal=4)

        x = torch.randn(10, 5, 32, 32)
        y1 = F.conv2d(x, W, padding=pad)
        y2 = F.conv2d(x, W2, padding=pad)
        np.testing.assert_array_almost_equal(y1.detach().numpy(), y2.detach().numpy(), decimal=4)
        y_target = torch.randn(10, 8, 31, 31)

        output1 = loss1(y1, y_target)
        output2 = loss2(y2, y_target)
        output1.backward()
        output2.backward()

        # Check the gradients are the same before regularization
        np.testing.assert_array_almost_equal(W2.grad.numpy(), W.grad.numpy(), decimal=4)

        reg1 = wd*reg_loss(W, 'l2')
        reg2 = wd*reg_loss(W_hat_lp, 'l2') + wd*reg_loss(W_hat_bp, 'l2')

        # Do some sanity checks on gradients
        np.testing.assert_array_almost_equal(W.grad.data.numpy(), W2.grad.data.numpy(), decimal=4)

        # DTCWT = a, DTCWT_grad = b, pixel = c, pixel_grad = d
        a_lp = W_hat_lp.data.clone()
        a_bp = W_hat_bp.data.clone()
        da_lp = W_hat_lp.grad.data.clone()
        da_bp = W_hat_bp.grad.data.clone()
        b = W.data.clone()
        db = W.grad.data.clone()

        # Check that c -> a and d -> b
        b_lp, (b_bp,) = xfm(b)
        db_lp, (db_bp,) = xfm2(db)
        np.testing.assert_array_almost_equal(a_lp, b_lp, decimal=4)
        np.testing.assert_array_almost_equal(a_bp, b_bp, decimal=4)
        np.testing.assert_array_almost_equal(da_lp, db_lp, decimal=4)
        np.testing.assert_array_almost_equal(da_bp, db_bp, decimal=4)

        # Check that a -> c and b -> d
        a = ifm((a_lp, (a_bp,)))
        da = ifm2((da_lp, (da_bp,)))
        np.testing.assert_array_almost_equal(a, b, decimal=4)
        np.testing.assert_array_almost_equal(da, db, decimal=4)

        # Add in regularization
        #  reg1.backward()
        #  reg2.backward()
        optim1.step()
        optim2.step()


    print('Done! They matched')

if __name__=='__main__':
    main()


