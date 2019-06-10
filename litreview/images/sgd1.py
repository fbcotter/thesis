import matplotlib.pyplot as plt
import numpy as np
import torch

def objective(X, Y):
    if isinstance(X, torch.Tensor):
        return torch.log(6*X**2 + 2*4*X*Y + 6*Y**2 + 0.1)
        #  return 6*X**2 + 2*4*X*Y + 6*Y**2 + 0.1
    else:
        return np.log(6*X**2 + 2*4*X*Y + 6*Y**2 + 0.1)
        #  return 6*X**2 + 2*4*X*Y + 6*Y**2 + 0.1


def objective2(X, Y):
    return 0.5*(X**2 + 10*Y**2)

def main():
    X, Y = np.meshgrid(np.linspace(-3,3,100), np.linspace(-3,3,100))
    Z = objective(X, Y)
    plt.figure()
    plt.contour(X, Y, Z)

    #  x = torch.randn(1,requires_grad=True)
    #  y = torch.randn(1,requires_grad=True)
    N = 10
    path = np.zeros((3, N+1, 2))
    x_init = [-2., -1., 2.]
    y_init = [0., 2., 0.]

    for i, lr in enumerate([0.3, 0.1, 0.08]):
        #  x = torch.tensor(10., requires_grad=True)
        #  y = torch.tensor(1., requires_grad=True)
        x = torch.tensor(x_init[i],requires_grad=True)
        y = torch.tensor(y_init[i],requires_grad=True)
        optim = torch.optim.SGD([x, y], lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=0.9)

        path[i, 0] = np.array([x.item(), y.item()])
        for j in range(N):
            z = objective(x, y)
            z.backward()
            optim.step()
            path[i, j+1] = np.array([x.item(), y.item()])
            scheduler.step()

        #  plt.scatter(path[:,0], path[:,1])
    for i in range(3):
        plt.plot(path[i,:,0], path[i,:,1], lw=0.2, marker='.')
    plt.show()


if __name__ == '__main__':
    main()
