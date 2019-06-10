import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl


def main():
    x = np.linspace(-2, 2, 100)
    sig = 1/(1+np.exp(-x))
    tanh = (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
    relu = np.maximum(x, 0)
    plt.figure()
    plt.plot(x, sig, label='sigmoid')
    plt.plot(x, tanh, label='tanh')
    plt.plot(x, relu, label='ReLU')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    mpl.rcParams['font.size'] = 12
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = 'Computer Modern Roman'
    main()
