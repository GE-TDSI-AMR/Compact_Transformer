import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline


figures_path = "figures/"


def vis_accuracy(param, train_acc, test_acc, param_name="param"):
    param_, train_acc = smoothing(param, train_acc)
    param, test_acc = smoothing(param, test_acc)
    plt.plot(param, train_acc, color="b", label="train")
    plt.plot(param, test_acc, color="r", label="test")

    # putting labels
    plt.xlabel("Epochs")
    plt.ylabel(param_name)

    #plt.legend(loc='upper center', shadow=True, fontsize='x-large')
    #plt.xscale("log")
    plt.title(f"Evolution of the train and test {param_name} during epochs")

    # function to show plot
    plt.legend()
    plt.show()


def smoothing(x, y):
    x = np.array(x)
    y = np.array(y)
    x_y_spline = make_interp_spline(x, y)

    # Returns evenly spaced numbers
    # over a specified interval.
    x_ = np.linspace(x.min(), x.max(), 500)
    y_ = x_y_spline(x_)

    return x_, y_
