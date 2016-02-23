import numpy as np
import matplotlib.pyplot as plt


def error_plot():
    error = np.load("other/error_points.npy")
    x = error[:, 1].tolist()
    y = error[:, 2].tolist()

    plt.plot(x, y)
    plt.plot([0, 1], [0, 1])

    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("False Accept Rate")
    plt.ylabel("False Reject Rate")

    plt.show()

if __name__ == "__main__":
    error_plot()
