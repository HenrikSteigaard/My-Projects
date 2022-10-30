import numpy as np

def constructGauss(sigma, show=False):
    size = int(np.ceil(sigma * 8 + 1))
    gaussian = np.zeros((size, size))
    x_center, y_center = (len(gaussian) - 1) // 2, (len(gaussian) - 1) // 2
    interval = len(gaussian)
    sum = 0

    for x in range(interval):
        x_current = x - x_center

        for y in range(interval):
            y_current = y - y_center

            k = x_current**2 + y_current**2
            l = 2 * sigma**2
            val = round(np.exp(-k/l),3)

            gaussian[x][y] = val
            sum += val

    gaussian = gaussian * 1/sum

    if show:
        print(gaussian)

    return gaussian







