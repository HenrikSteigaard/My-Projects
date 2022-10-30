import numpy as np


# this thresholding considers the eight neighbors of every pixel in the input image
def thresHolding(image, low, high, neighborhood):
    N, M = image.shape
    f_low, f_high = np.zeros((N, M)), np.zeros((N, M))
    neihborhoodSizeX, neihborhoodSizeY = neighborhood.shape[0], neighborhood.shape[1]
    offsetX = (neihborhoodSizeX - 1) // 2
    offsetY = (neihborhoodSizeY - 1) // 2

    for x in range(N):
        for y in range(M):
            pixelValue = image[x][y]
            if pixelValue >= high:
                f_high[x][y] = pixelValue

            elif pixelValue >= low:
                f_low[x][y] = pixelValue

    placed = True

    while placed:
        count = 0

        for x in range(N):
            for y in range(M):
                pixelValue = f_high[x][y]

                if pixelValue > 0:
                    xCurrent = x - offsetX

                    for i in range(neihborhoodSizeX):
                        yCurrent = y - offsetY

                        for j in range(neihborhoodSizeY):
                            xSubs, ySubs = xCurrent, yCurrent

                            if xSubs not in range(N):
                                break
                            if ySubs not in range(N):
                                break

                            pixelValue = f_low[xSubs][ySubs]

                            if pixelValue > 0:
                                f_high[xSubs][ySubs] = pixelValue
                                f_low[xSubs][ySubs] = 0
                                count = 1

                            yCurrent += 1
                        xCurrent += 1

        if count == 0:
            placed = False

    return f_high




