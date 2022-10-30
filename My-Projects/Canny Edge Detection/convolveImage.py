import numpy as np
from seperateKernel import seperateSymetricKernelForConvolution

def seperateConvolution(image, kernel):

    # we want to do seperate convolution over the input image with the gaussian filter
    horizontalKernel, verticalKernel = seperateSymetricKernelForConvolution(kernel)

    N, M = image.shape
    f_out = np.zeros((N, M))
    f_current = np.zeros((N, M))
    x_center, y_center = (len(verticalKernel) - 1) // 2, (len(horizontalKernel) - 1) // 2

    # we first do the vertical filter along the x-axis
    kernelSumVert = 0

    for i in range(len(verticalKernel)):
        kernelSumVert += verticalKernel[i]

    for x in range(N):
        for y in range(M):
            x_cord = x - x_center
            sum = 0

            for k in range(len(verticalKernel)):
                fac = verticalKernel[k]

                if x_cord < 0:
                    pixelValue = fac * image[0][y]
                elif x_cord > N - 1:
                    pixelValue = fac * image[N - 1][y]
                else:
                    pixelValue = fac * image[x_cord][y]

                sum += pixelValue
                x_cord += 1
            f_current[x][y] = sum * 1/kernelSumVert

    # we continue with the horizontal filter along the y-axis, now with the result from the vertical filtering
    kernelSumHor = 0

    for i in range(len(horizontalKernel)):
        kernelSumHor += horizontalKernel[i]

    for x in range(N):
        for y in range(M):
            y_cord = y - y_center
            sum = 0

            for k in range(len(horizontalKernel)):
                fac = horizontalKernel[k]

                if y_cord < 0:
                    pixelValue = fac * f_current[x][0]
                elif y_cord > M - 1:
                    pixelValue = fac * f_current[x][M - 1]
                else:
                    pixelValue = fac * f_current[x][y_cord]

                sum += pixelValue
                y_cord += 1
            f_out[x][y] = sum * 1 / kernelSumHor

    return f_out

def calculateMagnitude(image, gX, gY):

    N, M = image.shape
    f_out = np.zeros((N, M))

    for x in range(N):
        for y in range(M):
            g = np.sqrt(gX[x][y]**2 + gY[x][y]**2)
            f_out[x][y] = g

    return f_out

def convolve(image, kernel):
    N, M = image.shape
    f_out = np.zeros((N, M))
    kernel = np.flipud(np.fliplr(kernel))
    kernelSize = int(np.sqrt(np.prod(kernel.shape)))
    offset = (kernelSize - 1) // 2

    for x in range(N):
        for y in range(M):
            xCurrent = x - offset

            kernelSum = 0
            for i in range(kernelSize):
                yCurrent = y - offset
                for j in range(kernelSize):
                    xSubs, ySubs = xCurrent, yCurrent

                    if xCurrent not in range(N):
                        if xCurrent < 0:
                            xSubs = 0
                        else:
                            xSubs = N - 1

                    if yCurrent not in range(M):
                        if yCurrent < 0:
                            ySubs = 0
                        else:
                            ySubs = M - 1

                    pixelValue = image[xSubs][ySubs] * kernel[i][j]
                    kernelSum += pixelValue

                    yCurrent += 1
                xCurrent += 1

            f_out[x][y] = kernelSum

    return f_out

def magnitudeThinning(edges, gX, gY):
    N, M = edges.shape
    gradiantAngles = np.zeros((N, M))

    for x in range(N):
        for y in range(M):
            angle = np.rad2deg(np.arctan2(gY[x][y], gX[x][y]))
            angleRounded = (round(angle* 1/45) * 45) % 180
            gradiantAngles[x][y] = angleRounded
            print(angleRounded)

            if angleRounded == 90:
                x1, y1 = x, y - 1
                x2, y2 = x, y + 1

                if y1 not in range(M):
                    y1 = 0

                if y2 not in range(M):
                    y2 = M - 1

                mWest = edges[x][y1]
                mEast = edges[x][y2]
                mCurrent = edges[x][y]

                if mCurrent < mWest or mCurrent < mEast:
                    edges[x][y] = 0

            elif angleRounded == 135:
                x1, y1 = x + 1, y - 1
                x2, y2 = x - 1, y + 1

                if x1 not in range(N) or y1 not in range(M):
                    if x1 not in range(N):
                        x1 = N - 1
                    if y1 not in range(M):
                        y1 = 0

                if x2 not in range(N) or y2 not in range(M):
                    if x2 not in range(N):
                        x2 = 0
                    if y2 not in range(M):
                        y2 = M - 1

                mSW = edges[x1][y1]
                mNE = edges[x2][y2]
                mCurrent = edges[x][y]

                if mCurrent < mSW or mCurrent < mNE:
                    edges[x][y] = 0

            elif angleRounded == 0:
                x1, y1 = x - 1, y
                x2, y2 = x + 1, y

                if x1 not in range(N):
                    x1 = 0

                if x2 not in range(N):
                    x2 = N - 1

                mN = edges[x1][y1]
                mS = edges[x2][y2]
                mCurrent = edges[x][y]

                if mCurrent < mN or mCurrent < mS:
                    edges[x][y] = 0

            elif angleRounded == 45:
                x1, y1 = x - 1, y - 1
                x2, y2 = x + 1, y + 1

                if x1 not in range(N) or y1 not in range(M):
                    if x1 not in range(N):
                        x1 = 0
                    if y1 not in range(M):
                        y1 = 0

                if x2 not in range(N) or y2 not in range(M):
                    if x2 not in range(N):
                        x2 = N - 1
                    if y2 not in range(M):
                        y2 = M - 1

                mNW = edges[x1][y1]
                mSE = edges[x2][y2]
                mCurrent = edges[x][y]

                if mCurrent < mNW or mCurrent < mSE:
                    edges[x][y] = 0

    return edges, gradiantAngles


















