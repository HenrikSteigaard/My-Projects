import numpy as np

def seperateSymetricKernelForConvolution(kernel):

    # since we are doing convolution, we need to rotate the input filters pi radians, in contrast to correlation
    kernel = np.flipud(np.fliplr(kernel))

    middle = (len(kernel) - 1) // 2
    horizontalKernel = kernel[middle][0:]
    verticalKernel = kernel[0:][middle]

    return horizontalKernel, verticalKernel

