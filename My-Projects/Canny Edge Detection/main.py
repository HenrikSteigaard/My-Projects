from imageRead import readGraytone
from showThis import showImagePlot
from gauss import constructGauss
from convolveImage import seperateConvolution, calculateMagnitude, convolve, magnitudeThinning
from hysteresisThresholding import thresHolding as hysThres
import numpy as np

'''
    ---------------------------------------------------------------------------------------------------------------

                                        ** The Canny Edge Detection **

    ---------------------------------------------------------------------------------------------------------------

                                                 step by step
                                            by Henrik Skaanes Steigard
                            
                                               Thursday march 24th 
    ---------------------------------------------------------------------------------------------------------------
    The six (6) steps of Canny's edge detection
    '''

def main():

    # ** (1) we first read in the image as a gray-tone image **
    image = readGraytone("cellekjerner.png")

    # shows the original input image in gray-tone scale
    showImagePlot(image, "Original input image converted to gray-tone scale")

    # ** (2) construction of the gauss filter **
    gaussFilter = constructGauss(2.25, True)

    # ** (3) seperate convolution over the input gray-tone image with specified gaussian filter **
    gaussImage = seperateConvolution(image, gaussFilter)

    # shows the filtered input graytone-image with a gaussian blur
    showImagePlot(gaussImage, "Result of gaussian filtering with simga = 2.25")

    # imsave the g_x and g_y

    # ** (4) calculatiing the magnitudes **
    # kernels will be rotated during the convolution
    filterX = np.array([[0, -1, 0],
                        [0, 0, 0],
                        [0, 1, 0]])

    filterY = np.array([[0, 0, 0],
                        [-1, 0, 1],
                        [0, 0, 0]])

    gX = convolve(gaussImage, filterX)
    gY = convolve(gaussImage, filterY)

    magnitude = calculateMagnitude(image, gX, gY)
    showImagePlot(magnitude, "Magnitudes of the image using gaussian image as input")

    # ** (5) thinning of the magnitude edges
    thinned, angles = magnitudeThinning(magnitude, gX, gY)
    showImagePlot(thinned, "Thinned magnitudes")
    showImagePlot(angles, "Edge angles of the magnitudes")

    # ** (6) we are now ready for the hysteresis thresholding of the thinned magnitude image as the input image
    neighborhood = np.array([[1, 1, 1],
                             [1, 1, 1],
                             [1, 1, 1]])

    thresholded = hysThres(thinned, 14, 22, neighborhood)
    showImagePlot(thresholded, "Thresholded image with T_low = 14 and T_high = 22")

if __name__ == '__main__':
    main()
