import matplotlib.pyplot as plt
from matplotlib import pyplot


def showImagePlot(image, title=None):
    plt.figure()
    if title != None:
        plt.title(title)

    plt.imshow(image, cmap='gray')
    plt.show()