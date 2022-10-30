import imageio

def readGraytone(image):
    return imageio.imread(image, as_gray=True)