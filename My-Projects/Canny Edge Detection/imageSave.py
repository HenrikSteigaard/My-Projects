import imageio


def save(image, fileName):
    imageio.imwrite(fileName, image)