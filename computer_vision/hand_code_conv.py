import numpy as np

def conv(image, filter, threshold):
    image_height, image_width = image.shape
    filter_height, filter_width = filter.shape

    output_width = image_width - filter_width + 1
    output_height = image_height - filter_height + 1

    output = np.zeros((output_height, output_width))

    for y in range(output_height):
        for x in range(output_width):
            region = image[y:y+filter_height, x:x+filter_width]
            value = np.sum(region * filter)
            output[y, x] = value

    result = np.where(value > threshold, 255, 0)
    return output, result

image = np.random.randint(0, 256, (320, 320), dtype = np.uint8)

filter = np.random.randint(-2, 3, (3, 3), dtype = np.int16)

