import cv2
import numpy as np

def remove_background(image):
    """Remove background from an RGB image."""
    rgb_planes = cv2.split(image)

    result_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        result_planes.append(diff_img)

    result = cv2.merge(result_planes)
    return result


def sketch_to_binary(image, threshold=100):
    """Convert a shadow-removed grayscale image to a binary mask."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inverted = cv2.bitwise_not(gray)
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    sketch = cv2.divide(inverted, 255 - blurred, scale=256)
    binary = cv2.threshold(sketch, threshold, 255, cv2.THRESH_BINARY)[1]
    return binary


def extract_white_segments_from_path(img_path, threshold=170):
    """Reads a grayscale image from path and extracts white segments using a sharpening filter."""
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    sharpen_kernel = np.array([[0.0625, 0.125, 0.0625],
                               [0.125, 0.25, 0.125],
                               [0.0625, 0.125, 0.0625]])

    sharp_image = cv2.filter2D(image, -1, sharpen_kernel)
    _, white_mask = cv2.threshold(sharp_image, threshold, 255, cv2.THRESH_BINARY)
    white_segment = cv2.bitwise_and(sharp_image, sharp_image, mask=white_mask)

    white_segment[white_segment > threshold] = 255
    white_segment[white_segment != 255] = 0

    return white_segment




