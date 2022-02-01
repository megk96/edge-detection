from skimage.io import imread, imsave
from skimage.color import rgb2gray
from skimage import img_as_uint
from skimage.util import random_noise
from scipy.ndimage import gaussian_filter, uniform_filter
from skimage.segmentation import slic, mark_boundaries
from skimage.transform import probabilistic_hough_line
from skimage.feature import canny
import matplotlib.pyplot as plt
import numpy as np
import os

DATA_FOLDER = "../data/image_data"
OUTPUT_FOLDER = "../outputs"
Q1_FILE_NAME = "avengers_imdb.jpg"
Q2_FILE_NAME = "bush_house_wikipedia.jpg"
Q3_FILE_NAME = "forestry_commission_gov_uk.jpg"
Q4_FILE_NAME = "rolland_garros_tv5monde.jpg"

def end_line():
    print(
        "------------------------------------------------------------------------------------------------------------")

# Avengers - greyscale and black and white
def question_one():
    # Reads the input as color
    image_clr = imread(os.path.join(DATA_FOLDER, Q1_FILE_NAME), as_gray=False)
    # Outputs the shape
    print(image_clr.shape)
    image_gray = rgb2gray(image_clr)
    imsave(os.path.join(OUTPUT_FOLDER, "avengers_gray.jpg"), (image_gray * 255).astype(np.uint8))
    thresh = 0.5
    # The pixels are already normalized from 0 to 1 so a 0.5 threshold for black and white works perfectly
    binary = image_gray >= thresh
    imsave(os.path.join(OUTPUT_FOLDER, "avengers_bw.jpg"), img_as_uint(binary))
    end_line()

# Bush House - noise and filters
def question_two():
    image = imread(os.path.join(DATA_FOLDER, Q2_FILE_NAME))
    # The image is first perturbed by random gaussian noise of var=0.1
    gaussian_image = random_noise(image, mode='gaussian', var=0.1)
    imsave(os.path.join(OUTPUT_FOLDER, "bush_house_random_noise.jpg"), (gaussian_image * 255).astype(np.uint8))
    # The perturbed image is then passed through a gaussian filter
    filtered_gaussian = gaussian_filter(gaussian_image, sigma=1)
    imsave(os.path.join(OUTPUT_FOLDER, "bush_house_gaussian_filter.jpg"), (filtered_gaussian * 255).astype(np.uint8))
    # The perturbed image is then passed through a uniform filter
    uniform_on_perturbed = uniform_filter(gaussian_image, size=(9, 9, 1))
    imsave(os.path.join(OUTPUT_FOLDER, "bush_house_uniform_on_perturbed.jpg"),
           (uniform_on_perturbed * 255).astype(np.uint8))
    # The gaussian filtered image is then passed through a uniform filter
    uniform_on_gaussian_filter = uniform_filter(filtered_gaussian, size=(9, 9, 1))
    imsave(os.path.join(OUTPUT_FOLDER, "bush_house_uniform_on_gaussian.jpg"),
           (uniform_on_gaussian_filter * 255).astype(np.uint8))

# Forestry - Segmentation
def question_three():
    image = imread(os.path.join(DATA_FOLDER, Q3_FILE_NAME))
    # Compactness is the control parameter
    segments = slic(image, n_segments=5, compactness=15)
    imsave(os.path.join(OUTPUT_FOLDER, "forestry_segments.jpg"),
           segments)
    # To visually represent the results, mark_boundaries is used to show superimposed image
    superimposed = mark_boundaries(image, segments)
    imsave(os.path.join(OUTPUT_FOLDER, "forestry_superimposed.jpg"),
           superimposed)

# Canny Edge Detection and Hough Transform
def question_four():
    image = imread(os.path.join(DATA_FOLDER, Q4_FILE_NAME), as_gray=True)
    # Gaussian filter is first applied to remove the noise, different values of sigma are experimented before settling on this value - 0.55
    image = gaussian_filter(image, sigma=0.55)
    # canny method is used to perform the edge detection
    canny_image = canny(image, sigma=1)
    imsave(os.path.join(OUTPUT_FOLDER, "roland_garros_canny.jpg"),
           (canny_image * 255).astype(np.uint8))
    # This image is then passed through probabilistic hough line
    lines = probabilistic_hough_line(canny_image, threshold=10, line_length=3,
                                    line_gap=2)
    # After obtaining the Hough Lines, they are plotted using matplotlib
    for line in lines:
        p0, p1 = line
        plt.plot((p0[0], p1[0]), (p0[1], p1[1]))
    plt.gca().invert_yaxis()
    plt.savefig(os.path.join(OUTPUT_FOLDER, "roland_garros_hough_transform.jpg"))

def main():
    question_one()
    question_two()
    question_three()
    question_four()



if __name__ == "__main__":
    main()
