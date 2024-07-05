import cv2 as cv
import numpy as np
import random
import sys
from numpy.random import randint
import time


def display_image(window_name, img):
    """
    Displays image with given window name.
    :param window_name: name of the window
    :param img: image object to display
    """
    cv.imshow(window_name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def my_equalizing(img):
    """
    Histogram equalization of the sample image
    :param img: Sample image
    :return: Historgram equalization
    """
    values = np.unique(img.flatten(), return_counts=True)
    values = dict(zip(values[0], values[1]))
    keys = sorted(values)

    omega = img.shape[0] * img.shape[1]
    p = lambda y, values: values[y] / omega

    def F(values, keys, y):
        index = 0
        ssum = 0

        while index < len(keys) and keys[index] <= y:
            ssum += p(keys[index], values)
            index += 1
        return ssum

    result = np.empty_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            result[i][j] = F(values, keys, img[i][j]) * 255

    return result


def integral_image(img):
    """
    Computes an integral image using intensity image
    :param img: image object
    :return: an integral image: 2d np-array
    """
    integ_img = np.array(np.zeros((img.shape[0], img.shape[1])))
    integ_img[0][0] = img[0][0]
    for i in range(img.shape[0]):
        integ_img[i][0] = img[i][0] + integ_img[i - 1][0]
    for j in range(img.shape[1]):
        integ_img[0][j] = img[0][j] + integ_img[0][j - 1]

    for i in range(1, img.shape[0]):
        for j in range(1, img.shape[1]):
            integ_img[i][j] = integ_img[i - 1][j] + integ_img[i][j - 1] - integ_img[i - 1][j - 1] + img[i][j]

    return integ_img


def mean_grey_value(img, x0, y0, x1, y1):
    """
    Computes mean grey value of the intensity image using formula
    :param img: Image object
    :param x0, y0: left-top coordinate of the area
    :param x1, y1: right-bottom coordinate of the area
    :return: Mean grey value: float
    """
    return (img[x1 - 1][y1 - 1] - img[x0][y1 - 1] - img[x1 - 1][y0] + img[x0][y0]) / (x1 - x0) / (y1 - y0)


def mean_grey_value_sum(img, x0, y0, x1, y1):
    """
    Computes mean grey value of the intensity image using pixel-wise sum
    :param img: Image object
    :param x0, y0: left-top coordinate of the area
    :param x1, y1: right-bottom coordinate of the area
    :return: Mean grey value: float
    """
    m_value = 0

    for i in range(x0 + 1, x1):
        for j in range(y0 + 1, y1):
            m_value += img[i][j]
    return m_value / (x1 - x0) / (y1 - y0)


def salt_paper_noise(img, chance):
    """
    Add salt and paper noise (black and white pixels) to intensity image with a certain chance
    :param img: Image object
    :param chance: Chance of adding noise [0.0-1.0]
    :return: Noised image: 2d np-array
    """
    numb_noised_pix = int(img.shape[0] * img.shape[1] * chance)
    for i in range(numb_noised_pix):
        x, y = random.randint(0, img.shape[0] - 1), random.randint(0, img.shape[1] - 1)
        r = randint(0, 2)
        img[x][y] = 255 if r == 0 else 0

    return img


def find_closest_mean_value(original_mean, list_of_means):
    """
    Find kernel size with which mean grey value is the closest
    :param original_mean: Mean grey value of the original sample
    :param list_of_means: List of mean grey values with different kernel sizes [(mean grey value, kernel_size)]
    :return: (the closest mean grey value, kernel size)
    """
    m_mean = list_of_means[0]
    for i in list_of_means:
        if abs(i[0] - original_mean) < abs(m_mean[0] - original_mean):
            m_mean = i
    return m_mean


def my_get_Gaussian_kernel_1D(sigma, ksize=None):
    """
    Getting Gaussian 1d kernel
    :param sigma: Sigma sample
    :param ksize: size of kernel
    :return: Gaussian 1d kernel
    """
    if ksize is None:
        ksize = round(sigma * 3)

    G = np.ones((ksize, 1))

    for i in range(len(G)):
        G[i] = np.exp(-(i - (ksize - 1) / 2) ** 2 / (2 * sigma ** 2))

    alpha = 1 / G.sum()
    G = G * alpha

    return G


def my_get_Gaussian_kernel_2D(sigma, ksize=None):
    """
    Getting Gaussian 2d kernel
    :param sigma: Sigma sample
    :param ksize: Kernel size
    :return: Gaussian kernel 2d
    """
    if ksize is None:
        ksize = round(sigma * 3)
        if ksize > 1 and ksize % 2 == 0:
            ksize -= 1

    G = np.ones((ksize, ksize))

    for i in range(ksize):
        for j in range(ksize):
            G[i][i] = np.exp(- ((i - (ksize - 1) / 2) ** 2 + (j - (ksize - 1) / 2) ** 2) / (2 * sigma ** 2))

    alpha = 1 / G.sum()
    G = G * alpha

    return G


if __name__ == '__main__':
    # set image path
    img_path = 'bonn.png'

    # 1a. Compute and display the integral image without using the function integral
    img = cv.imread(img_path)
    intensity_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    integ_img = integral_image(intensity_image)
    cv_integ_img = cv.integral(intensity_image)
    # display_image('1a. Intensity Image', integ_img)
    cv_integ_img = cv_integ_img[1:, 1:]

    # 1b. Compute the mean grey value of the image
    n, m = integ_img.shape[0], integ_img.shape[1]
    print("Mean grey value of the image by OpenCV = {:5f}".format(mean_grey_value(integ_img, 0, 0, n, m)))
    print("Mean grey value of the image by own function = {:5f}".format(mean_grey_value(cv_integ_img, 0, 0, n, m)))
    print("Mean grey value of the image by sum = {:5f}".format(mean_grey_value_sum(intensity_image, 0, 0, n, m)))
    print("-" * 50 + "\n")

    # 1c. Select 10 random squares of size 100×100 within the image and compute the
    # mean gray value using the three versions. Output the runtime of this task for
    # the three versions in seconds using time.
    for i in range(10):
        x0, y0 = int(random.random() * 199), int(random.random() * 379)
        start = time.time()
        print("Mean grey value by OpenCV = {:.3f}, time = {:.3f}".format(
            mean_grey_value(integ_img, x0, y0, x0 + 100, y0 + 100), time.time() - start))

        start = time.time()
        print("Mean grey value by own function = {:.3f}, time = {:.3f}".format(
            mean_grey_value(cv_integ_img, x0, y0, x0 + 100, y0 + 100),
            time.time() - start))
        start = time.time()
        print("Mean grey value by sum = {:.3f}, time = {:.3f}".format(
            mean_grey_value_sum(intensity_image, x0, y0, x0 + 100, y0 + 100),
            time.time() - start))
        print("-" * 50)

# 2. Read the image and convert the image into a gray image and perform histogram
    # equalization
    # • using equalizeHist
    # • using your own implementation of the function equalizeHist
    # and display both results. Compute the absolute pixelwise difference between the
    # results and print the maximum pixel error.
    # set image path
    img_path = 'bonn.png'

    # read and display the image
    img = cv.imread(img_path)

    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    display_image('Gray Image', img_gray)

    result_cv_func = cv.equalizeHist(img_gray)
    result_my_func = my_equalizing(img_gray)

    display_image('Gray equalized Image by openCV', result_cv_func)
    display_image('Gray equalized Image custom function', result_my_func)

    diff_res = abs(result_cv_func.astype(np.int16) - result_my_func.astype(np.int16))
    print(f"Maximal difference between pixles is {diff_res.max()}")

    # cv.imwrite('my_equalized_image.jpg', result)

# 4. Read the image convert it into a gray image, and display it. Filter the
    # image with a Gaussian kernel with σ = 2√2
    # • using GaussianBlur
    # • using filter2D without using getGaussianKernel
    # • using sepFilter2D without using getGaussianKernel
    # and display the three results. Compute the absolute pixel-wise difference between
    # all pairs (there are three pairs) and print the maximum pixel error for each pair.

    # set image path
    img_path = 'bonn.png'

    # read and display the image
    img = cv.imread(img_path)

    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    display_image('Original image', img_gray)

    SIGMA = 2 * 2 ** 0.5
    ksize = round(3 * SIGMA)

    kernel1D = my_get_Gaussian_kernel_1D(SIGMA, ksize)
    kernel2D = my_get_Gaussian_kernel_2D(SIGMA, ksize)

    img_res_1 = cv.GaussianBlur(img_gray, (0, 0), SIGMA)
    img_res_2 = cv.filter2D(img_gray, -1, kernel2D)
    img_res_3 = cv.sepFilter2D(img_gray, -1, kernel1D, kernel1D)

    display_image('GaussianBlur', img_res_1)
    display_image('Filter2D', img_res_2)
    display_image('sepFilter2D', img_res_3)

    diff_pair12 = abs(img_res_1.astype(np.int16) - img_res_2.astype(np.int16))
    diff_pair13 = abs(img_res_1.astype(np.int16) - img_res_3.astype(np.int16))
    diff_pair23 = abs(img_res_3.astype(np.int16) - img_res_2.astype(np.int16))

    print(
        "Maximal differences:\n\tbetween 1 and 2 pairs is {dp12}\n\tbetween 1 and 3 pairs is {dp13}\n\tbetween 2 and "
        "3 pairs is {dp23}\n".format(dp12=diff_pair12.max(), dp13=diff_pair13.max(), dp23=diff_pair23.max()))


# 5. Read the image bonn.png, convert it into a gray image, and display it. Filter the image
    # • twice with a Gaussian kernel with σ = 2
    # • once with a Gaussian kernel with σ = 2√2
    # and display both results, compute the absolute pixel-wise difference between the
    # results, and print the maximum pixel error

    # set image path
    img_path = 'bonn.png'

    # read and display the image
    img = cv.imread(img_path)

    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    display_image('Original image', img_gray)

    SIGMA1 = 2
    SIGMA2 = 2 * 2 ** 0.5

    ksize = round(SIGMA1 * 3)
    if ksize > 1 and ksize % 2 == 0:
        ksize += 1

    kernel2D_sigma1 = cv.getGaussianKernel(ksize, SIGMA1)

    ksize = round(SIGMA2 * 3)
    if ksize > 1 and ksize % 2 == 0:
        ksize += 1
    kernel2D_sigma2 = cv.getGaussianKernel(ksize, SIGMA2)

    img_res_1 = cv.filter2D(cv.filter2D(img_gray, -1, kernel2D_sigma1), -1, kernel2D_sigma1)
    img_res_2 = cv.filter2D(img_gray, -1, kernel2D_sigma2)

    display_image('Filtered image with sigma = 2', img_res_1)
    display_image('Filtered image with sigma = 2*sqrt(2)', img_res_2)

    diff_pair = abs(img_res_1.astype(np.int16) - img_res_2.astype(np.int16))
    print(f"Maximal difference between pixles is {diff_pair.max()}")

# 7. Read the image bonn.png, convert it into a gray image, add 30% (the chance that
    # a pixel is converted into a black or white pixel is 30%) salt and pepper noise, and
    # display it.
    img = cv.imread(img_path)
    intensity_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    original_integral_img = integral_image(intensity_image)
    original_mean_gray_value = mean_grey_value(original_integral_img, 0, 0, original_integral_img.shape[0], original_integral_img.shape[1])

    salt_paper_img = salt_paper_noise(intensity_image, 0.3)
    display_image('2. Image with salt&paper noise', salt_paper_img)

    filter_size = [1, 3, 5, 7, 9]

    # Blur for Gauss Kernel
    cl_mean_list = []
    for size in filter_size:
        gaus_blur = cv.GaussianBlur(salt_paper_img, (size, size), 0)
        integ_img = integral_image(gaus_blur)
        mean_g_v = mean_grey_value(integ_img, 0, 0, integ_img.shape[0], integ_img.shape[1])
        cl_mean_list.append((mean_g_v, size))

    cl_mean = find_closest_mean_value(original_mean_gray_value, cl_mean_list)
    gaus_blur = cv.GaussianBlur(salt_paper_img, (cl_mean[1], cl_mean[1]), 0)
    display_image("GaussianBlur with the closest mean grey value, kernel size = {}".format(cl_mean[1]), gaus_blur)

    # Median Blur
    cl_mean_list = []
    for size in filter_size:
        median_blur = cv.medianBlur(salt_paper_img, size)
        integ_img = integral_image(median_blur)
        mean_g_v = mean_grey_value(integ_img, 0, 0, integ_img.shape[0], integ_img.shape[1])
        cl_mean_list.append((mean_g_v, size))

    cl_mean = find_closest_mean_value(original_mean_gray_value, cl_mean_list)
    median_blur = cv.medianBlur(salt_paper_img, cl_mean[1])
    display_image("MedianBlur with the closest mean grey value, kernel size = {}".format(cl_mean[1]), median_blur)

    # Bilateral Filter
    cl_mean_list = []
    for size in filter_size:
        bilateral_blur = cv.bilateralFilter(salt_paper_img, d=size, sigmaColor=75, sigmaSpace=75)
        integ_img = integral_image(bilateral_blur)
        mean_g_v = mean_grey_value(integ_img, 0, 0, integ_img.shape[0], integ_img.shape[1])
        cl_mean_list.append((mean_g_v, size))

    cl_mean = find_closest_mean_value(original_mean_gray_value, cl_mean_list)
    bilateral_blur = cv.bilateralFilter(salt_paper_img, d=cl_mean[1], sigmaColor=75, sigmaSpace=75)
    display_image("Bilateral Filter with the closest mean grey value, kernel size = {}".format(cl_mean[1]), bilateral_blur)

# 8. Read the image bonn.png and convert it into a gray image.
    # • Filter the images using the two 2D filter kernels given below
    # • Use the class SVD of OpenCV to separate each kernel. If a kernel is not
    # separable, use an approximation by taking only the highest singular value.
    # Filter the images with the obtained 1D kernels and display the results.
    # • Compute the absolute pixel-wise difference between the results of (a) and (b),
    # and print the maximum pixel error.
    # set image path

    img_path = 'bonn.png'

    # read and display the image
    img = cv.imread(img_path)

    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # display_image('Original image', img_gray)

    kernel1 = np.array([
        [0.0113, 0.0838, 0.0113],
        [0.0838, 0.6193, 0.0838],
        [0.0113, 0.0838, 0.0113],
    ])

    kernel2 = np.array([
        [-0.8984, 0.1472, 1.1410],
        [-1.9075, 0.1566, 2.1359],
        [-0.8659, 0.0573, 1.0337],
    ])

    res_a1 = cv.filter2D(img_gray, -1, kernel1)
    res_b1 = cv.filter2D(img_gray, -1, kernel2)

    display_image('Filtered image with kernel1', res_a1)
    display_image('Filtered image with kernel2', res_b1)

    res_svd1_s, res_svd1_u, res_svd1_v = cv.SVDecomp(kernel1)
    res_svd2_s, res_svd2_u, res_svd2_v = cv.SVDecomp(kernel2)

    X = res_svd1_v[0];
    X /= X.sum()
    Y = res_svd1_u.transpose()[0];
    Y /= Y.sum()
    res_a2 = cv.sepFilter2D(img_gray, -1, X, Y)

    X = res_svd2_v[0];
    X /= X.sum()
    Y = res_svd2_u.transpose()[0];
    Y /= Y.sum()
    res_b2 = cv.sepFilter2D(img_gray, -1, X, Y)

    diff_pair_a = abs(res_a1.astype(np.int16) - res_a2.astype(np.int16))
    diff_pair_b = abs(res_b1.astype(np.int16) - res_b2.astype(np.int16))

    print(
        f"Difference between 2D and separated 1D for kernel1 is {diff_pair_a.max()}\nDifference between 2D and separated 1D for kernel2 is {diff_pair_b.max()}")

    display_image('Filtered image with kernel1', res_a2)
    display_image('Filtered image with kernel2', res_b2)
