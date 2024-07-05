import numpy as np
import cv2 as cv
import random
import matplotlib.pyplot as plt


def task_1_a():
    print("Task 1 (a) ...")
    # img = cv.imread('../images/shapes.png', cv.IMREAD_GRAYSCALE)
    img = cv.imread('../images/shapes.png', cv.IMREAD_GRAYSCALE)

    t1, t2 = 50, 200
    edges = cv.Canny(img, t1, t2)
    coord_lines = cv.HoughLines(edges, 0.8, 2*np.pi / 180, t1)

    img_copy = img.copy()
    img_copy = cv.cvtColor(img_copy, cv.COLOR_GRAY2BGR)

    for line_point in coord_lines:
        ro = line_point[0][0]
        theta = line_point[0][1]

        x = round(ro*np.cos(theta))
        y = round(ro*np.sin(theta))
        
        cos_phi = np.cos(theta)
        sin_phi = np.sin(theta)

        # Inspired on openCV documentation: https://docs.opencv.org/4.x/d9/db0/tutorial_hough_lines.html
        pt1 = (int(x + 1000*(-sin_phi)), int(y + 1000*(cos_phi)))
        pt2 = (int(x - 1000*(-sin_phi)), int(y - 1000*(cos_phi)))

        cv.line(img_copy, pt1, pt2, (255, 0, 0), 1, cv.LINE_AA)

    return img_copy


def myHoughLines(img_edges, d_resolution, theta_step_sz, threshold):
    """
    Your implementation of HoughLines
    :param img_edges: single-channel binary source image (e.g: edges)
    :param d_resolution: the resolution for the distance parameter
    :param theta_step_sz: the resolution for the angle parameter
    :param threshold: minimum number of votes to consider a detection
    :return: list of detected lines as (d, theta) pairs and the accumulator
    """
    accumulator = np.zeros((round(180 / theta_step_sz), round(np.linalg.norm(img_edges.shape) / d_resolution)*2+1), dtype=np.int16)
    detected_lines = []
    
    # To avoid case where d is negative we used hints from 
    # Basic Hough Transform Algorithm by Udacity (accses only via link)
    # https://www.youtube.com/watch?v=2oGYGXJfjzw
    dmax = round(np.linalg.norm(img_edges.shape) / d_resolution)
    ds = np.arange(-dmax, dmax+1, 1)

    for y in range(img_edges.shape[0]):
        for x in range(img_edges.shape[1]):
            if img_edges[y][x] != 0:                
                theta = 0
                while theta < round(180/theta_step_sz):
                    theta_r = theta * np.pi / 180
                    d = round(x*np.cos(theta_r) + y*np.sin(theta_r))
                    accumulator[theta, np.argmin(np.abs(ds - d))] += 1
                    theta += 1
                    
    for theta in range(accumulator.shape[0]):
        for d in range(accumulator.shape[1]):
            if accumulator[theta][d] >= threshold:
                detected_lines.append((abs(ds[d]), theta * np.pi / 180))


    return np.array(detected_lines), accumulator


def draw_Hough_lines_on_image(img3Colors, detected_lines):
    img_copy = img3Colors.copy()
    for line_point in detected_lines:
        ro = line_point[0]
        theta = line_point[1]

        cos_phi = np.cos(theta)
        sin_phi = np.sin(theta)
        x = round(ro*cos_phi)
        y = round(ro*sin_phi)

        # Inspired on openCV documentation: https://docs.opencv.org/4.x/d9/db0/tutorial_hough_lines.html
        pt1 = (int(x + 1000*(-sin_phi)), int(y + 1000*(cos_phi)))
        pt2 = (int(x - 1000*(-sin_phi)), int(y - 1000*(cos_phi)))

        cv.line(img_copy, pt1, pt2, (255, 0, 0), 1, cv.LINE_AA)
    
    return img_copy


def task_1_b():
    print("Task 1 (b) ...")
    img = cv.imread('../images/shapes.png')
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # convert the image into grayscale
    t1, t2 = 59, 200
    edges = cv.Canny(img_gray, t1, t2) # detect the edges
    d_resolution, theta_sep_sz = 0.5, 1.5

    detected_lines, accumulator = myHoughLines(edges, d_resolution, theta_sep_sz, t1)
    
    img_copy = draw_Hough_lines_on_image(img, detected_lines)
    
    return img_copy




def myKmeans(data, k):
    """
    Your implementation of k-means algorithm
    :param data: list of data points to cluster
    :param k: number of clusters
    :return: centers and list of indices that store the cluster index for each data point
    """
    centers = np.zeros((k, data.shape[1]), np.float16)
    index = np.zeros(data.shape[0], dtype=int)

    # initialize centers using some random points from data
    for ind in range(len(centers)):
        rand_point_ind = np.random.randint(0, data.shape[0])
        centers[ind] = data[rand_point_ind].copy()
    
    convergence = False
    iterationNo = 0

    while not convergence:
        # assign each point to the cluster of closest center
        for i in range(len(index)):
            index[i] = np.argmin(np.linalg.norm(np.abs(centers - data[i]), axis=1))

        # update clusters' centers and check for convergence
        not_updated_centers = np.zeros(k, dtype=bool)
        for i in range(k):
            indeces_center_i = index==i
            new_center = data[indeces_center_i].copy().mean(axis=0)
            
            if np.abs(new_center-centers[i]).mean() > 1e-1:
                centers[i] = new_center.copy()
            else:
                not_updated_centers[i] = True

        if not_updated_centers.any():
            convergence = True
            
        iterationNo += 1
        print('iterationNo = ', iterationNo)

    return index, centers


def task_3_a(k=5):
    print("Task 3 (a) ...")
    img = cv.imread('../images/flower.png')
    data = cv.cvtColor(img, cv.COLOR_BGR2GRAY).reshape((-1, 1))


    ind, clusts = myKmeans(data, k)
    return ind.reshape(img.shape[:2])


def task_3_b(k=5):
    print("Task 3 (b) ...")
    img = cv.imread('../images/flower.png')
    
    data = img.copy().reshape(-1, 3)

    ind, clusts = myKmeans(data, k)

    return ind.reshape(img.shape[:2])


def task_3_c(k=5):
    print("Task 3 (c) ...")
    img = cv.imread('../images/flower.png')
    img_grey = cv.cvtColor(img , cv.COLOR_BGR2GRAY)

    intensity_data = img_grey.reshape((-1,1))

    # Source: documentation
    # https://numpy.org/doc/stable/reference/generated/numpy.argwhere.html
    pos_data = np.argwhere(img_grey) / (np.array(img_grey.shape) * 5)

    # Source: documentation
    # https://numpy.org/doc/stable/reference/generated/numpy.hstack.html
    data = np.hstack((intensity_data, pos_data))

    ind, clusts = myKmeans(data, k)

    return ind.reshape(img.shape[:2])


def get_derivative_of_gaussian_kernel(size, sigma):
    ker_x = cv.getGaussianKernel(size, sigma)
    ker_y = cv.getGaussianKernel(size, sigma).transpose()

    ker_gauss = ker_x @ ker_y
    gradient = np.gradient(ker_gauss)

    return gradient


def task_3_d(k=5):
    print("Task 3 (d) ...")
    img = cv.imread('../images/flower.png')
    img_grey = cv.cvtColor(img , cv.COLOR_BGR2GRAY)

    color_data = img.reshape((-1,3))

    # Source: documentation
    # https://numpy.org/doc/stable/reference/generated/numpy.argwhere.html
    pos_data = np.argwhere(img_grey) / (np.array(img_grey.shape) * 10)

    kernel_x, kernel_y = get_derivative_of_gaussian_kernel(7, 2)

    edges_x = cv.filter2D(img_grey, -1, kernel_x)  
    edges_y = cv.filter2D(img_grey, -1, kernel_y) 

    magnitude = np.sqrt(np.square(edges_x) + np.square(edges_y)).reshape((-1, 1))

    # Source: documentation
    # https://numpy.org/doc/stable/reference/generated/numpy.hstack.html
    data = np.hstack((color_data, pos_data, magnitude))

    ind, clusts = myKmeans(data, k)
    return ind.reshape(img.shape[:2])


def color_per_cluster(img, clusters):
    result = []
    for i in range(np.max(clusters)+1):
        mask = clusters == i
        colors = img[mask]
        result.append(list(map(int, np.mean(colors, axis=0))))
    return result

def change_cluster_to_color(cluster, colors):
    cluster = cluster.tolist()
    for i in range(len(cluster)):
        for j in range(len(cluster[0])):
            cluster[i][j] = colors[cluster[i][j]]
    return np.array(cluster)

def K(x):
    norm_x = np.linalg.norm(x)
    if norm_x <= 1:
        # return -2*norm_x
        return 1 - norm_x**2
    else:
        return 0
    
    
def meanShift(data, window_size, kernel, x1, y1):
    """
    Implementation of mean shift algorithm
    :param data: data points to cluster
    :param window_size: the size of the window is 2 * window_size + 1 
    :param kernel: the chosen kernel (slide 74)
    :param x1, y1: original position (x1, y1)
    :return: shifted position (x2, y2) and sum of weights within the window
    """
    
    center = np.array([x1*343 + y1, x1, y1], dtype=np.float32)
    h = window_size
    # data = data / data.max()*255

    # g = lambda x: K(x)
    g = lambda x: -np.linalg.norm(x)*np.exp(-0.5*np.linalg.norm(x)**2)
    # g = lambda x: np.exp(-0.5*np.linalg.norm(x)**2/h/h)
    # g = lambda x: 1 if np.linalg.norm(x) <= window_size else 0
    # g = lambda x: -2*np.linalg.norm(x) if -2*np.linalg.norm(x) <= 1 else 0 

    convergence = False
    iterationNo = 0
    new_center = center.copy()
    while not convergence:
        # center = new_center.copy().astype(np.int16)
        print(center)

        sum_numerator = np.zeros(data.shape[1], dtype=np.float64)
        sum_denumirator = 0.
        
        center_int = center.copy().astype(np.int32)
        indeces = []
        for x in range(343):
            for y in range(180):
                if (x - center[1])**2 + (y - center[2])**2 <= h*h:
                    indeces.append(x*343+y)
        indeces = np.array(indeces)

        for pt in indeces:
            value = g( (data[pt] - data[center_int])/h )
            # TODO We need help here. It doesn't work besides we tried thousands times 
            # to change anything. So we would love to hear explanation and after that 
            # change solution 'case we spent 3 days for that function and we can't spend more
            sum_numerator += data[pt]*value
            sum_denumirator += value

        print("sum_numerator: {}\nsum_denumirator: {}\ncenter: {}\ndiv: {}\n".format(sum_numerator, sum_denumirator, center, sum_numerator/sum_denumirator))

        if abs(sum_numerator).sum() > 1e-0:
            center = new_center.copy().astype(np.float16)
            new_center = center + sum_numerator/sum_denumirator
        else: 
            convergence = True
        
        print("new_center: {}\ncenter: {}\nnew_center - center: {}\nabs(new_center - center).sum(): {}\n".format(new_center, center, new_center - center, abs(new_center - center).sum()))

        # + np.random.random_integers(-10, 10, pts.shape[1])/10*(window_size*0.1)

        iterationNo += 1
        print("-"*30, '\niterationNo = ', iterationNo, sep='')
    return new_center


def toy_example():
    # a = np.random.random_integers(0, 5, 20).reshape(-1, 2)
    a = np.array([
        [4, 1],
        [4, 2],
        [4, 0],
        [2, 2],
        [1, 5],
        [3, 0],
        [1, 3],
        [1, 3],
        [2, 1],
        [0, 1],
        [6, 1],
        [6, 3],
    ])
    x1, y1 = 5, 3
    # window_size = 2.5
    window_size = 3

    center = np.array([x1, y1])

    def pts(data, x1, y1, window_size):
        x1, y1 = 5, 2
        shifted_points = data-np.array([x1, y1])
        window_size = 2
        d_pnts2cntr = np.linalg.norm(shifted_points, axis=1)

        points_in_window_indeces = np.where(d_pnts2cntr<window_size)
        points_in_window = data[points_in_window_indeces]

        return points_in_window_indeces, points_in_window

    points_in_window_indeces, pts = pts(a, x1, y1, window_size)

    circl = plt.Circle((x1, y1), window_size, fill=False)
    ax = plt.gca()
    ax.cla()
    ax.plot(a.transpose()[0], a.transpose()[1], 'o', color='red')
    ax.plot(x1, y1, 'o', color='blue')

    g = lambda x: K(x)
    # g = lambda x: -np.linalg.norm(x)*np.exp(-0.5*np.linalg.norm(x)**2)
    # g = lambda x: np.exp(-0.5*np.linalg.norm(x)**2/h/h)
    # g = lambda x: 1 if np.linalg.norm(x) <= window_size else 0
    # g = lambda x: -2*np.linalg.norm(x) if -2*np.linalg.norm(x) <= 1 else 0 


    convergence = False
    iterationNo = 0
    new_center = center.copy()
    while not convergence:
        center = new_center.copy()
        sum_numerator = 0.
        sum_denumirator = 0.
        for i in range(pts.shape[0]):
            value = g((center-pts[i])/window_size)
            sum_numerator += pts[i]*value
            sum_denumirator += value
        # print(sum_numerator, sum_denumirator)

        if np.linalg.norm(abs(sum_numerator/sum_denumirator - center)) < 1e-1:
            convergence = True

        new_center = sum_numerator/ sum_denumirator + np.random.random_integers(-10, 10, pts.shape[1])/10*(window_size*0.1)
        ax.plot(*new_center, 'o', color='orange')
        line = plt.Line2D((center[0], new_center[0]), (center[1], new_center[1]))
        ax.add_line(line)

        iterationNo += 1
        print('iterationNo = ', iterationNo)        

    # shifted_points, d_pnts2cntr, points_in_window


    pts, ax.add_patch(circl)
    circl = plt.Circle(new_center, window_size, fill=False, linestyle='--')
    pts, ax.add_patch(circl)

    ax.set_xlim(0, 10)
    ax.set_ylim(-2, 6)
    # ax.save('toy_example.jpg')
    plt.savefig('toy_example.jpg')


if __name__ == "__main__":
    res_task_1a = task_1_a()
    plt.imshow(res_task_1a)
    plt.savefig('res_1_a.jpg')
    plt.clf()

    res_task_1b = task_1_b()
    plt.imshow(res_task_1b)
    plt.savefig('res_1_b.jpg')
    plt.clf()
    
    img = cv.imread('../images/flower.png')
    resA5, resB5, resC5, resD5 = task_3_a(), task_3_b(), task_3_c(), task_3_d()
    resA10, resB10, resC10, resD10 = task_3_a(10), task_3_b(10), task_3_c(10), task_3_d(10)

    # Mean color for all points with the same cluster
    cluster_colorsA5 = color_per_cluster(img, resA5)
    colored_clusterA5 = change_cluster_to_color(resA5, cluster_colorsA5)

    cluster_colorsB5 = color_per_cluster(img, resB5)
    colored_clusterB5 = change_cluster_to_color(resB5, cluster_colorsB5)

    cluster_colorsC5 = color_per_cluster(img, resC5)
    colored_clusterC5 = change_cluster_to_color(resC5, cluster_colorsC5)

    cluster_colorsD5 = color_per_cluster(img, resD5)
    colored_clusterD5 = change_cluster_to_color(resD5, cluster_colorsD5)

    cluster_colorsA10 = color_per_cluster(img, resA10)
    colored_clusterA10 = change_cluster_to_color(resA10, cluster_colorsA10)

    cluster_colorsB10 = color_per_cluster(img, resB10)
    colored_clusterB10 = change_cluster_to_color(resB10, cluster_colorsB10)

    cluster_colorsC10 = color_per_cluster(img, resC10)
    colored_clusterC10 = change_cluster_to_color(resC10, cluster_colorsC10)

    cluster_colorsD10 = color_per_cluster(img, resD10)
    colored_clusterD10 = change_cluster_to_color(resD10, cluster_colorsD10)

    plt.title("Colored clustered image of 10 clusters")
    plt.subplot(221)
    plt.imshow(colored_clusterA10)
    plt.subplot(222)
    plt.imshow(colored_clusterB10)
    plt.subplot(223)
    plt.imshow(colored_clusterC10)
    plt.subplot(224)
    plt.imshow(colored_clusterD10)
    plt.savefig('colored_10_clust.jpg')
    plt.clf()

    plt.title("Colored clustered image of 5 clusters")
    plt.subplot(221)
    plt.imshow(colored_clusterA5)
    plt.subplot(222)
    plt.imshow(colored_clusterB5)
    plt.subplot(223)
    plt.imshow(colored_clusterC5)
    plt.subplot(224)
    plt.imshow(colored_clusterD5);
    plt.savefig('colored_5_clust.jpg')
    plt.clf()

    plt.title("Clustered image of 5 clusters")
    plt.subplot(221)
    plt.imshow(resA5, cmap="gray")
    plt.subplot(222)
    plt.imshow(resB5, cmap="gray")
    plt.subplot(223)
    plt.imshow(resC5, cmap="gray")
    plt.subplot(224)
    plt.imshow(resD5, cmap="gray")
    plt.savefig('non_colored_5_clust.jpg')
    plt.clf()

    plt.title("Clustered image of 10 clusters")
    plt.subplot(221)
    plt.imshow(resA10, cmap="gray")
    plt.subplot(222)
    plt.imshow(resB10, cmap="gray")
    plt.subplot(223)
    plt.imshow(resC10, cmap="gray")
    plt.subplot(224)
    plt.imshow(resD10, cmap="gray")
    plt.savefig('non_colored_10_clust.jpg')
    plt.clf()


    toy_example()
