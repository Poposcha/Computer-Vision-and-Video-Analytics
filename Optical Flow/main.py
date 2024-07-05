import numpy as np
import os
import time
import cv2 as cv
import matplotlib.pyplot as plt
from prettytable import PrettyTable

from tqdm import tqdm


def load_FLO_file(filename):
    assert os.path.isfile(filename), 'file does not exist: ' + filename   
    flo_file = open(filename, 'rb')
    magic = np.fromfile(flo_file, np.float32, count=1)
    assert magic == 202021.25,  'Magic number incorrect. .flo file is invalid'
    w = np.fromfile(flo_file, np.int32, count=1)
    h = np.fromfile(flo_file, np.int32, count=1)
    #if error try: data = np.fromfile(flo_file, np.float32, count=2*w[0]*h[0])
    #the float values for u and v are interleaved in row order, i.e., u[row0,col0], v[row0,col0], u[row0,col1], v[row0,col1], ...,
    # in total, there are 2*w*h flow values
    data = np.fromfile(flo_file, np.float32, count=2*w[0]*h[0])
    # Reshape data into 3D array (columns, rows, bands)
    flow = np.resize(data, (int(h[0]), int(w[0]), 2))
    flo_file.close()
    return flow


class OpticalFlow:
    def __init__(self):
        self.EIGEN_THRESHOLD = 0.01
        self.WINDOW_SIZE = (25, 25)

        self.EPSILON= 0.002
        self.MAX_ITERS = 1000
        self.ALPHA = 1.0

        self.UNKNOWN_FLOW_THRESH = 1000

        self.prev = None
        self.next = None

    def next_frame(self, img):
        self.prev = self.next
        self.next = img

        if self.prev is None:
            return False

        frames = np.float32(np.array([self.prev, self.next]))
        frames /= 255.0

        #calculate image gradient
        self.Ix = cv.Sobel(frames[0], cv.CV_32F, 1, 0, 3)
        self.Iy = cv.Sobel(frames[0], cv.CV_32F, 0, 1, 3)
        self.It = frames[1]-frames[0]

        return True

  
    def Lucas_Kanade_flow(self):
        #***********************************************************************************
        #implement Lucas-Kanade Optical Flow 
        # Parameters:
        # frames: the two consecutive frames
        # Ix: Image gradient in the x direction
        # Iy: Image gradient in the y direction
        # It: Image gradient with respect to time
        # window_size: the number of points taken in the neighborhood of each pixel
        # returns the Optical flow based on the Lucas-Kanade algorithm

        IxIx_neighbors_sum = self.Ix**2
        IyIy_neighbors_sum = self.Iy**2
        IxIy_neighbors_sum = self.Ix * self.Iy

        IxIt_neighbors_sum = self.Ix * self.It
        IyIt_neighbors_sum = self.Iy * self.It

        # normalize=False to get the sum rather the average
        IxIx_neighbors_sum = cv.boxFilter(IxIx_neighbors_sum, -1, self.WINDOW_SIZE, normalize=False)
        IyIy_neighbors_sum = cv.boxFilter(IyIy_neighbors_sum, -1, self.WINDOW_SIZE, normalize=False)
        IxIy_neighbors_sum = cv.boxFilter(IxIy_neighbors_sum, -1, self.WINDOW_SIZE, normalize=False)
        IxIt_neighbors_sum = cv.boxFilter(IxIt_neighbors_sum, -1, self.WINDOW_SIZE, normalize=False)
        IyIt_neighbors_sum = cv.boxFilter(IyIt_neighbors_sum, -1, self.WINDOW_SIZE, normalize=False)
        h, w = self.next.shape
        flow = np.zeros((h, w, 2))
        for r in range(h):
            for c in range(w):
                A = np.array([[IxIx_neighbors_sum[r, c], IxIy_neighbors_sum[r, c]],
                              [IxIy_neighbors_sum[r, c], IyIy_neighbors_sum[r, c]]])
                b = np.array([[-IxIt_neighbors_sum[r, c]], [-IyIt_neighbors_sum[r, c]]])

                eigen_vals = np.linalg.eigvals(A)
                min_eigen_val = np.min(eigen_vals)
                if min_eigen_val < self.EIGEN_THRESHOLD: # the flow is not valid for this pixel
                    print('invalid flow at this point!!!')
                    print(eigen_vals)
                    continue
                uv_flow = np.linalg.pinv(A) @ b
                flow[r, c] = uv_flow[:, 0]

        flow_bgr = self.flow_map_to_bgr(flow)

        return flow, flow_bgr

    def Horn_Schunck_flow(self):
        """
    implement Horn-Schunck Optical Flow
    Parameters:
        frames: the two consecutive frames
        Ix: Image gradient in the x direction
        Iy: Image gradient in the y direction
        It: Image gradient with respect to time
        alpha: smoothness term, try different values to see its influence

        returns the Optical flow based on the Horn-Schunck algorithm
        """

        denom = self.ALPHA**2+self.Ix**2+self.Iy**2
        u = np.zeros((self.Ix.shape[0], self.Ix.shape[1]) )
        v = np.zeros((self.Ix.shape[0], self.Ix.shape[1]) )
        flow = np.zeros((self.Ix.shape[0], self.Ix.shape[1], 2) )
        diff = 1
        iter = 0
        while diff > self.EPSILON and iter < self.MAX_ITERS:
            u_laplace = cv.Laplacian(u, -1, ksize=1, scale=0.25)
            v_laplace = cv.Laplacian(v, -1, ksize=1, scale=0.25)
            u_mean = u + u_laplace
            v_mean = v + v_laplace
            mult_term = (self.Ix*u_mean + self.Iy*v_mean + self.It)/denom
            prev_u = u.copy()
            prev_v = v.copy()
            u = u_mean - self.Ix * mult_term
            v = v_mean - self.Iy * mult_term
            diff = cv.norm(prev_u, u, normType=cv.NORM_L2) + cv.norm(prev_v, v, normType=cv.NORM_L2)
            iter += 1
        flow[:,:,0] = u
        flow[:,:,1] = v

        flow_bgr = self.flow_map_to_bgr(flow)
        return flow, flow_bgr

    def calculate_angular_error(self, estimated_flow, groundtruth_flow):
        u, v = estimated_flow[..., 0], estimated_flow[..., 1]
        u_gt, v_gt = groundtruth_flow[..., 0], groundtruth_flow[..., 1]

        numerator = u * u_gt + v * v_gt + 1
        denominator = np.sqrt((u**2 + v**2 + 1) * (u_gt**2 + v_gt**2 + 1))
        aae_per_point = numerator / denominator
        aae = np.mean(np.arccos(aae_per_point))
        return np.rad2deg(aae), np.rad2deg(aae_per_point)

    def calculate_endpoint_error(self, estimated_flow, groundtruth_flow):
      aee_per_point = (groundtruth_flow[:,:,0]-estimated_flow[:,:,0])**2+(groundtruth_flow[:,:,1]-estimated_flow[:,:,1])**2
      aee = np.sum(aee_per_point) / aee_per_point.size
      return aee, aee_per_point


    #function for converting flow map to to BGR image for visualisation
    # the optical flow is color coded based on direction on magnitude, color is determined by flow angle, intensity of color is determined by magnitude
    def flow_map_to_bgr(self, flow, BGR=True):
        # u, v = flow[:,:, 0], flow[:,:, 1]
        flow = np.clip(flow, -15, 15)

        hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype = np.uint8)
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
        hsv[..., 0] = (ang / 2).astype(np.uint8)
        hsv[..., 1] = (cv.normalize(mag, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)).astype(np.uint8)
        hsv[..., 2] = 255
        img = []
        if BGR:
            img = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        else:
            img[0] = cv.cvtColor(u, cv.COLOR_HSV2RGB)
            img[1] = cv.cvtColor(v, cv.COLOR_HSV2RGB)
        return img
  

if __name__ == "__main__":

    data_list = [
        'frame_0001.png',
        'frame_0002.png',
        'frame_0007.png',
    ]

    gt_list = [
        'frame_0001.flo',
        'frame_0002.flo',
        'frame_0007.flo',
    ]

    Op = OpticalFlow()

    tab = PrettyTable()
    tab.field_names = ["Run", "AAE_lucas_kanade", "AEE_lucas_kanade", "AAE_horn_schunk", "AEE_horn_schunk"]


    results = []
    count = 0
    for (i, (frame_filename, gt_filemane)) in tqdm(enumerate(zip(data_list, gt_list)), total=len(data_list)):
        groundtruth_flow = load_FLO_file(os.path.join("./data", gt_filemane))
        img_rgb = cv.cvtColor(cv.imread(os.path.join("./data", frame_filename)), cv.COLOR_BGR2RGB)
        img = cv.cvtColor(img_rgb, cv.COLOR_RGB2GRAY)
        if not Op.next_frame(img):
            continue

        flow_lucas_kanade, flow_lucas_kanade_bgr = Op.Lucas_Kanade_flow()
        aae_lucas_kanade, aae_lucas_kanade_per_point = Op.calculate_angular_error(flow_lucas_kanade, groundtruth_flow)
        aee_lucas_kanade, aee_lucas_kanade_per_point = Op.calculate_endpoint_error(flow_lucas_kanade, groundtruth_flow)

        flow_horn_schunck, flow_horn_schunck_bgr = Op.Horn_Schunck_flow()
        aae_horn_schunk, aae_horn_schunk_per_point = Op.calculate_angular_error(flow_horn_schunck, groundtruth_flow) 
        aee_horn_schunk, aee_horn_schunk_per_point = Op.calculate_endpoint_error(flow_horn_schunck, groundtruth_flow)
        count += 1

        tab.add_row([count,
                     np.round(aae_lucas_kanade, decimals=2),
                     np.round(aee_lucas_kanade, decimals=2),
                     np.round(aae_horn_schunk, decimals=2),
                     np.round(aee_horn_schunk, decimals=2)])

        flow_bgr_gt = Op.flow_map_to_bgr(groundtruth_flow)

        results.append({
            "Flow GT": flow_bgr_gt,
            "Lucas Kanade": flow_lucas_kanade_bgr,
            "AAE Lucas Kanade": aae_lucas_kanade_per_point,
            "AEE Luca Kanade": aee_lucas_kanade_per_point,
            "Horn-Schunk": flow_horn_schunck_bgr,
            "AAE Horn-Schunk": aae_horn_schunk_per_point,
            "AEE Horn-Schunk": aee_horn_schunk_per_point,
        })

    fig, axes = plt.subplots(nrows=len(results),
                             ncols=7,
                             figsize=(30, 5))
    for r, res in enumerate(results):
        for c, (name, value) in enumerate(res.items()):
            ax = axes[r][c]
            ax.imshow(value)
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_title(name)
    fig.subplots_adjust(wspace=0.00, hspace=0.0, left=0, right=1.0, top=1.0, bottom=0)
    plt.savefig("results.png")
    plt.show()
    print(tab)
