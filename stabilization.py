import os
import numpy as np
import cv2
from glob import glob
import matplotlib.pyplot as plt
import cvxpy as cp


# A5 - Video Texture
def video2frames(video_name):
    video_path = os.path.join("source", video_name.replace('.', '_'))

    exist = os.path.exists(video_path)
    if not exist:
        # Create folder for frame
        os.makedirs(video_path)
        print("Directory '% s' created" % video_path)

    # video to frames
    # ffmpeg -i video.mov video/%04d.png -hide_banner
    cmd = "ffmpeg -i " + video_name + " " + video_path + "/%04d.png -hide_banner"
    print(cmd)
    os.system(cmd)

    return video_path


# https://stackoverflow.com/questions/24961127/how-to-create-a-video-from-images-with-ffmpeg
def frames2video(input_path):
    output_path = os.path.join(input_path, 'result.mp4')

    # frames to video
    # ffmpeg -r 30 -start_number 1 -i "rect/frame%04d.png" -c:v libx264 -vf "fps=30,format=yuv420p, scale=640:-2" stable.mp4
    cmd = 'ffmpeg -r 30 -start_number 1 -i ' + input_path +  '/frame%04d.png -c:v libx264 -vf "fps=30,format=yuv420p, scale=640:-2" ' + output_path
    print(cmd)
    os.system(cmd)

    return output_path


# This function is from A5-Video_Textures main.py
def readImages(image_dir):
    extensions = ['bmp', 'pbm', 'pgm', 'ppm', 'sr', 'ras', 'jpeg',
                  'jpg', 'jpe', 'jp2', 'tiff', 'tif', 'png']

    search_paths = [os.path.join(image_dir, '*.' + ext) for ext in extensions]
    image_files = sorted(sum(map(glob, search_paths), []))
    images = [cv2.imread(f, cv2.IMREAD_UNCHANGED | cv2.IMREAD_COLOR) for f in image_files]

    bad_read = any([img is None for img in images])
    if bad_read:
        raise RuntimeError(
            "Reading one or more files in {} failed - aborting."
                .format(image_dir))

    return images


# Module 4 and 5 Demo Code - feature_detection and feature matching
# https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
def extractFeatures(frames):
    n = len(frames)
    features = []

    for i in range(n):
        # Initiate SIFT detector
        # orb = cv2.SIFT_create()   # slow
        orb = cv2.ORB_create()
        img_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        # find the keypoints and descriptors with SIFT
        kp, des = orb.detectAndCompute(img_gray, None)

        if np.array(kp).shape[0] == 0:
            print("frame: " + str(i) + " has no keypoints")

        features.append((kp, des))

    return features


# panorama.py (A4), feature_matching.py and warp_affine.py (Module 4 and 5 Demo Code)
# https://docs.opencv.org/master/d1/de0/tutorial_py_feature_homography.html
def findCameraPath(frames, features):
    n = len(frames)
    path = []

    # Create BFMatcher (Brute Force Matcher) object
    bf = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=False)
    # bf = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=False)

    for i in range(n - 1):
        kp1, des1 = features[i][0], features[i][1]
        kp2, des2 = features[i + 1][0], features[i + 1][1]
        # Match descriptors
        matches = bf.match(des1, des2)
        # get the matching key points for each of the frames
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pst2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Computes an optimal affine transformation between two point sets
        mat = cv2.estimateAffine2D(pts1, pst2, method=cv2.RANSAC)[0]
        # Affine => | a b c |
        # (3 x 3)   | d e f |
        #           | 0 0 1 |
        mat = np.vstack([mat, [0, 0, 1]])
        path.append(mat)

    return path


# https://publik.tuwien.ac.at/files/publik_274527.pdf (pg 16 - 19)
# [Grundmann, Kwatra, and Essa (2011)] Algorithm: Summarized LP for the optimal camera path
# https://www.cc.gatech.edu/cpl/projects/videostabilization/stabilization.pdf
def findOptimalPath(original_path, frame_size, crop_ratio=1):
    n = len(original_path)

    # Weights of the objective eq (see Figure 8(d) from paper)
    w1, w2, w3 = 10, 1, 100

    # Use weighting of 100:1 for affine to translational parts
    affine_weights = np.transpose([1, 1, 100, 100, 100, 100])

    # slack variables and constraint list
    pt = cp.Variable((n, 6))    # optimized path
    e1 = cp.Variable((n, 6))
    e2 = cp.Variable((n, 6))
    e3 = cp.Variable((n, 6))
    constraints = []

    # Objective Eq (3)
    objective = cp.Minimize(cp.sum(w1 * e1 @ affine_weights + w2 * e2 @ affine_weights + w3 * e3 @ affine_weights))

    # 1. Smoothness constraints
    for i in range(n - 3):
        # Pt is the optimized path
        # pt = (dx, dy, a, b, c, d).T
        #       0    1  2  3  4  5
        # Bt is the transformation between old and new path
        # Bt = | a  c  0 |
        #      | b  d  0 |
        #      | dx dy 1 |
        Bt = np.array([[pt[i, 2], pt[i, 4], 0],
                       [pt[i, 3], pt[i, 5], 0],
                       [pt[i, 0], pt[i, 1], 1]])

        Bt1 = np.array([[pt[i + 1, 2], pt[i + 1, 4], 0],
                        [pt[i + 1, 3], pt[i + 1, 5], 0],
                        [pt[i + 1, 0], pt[i + 1, 1], 1]])

        Bt2 = np.array([[pt[i + 2, 2], pt[i + 2, 4], 0],
                        [pt[i + 2, 3], pt[i + 2, 5], 0],
                        [pt[i + 2, 0], pt[i + 2, 1], 1]])

        Bt3 = np.array([[pt[i + 3, 2], pt[i + 3, 4], 0],
                        [pt[i + 3, 3], pt[i + 3, 5], 0],
                        [pt[i + 3, 0], pt[i + 3, 1], 1]])

        # Eq (4) Rt = Ft+1 @ Bt+1 - Bt
        # original_path = | a b dx |     original_path.T = | a  c  0 |
        #                 | c d dy |                       | b  d  0 |
        #                 | 0 0  1 |                       | dx dy 1 |
        Rt = np.transpose(original_path[i + 1]) @ Bt1 - Bt
        Rt1 = np.transpose(original_path[i + 2]) @ Bt2 - Bt1
        Rt2 = np.transpose(original_path[i + 3]) @ Bt3 - Bt2

        # rearrange Residuals [dx, dy, a, b, c, d]
        Rt = np.array([Rt[2, 0], Rt[2, 1], Rt[0, 0], Rt[1, 0], Rt[0, 1], Rt[1, 1]])
        Rt1 = np.array([Rt1[2, 0], Rt1[2, 1], Rt1[0, 0], Rt1[1, 0], Rt1[0, 1], Rt1[1, 1]])
        Rt2 = np.array([Rt2[2, 0], Rt2[2, 1], Rt2[0, 0], Rt2[1, 0], Rt2[0, 1], Rt2[1, 1]])

        # -e1 <= Rt(p) <= e1
        # -e2 <= Rt1(p) - Rt(p) <= e2
        # -e3 <= Rt2(p) -2*Rt1(p) - Rt(p) <= e3
        for j in range(6):
            constraints.append(-e1[i, j] <= Rt[j])
            constraints.append(Rt[j] <= e1[i, j])

            constraints.append(-e2[i, j] <= Rt1[j] - Rt[j])
            constraints.append(Rt1[j] - Rt[j] <= e2[i, j])

            constraints.append(-e3[i, j] <= Rt2[j] - 2 * Rt1[j] + Rt[j])
            constraints.append(Rt2[j] - 2 * Rt1[j] + Rt[j] <= e3[i, j])

    # e >= 0
    constraints.append(e1 >= 0)
    constraints.append(e2 >= 0)
    constraints.append(e3 >= 0)

    # 2. Proximity Constraints
    # 0.9 <= a, d <= 1.1
    # -0.1 <= b, c <= 0.1
    # -0.05 <= b + c <= 0.05
    # -0.1 <= a - d <= 0.1
    lb = np.array([0.9, -0.1, -0.1, 0.9, -0.05, -0.1])
    ub = np.array([1.1, 0.1, 0.1, 1.1, 0.05, 0.1])
    # U =        a b c d b+c a-d
    #       tx | 0 0 0 0  0   0 |
    #       ty | 0 0 0 0  0   0 |
    #       a  | 1 0 0 0  0   1 |
    #       b  | 0 1 0 0  1   0 |
    #       c  | 0 0 1 0  1   0 |
    #       d  | 0 0 0 1  0  -1 |
    U = np.array([[0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [1, 0, 0, 0, 0, 1],
                  [0, 1, 0, 0, 1, 0],
                  [0, 0, 1, 0, 1, 0],
                  [0, 0, 0, 1, 0, -1]])
    # Eq (7) lb <= U @ pt <= ub
    for i in range(n):
        constraints.append(lb <= pt[i, :] @ U)
        constraints.append(pt[i, :] @ U <= ub)

    # 3. Inclusion Constraints
    # Eq (8)
    # | 0 | <= | 1 0 cx cy 0  0  | <= | w |
    # | 0 |    | 0 1 0  0  cx cy |    | h |
    h, w = frame_size[:2]
    center_w, center_h = w // 2, h // 2
    dw, dh = crop_ratio * center_w, crop_ratio * center_h

    # corners
    top_left = (center_w - dw, center_h - dh)
    top_right = (center_w + dw, center_h - dh)
    bottom_left = (center_w - dw, center_h + dh)
    bottom_right = (center_w + dw, center_h + dh)

    crop_frame = [top_left, top_right, bottom_right, bottom_left]
    for i in range(len(crop_frame)):
        cx, cy = crop_frame[i]
        constraints.append(0 <= pt @ np.transpose([1, 0, cx, cy, 0, 0]))
        constraints.append(pt @ np.transpose([1, 0, cx, cy, 0, 0]) <= w)

        constraints.append(0 <= pt @ np.transpose([0, 1, 0, 0, cx, cy]))
        constraints.append(pt @ np.transpose([0, 1, 0, 0, cx, cy]) <= h)

    print('Running ECOS solver ...')
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.ECOS)
    # print("optimal value", problem.value)
    # print(pt.value)

    # Convert from parametric to matrix form
    # | dx dy a b c d | => | a b dx |
    #   0  1  2 3 4 5      | c d dy |
    #                      | 0 0 1  |
    optimal_path = []
    for i in range(n):
        mat = np.array([[pt.value[i, 2], pt.value[i, 3], pt.value[i, 0]],
                        [pt.value[i, 4], pt.value[i, 5], pt.value[i, 1]],
                        [0, 0, 1]])
        optimal_path.append(mat)

    return optimal_path


# https://stackoverflow.com/questions/46795669/python-opencv-how-to-draw-ractangle-center-of-image-and-crop-image-inside-rectan/46803516
def drawRectangle(img, crop_ratio=0.6):
    h, w = img.shape[:2]

    center_w, center_h = w // 2, h // 2
    dw, dh = crop_ratio * w, crop_ratio * h

    upper_left = (int(center_w - dw // 2), int(center_h - dh // 2))
    bottom_right = (int(center_w + dw // 2), int(center_h + dh // 2))

    crop = img[int(center_h - dh // 2 + 1):int(center_h + dh // 2), int(center_w - dw // 2 + 1):int(center_w + dw // 2)]
    rect = cv2.rectangle(img, upper_left, bottom_right, (0, 225, 0), thickness=1)

    # cv2.imshow('crop', crop)
    # cv2.imshow('rect', rect)
    # cv2.waitKey()

    return rect, crop


# panorama.py (A4)
# https://www.pyimagesearch.com/2014/05/05/building-pokedex-python-opencv-perspective-warping-step-5-6/
def applyStabilization(filename, original_frames, optimal_path, crop_ratio=1):
    n = len(original_frames)
    h, w = original_frames[0].shape[:2]

    # Create folder for the output frames
    rect_output_path = os.path.join("output", filename, "rect")
    crop_output_path = os.path.join("output", filename, "crop")

    exist = os.path.exists(rect_output_path)
    if not exist:
        os.makedirs(rect_output_path)
        print("Directory '% s' created" % rect_output_path)

    exist = os.path.exists(crop_output_path)
    if not exist:
        os.makedirs(crop_output_path)
        print("Directory '% s' created" % crop_output_path)

    # rect_frames = []
    # crop_frames = []
    for i in range(1, n):
        # apply transformation by calling cv2.warpPerspective
        warp_frame = cv2.warpPerspective(original_frames[i], optimal_path[i - 1], (w, h))
        rect, crop = drawRectangle(warp_frame, crop_ratio)
        frame = 'frame{0:04d}.png'.format(i)
        cv2.imwrite(rect_output_path + '/' + frame, rect)
        cv2.imwrite(crop_output_path + '/' + frame, crop)

        # rect_frames.append(rect)
        # crop_frames.append(crop)

    return rect_output_path, crop_output_path


# https://www.geeksforgeeks.org/graph-plotting-in-python-set-1/
# https://matplotlib.org/3.1.0/gallery/lines_bars_and_markers/nan_test.html#sphx-glr-gallery-lines-bars-and-markers-nan-test-py
def graphPaths(video_name, original_path, optimize_path):

    # number of frames
    n = len(original_path)

    original_x = np.zeros(n)
    original_y = np.zeros(n)
    optimal_x = np.zeros(n)
    optimal_y = np.zeros(n)

    #              x  y  w
    pt = np.array([1, 1, 1])

    # push point through the path and collect the result
    for i in range(n):
        pt = np.dot(original_path[i], pt)
        original_x[i] = pt[0]
        original_y[i] = pt[1]

        optimal_pt = np.dot(optimize_path[i], pt)
        optimal_x[i] = optimal_pt[0]
        optimal_y[i] = optimal_pt[1]

    # place data on the subplots
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(0, n), original_x, '-r', label="original path")
    plt.plot(np.arange(0, n), optimal_x, '-b', label="optimal L1 path")
    plt.title('Motion in x over frames')
    plt.xlabel('Frames')
    plt.ylabel('Motion')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(np.arange(0, n), original_y, '-r', label="original path")
    plt.plot(np.arange(0, n), optimal_y, '-b', label="optimal L1 path")
    plt.title('Motion in y over frames')
    plt.xlabel('Frames')
    plt.ylabel('Motion')
    plt.legend()

    plt.tight_layout()
    filename = 'output/' + video_name + '/' + 'motion_plots.png'
    plt.savefig(filename)
    plt.show()

