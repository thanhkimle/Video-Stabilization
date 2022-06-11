import os
import numpy as np
import cv2
from glob import glob
import stabilization as st

# user inputs
VIDEO = "penguins.mp4"
CROP_RATIO = 0.8


def main():
    video_name = VIDEO.replace('.', '_')
    # print("Converted video to frames ...")
    frames_path = st.video2frames(VIDEO)
    print("Reading images ...")
    images_list = st.readImages(frames_path)
    print('Total frames: ', len(images_list))
    image_size = images_list[0].shape
    print("Extracting features ...")
    features = st.extractFeatures(images_list)
    print("Calculating original camera path ...")
    original_path = st.findCameraPath(images_list, features)
    print("Calculating optimal camera path ...")
    optimal_path = st.findOptimalPath(original_path, image_size, CROP_RATIO)
    rect_output_path, crop_output_path = st.applyStabilization(video_name, images_list, optimal_path, 0.6)
    print("Converting frames to video ...")
    st.frames2video(rect_output_path)
    st.frames2video(crop_output_path)
    print("Plotting camera paths ...")
    st.graphPaths(video_name, original_path, optimal_path)


if __name__ == "__main__":
    # execute only if run as a script
    main()
    print("End of script")
