import os
import time
import argparse

import cv2
import torch
import torch.nn as nn
import numpy as np
import mediapipe as mp
from PIL import Image

from models import TalkingAnimeLight
from pose import get_pose
from utils import preprocessing_image, postprocessing_image


print(torch.cuda.is_available())
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

@torch.no_grad()
def main():
    model = TalkingAnimeLight().to(device)
    model = model.eval()
    model = model
    # 캐릭터 변화
    img = Image.open(f"character/yumi.png")
    img = img.resize((256, 256))
    input_image = preprocessing_image(img).unsqueeze(0)

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret is None:
        raise Exception("Can't find Camera")

    facemesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

    mouth_eye_vector = torch.empty(1, 27)
    pose_vector = torch.empty(1, 3)
    input_image = input_image.to(device)
    mouth_eye_vector = mouth_eye_vector.to(device)
    pose_vector = pose_vector.to(device)

    pose_queue = []

    while cap.isOpened():
        ret, frame = cap.read()
        input_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = facemesh.process(input_frame)

        if results.multi_face_landmarks is None:
            continue

        facial_landmarks = results.multi_face_landmarks[0].landmark


        pose, debug_image = get_pose(facial_landmarks, frame)


        if len(pose_queue) < 3:
            pose_queue.append(pose)
            pose_queue.append(pose)
            pose_queue.append(pose)
        else:
            pose_queue.pop(0)
            pose_queue.append(pose)

        np_pose = np.average(np.array(pose_queue), axis=0, weights=[0.6, 0.3, 0.1])

        eye_l_h_temp = np_pose[0]
        eye_r_h_temp = np_pose[1]
        mouth_ratio = np_pose[2]
        eye_y_ratio = np_pose[3]
        eye_x_ratio = np_pose[4]
        x_angle = np_pose[5]
        y_angle = np_pose[6]
        z_angle = np_pose[7]

        mouth_eye_vector[0, :] = 0

        mouth_eye_vector[0, 2] = eye_l_h_temp
        mouth_eye_vector[0, 3] = eye_r_h_temp

        mouth_eye_vector[0, 14] = mouth_ratio * 1.5

        mouth_eye_vector[0, 25] = eye_y_ratio
        mouth_eye_vector[0, 26] = eye_x_ratio

        pose_vector[0, 0] = (x_angle - 1.5) * 1.6
        pose_vector[0, 1] = y_angle * 2.0  # temp weight
        pose_vector[0, 2] = (z_angle + 1.5) * 2  # temp weight

        output_image = model(input_image, mouth_eye_vector, pose_vector)

        output_frame = cv2.cvtColor(postprocessing_image(output_image.cpu()), cv2.COLOR_RGBA2BGR)
        resized_frame = cv2.resize(output_frame, (np.min(debug_image.shape[:2]), np.min(debug_image.shape[:2])))
        output_frame = np.concatenate([debug_image, resized_frame], axis=1)
        cv2.imshow("frame", output_frame)
        # cv2.imshow("camera", debug_image)
        cv2.waitKey(1)



if __name__ == '__main__':
    main()
