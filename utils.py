import cv2
import mediapipe as mp
import numpy as np
import torch
from torchvision import transforms
from s3d_twostream import TwoStreamS3D
import pandas as pd

def detetect_sign(model, label_dict):
    """
    Open the webcam, detect people using MediaPipe, capture 120 frames with people,
    select 16 frames from those, and use the model to detect a sign.

    Parameters:
    - model: TwoStreamS3D model for sign detection.
    - label_dict: dictionary labels

    Returns:
    None
    """
        
    cap = cv2.VideoCapture(0)  # 0 for default webcam, change if you have multiple cameras

    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic()

    frame_list = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Make detections
        results = holistic.process(rgb_frame)

        # Check for people detection (you might need to adjust this condition)
        if results.pose_landmarks:
            if len(frame_list) < 120:
                frame_list.append(frame.copy())
            else:
                frame_list[-1] = frame.copy()
                num_frame = 16
                step = len(frame_list) // num_frame
                frame_list = frame_list[::step][:16]
                sign = detect(model, frame_list)
                print(label_dict.iloc[sign[0].item()])
        else:
            frame_list = []

        # Display the frame
        cv2.imshow("Webcam", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

def get_poses(frames):
    """
    Process a list of frames to detect faces and pose landmarks using MediaPipe.

    Parameters:
    - frames: List of frames (images) captured from the webcam.

    Returns:
    List of images with face and pose landmarks drawn.
    """

    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    poses = []

    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection, \
            mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose_detection:

        for frame in frames:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            w, h, c = image_rgb.shape
            image_black = np.zeros((w, h, c), dtype=np.uint8)

            face_results = face_detection.process(image_rgb)
            if face_results.detections:
                for detection in face_results.detections:
                    mp_drawing.draw_detection(frame, detection)

            pose_results = pose_detection.process(image_rgb)
            if pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(image_black, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                connections = [
                    (mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.LEFT_SHOULDER),
                    (mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.RIGHT_SHOULDER)]

                for connection in connections:
                    start_point = pose_results.pose_landmarks.landmark[connection[0]]
                    end_point = pose_results.pose_landmarks.landmark[connection[1]]
                    start_x, start_y = int(start_point.x * frame.shape[1]), int(start_point.y * frame.shape[0])
                    end_x, end_y = int(end_point.x * frame.shape[1]), int(end_point.y * frame.shape[0])
                    cv2.line(image_black, (start_x, start_y), (end_x, end_y), (255, 255, 255), 2)

            poses.append(image_black)
    return poses

def get_transform():
    """
    Get the image transformation pipeline for input frames.

    Returns:
    torchvision.transforms.Compose: Image transformation pipeline.
    """
    
    return transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def transform_frames(frames):
    """
    Apply image transformations to a list of frames.

    Parameters:
    - frames: List of frames (images) to be transformed.

    Returns:
    torch.Tensor: Transformed frames as a tensor with size (1, 16, 3, 224, 224).
    """

    transform = get_transform()
    frames = [transform(frame) for frame in frames]
    frames = torch.stack(frames)
    frames = frames.reshape((frames.shape[1], frames.shape[0], frames.shape[2], frames.shape[3]))
    return frames.unsqueeze(0)

def detect(model, frames):
    """
    Use the TwoStreamS3D model to predict signs based on input frames.

    Parameters:
    - model: TwoStreamS3D model for sign detection.
    - frames: Transformed frames as a tensor.

    Returns:
    torch.Tensor: Predicted sign labels.
    """

    poses = get_poses(frames)
    frames, poses = transform_frames(frames), transform_frames(poses)
    outputs = model(frames, poses)
    _, predict = torch.max(outputs, 1)
    return predict

if __name__ == "__main__":
    model = TwoStreamS3D(num_classes=226).to('cpu')
    label_dict = pd.read_csv('SignList_ClassId_TR_EN.csv', index_col=['ClassId'])
    # model.load_state_dict(torch.load('s3d_two_stream_model.pth'))
    model.eval()
    detetect_sign(model, label_dict)