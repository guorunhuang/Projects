import cv2
import mediapipe as mp
import numpy as np

# 初始化MediaPipe人脸网格
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,  # 获取更精确的眼睛和嘴巴轮廓
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 关键点索引（MediaPipe预定义）
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
MOUTH = [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318]

def extract_facial_features(image, landmarks):
    """提取眼睛和嘴巴区域的关键点"""
    h, w = image.shape[:2]
    
    # 转换归一化坐标为像素坐标
    left_eye_points = [(int(landmarks.landmark[i].x * w), 
                       int(landmarks.landmark[i].y * h)) for i in LEFT_EYE]
    right_eye_points = [(int(landmarks.landmark[i].x * w), 
                        int(landmarks.landmark[i].y * h)) for i in RIGHT_EYE]
    mouth_points = [(int(landmarks.landmark[i].x * w), 
                    int(landmarks.landmark[i].y * h)) for i in MOUTH]
    
    return left_eye_points, right_eye_points, mouth_points

def create_potato_filter(image, left_eye, right_eye, mouth, potato_img):
    """创建土豆滤镜效果"""
    # 计算人脸中心和大小
    face_center = calculate_face_center(left_eye + right_eye + mouth)
    face_size = calculate_face_size(left_eye + right_eye + mouth)
    
    # 调整土豆大小和位置
    potato_resized = cv2.resize(potato_img, (face_size, face_size))
    
    # 将土豆贴到人脸位置
    result = overlay_potato(image, potato_resized, face_center)
    
    # 提取并贴上眼睛和嘴巴
    result = overlay_facial_features(result, image, left_eye, right_eye, mouth)
    
    return result

# 视频编码器设置
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter('potato_filter_video.mp4', fourcc, 30.0, (width, height))

# 在主循环中保存每一帧
if recording:
    video_writer.write(filtered_frame)