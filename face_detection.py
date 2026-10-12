import cv2 as cv
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# Face Detection:
# 1. Cấu hình Face Detector
out_dir = './Open_CV/project/Image'
model_path = os.path.join('project','blaze_face_short_range.tflite')
base_options = python.BaseOptions(model_asset_path=model_path) # hoặc file .tflite bạn đang dùng
options = vision.FaceDetectorOptions(base_options = base_options, running_mode=vision.RunningMode.IMAGE)

with vision.FaceDetector.create_from_options(options) as detector:
    image_path = os.path.join('project', 'Image', 'test_2.jpg')
    mp_image = mp.Image.create_from_file(image_path)
    
    # Đọc ảnh OpenCV để vẽ lên
    image = cv.imread(image_path)
    # 3. Chạy Face Detection
    face_detector_result = detector.detect(mp_image)
    print(face_detector_result)
    # 4. Duyệt qua từng khuôn mặt phát hiện được và vẽ Bounding Box
    print("____________________________________________________________")
    for detection in face_detector_result.detections:
        print(detection.keypoints)
        bbox = detection.bounding_box
        x = bbox.origin_x
        y = bbox.origin_y
        w = bbox.width
        h = bbox.height
        start_point = (x, y)
        end_point = (x+ w, y + h)
        # Vẽ hình chữ nhật màu xanh lá (BGR: 0, 255, 0), độ dày 2p
        image[y:y+h,x:x+w,:] = cv.blur(src=  image[y:y+h,x:x+w,:],ksize=(70,70))
        cv.rectangle(image,start_point,end_point,color=[256,0,0],thickness= 1)

    # 5. Hiển thị kết quả
    cv.imshow('Face Detection', image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    # Lưu ảnh 
    cv.imwrite(os.path.join(out_dir,'image.jpg'),image)