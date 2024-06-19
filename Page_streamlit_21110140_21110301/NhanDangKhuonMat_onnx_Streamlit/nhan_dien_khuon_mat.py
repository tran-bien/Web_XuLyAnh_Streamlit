import streamlit as st
import numpy as np
import cv2 as cv
import joblib
import os
from tempfile import NamedTemporaryFile

st.write("## Nhận diện khuôn mặt ONNX")
FRAME_WINDOW = st.image([])

# Add file uploader for video
uploaded_file = st.file_uploader("Upload an MP4 video", type=["mp4"])

if 'stop' not in st.session_state:
    st.session_state.stop = False
if 'predict_camera' not in st.session_state:
    st.session_state.predict_camera = False

stop_button = st.button('stop/Continue')
predict_camera_button = st.button('Predict Camera')

if stop_button:
    st.session_state.stop = not st.session_state.stop
    if not st.session_state.stop:
        # Nếu đã nhấn "continue", xóa ảnh đi
        FRAME_WINDOW.empty()
if predict_camera_button:
    st.session_state.predict_camera = not st.session_state.predict_camera
    if not st.session_state.stop:
        # Nếu đã nhấn "continue", xóa ảnh đi
        FRAME_WINDOW.empty()

print('Trang thai nhan stop', st.session_state.stop)

if 'frame_stop' not in st.session_state:
    frame_stop_path = os.path.join(os.path.dirname(__file__), 'stop.jpg')
    frame_stop = cv.imread(frame_stop_path)
    st.session_state.frame_stop = frame_stop
    print('Đã load stop.jpg')

if st.session_state.stop:
    FRAME_WINDOW.image(st.session_state.frame_stop, channels='BGR')

# Đường dẫn tuyệt đối đến file ONNX và SVC
onnx_path = os.path.join(os.path.dirname(__file__), 'face_detection_yunet_2023mar.onnx')
svc_path = os.path.join(os.path.dirname(__file__), 'svc.pkl')
face_recognition_model_path = os.path.join(os.path.dirname(__file__), 'face_recognition_sface_2021dec.onnx')

# Kiểm tra xem file ONNX và SVC có tồn tại không
if not os.path.exists(onnx_path):
    st.error(f"Không tìm thấy file '{onnx_path}'.")
    st.stop()
if not os.path.exists(svc_path):
    st.error(f"Không tìm thấy file '{svc_path}'.")
    st.stop()
if not os.path.exists(face_recognition_model_path):
    st.error(f"Không tìm thấy file '{face_recognition_model_path}'.")
    st.stop()

# Đảm bảo rằng OpenCV có thể đọc được file ONNX
recognizer = cv.FaceRecognizerSF.create(
    face_recognition_model_path, "")

detector = cv.FaceDetectorYN.create(
    onnx_path,
    "",
    (320, 320),
    0.9,
    0.3,
    5000)

svc = joblib.load(svc_path)

mydict = ['DucPhu','HPCong', 'Loki', 'PhungThanh', 'ThayDuc','TranNgocBien']

def visualize(input, faces, fps, thickness=2):
    if faces[1] is not None:
        for idx, face in enumerate(faces[1]):
            coords = face[:-1].astype(np.int32)
            cv.rectangle(input, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), thickness)
            cv.circle(input, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
            cv.circle(input, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
            cv.circle(input, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
            cv.circle(input, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
            cv.circle(input, (coords[12], coords[13]), 2, (0, 255, 255), thickness)
    cv.putText(input, 'FPS: {:.2f}'.format(fps), (1, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def process_video(cap):
    tm = cv.TickMeter()

    frameWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    detector.setInputSize([frameWidth, frameHeight])

    while True:
        if st.session_state.stop:
            FRAME_WINDOW.image(st.session_state.frame_stop, channels='BGR')
            break

        hasFrame, frame = cap.read()
        if not hasFrame:
            print('No frames grabbed!')
            break

        tm.start()
        faces = detector.detect(frame)
        tm.stop()

        if faces[1] is not None:
            for face in faces[1]:
                face_align = recognizer.alignCrop(frame, face)
                face_feature = recognizer.feature(face_align)
                test_predict = svc.predict(face_feature)
                result = mydict[test_predict[0]]
                coords = face[:-1].astype(np.int32)
                cv.putText(frame, result, (coords[0], coords[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        visualize(frame, faces, tm.getFPS())
        FRAME_WINDOW.image(frame, channels='BGR')
    cap.release()
    cv.destroyAllWindows()

if uploaded_file is not None:
    with NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        temp_file.write(uploaded_file.read())
        temp_filename = temp_file.name
    cap = cv.VideoCapture(temp_filename)
    process_video(cap)
elif st.session_state.predict_camera:
    cap = cv.VideoCapture(0)
    process_video(cap)
