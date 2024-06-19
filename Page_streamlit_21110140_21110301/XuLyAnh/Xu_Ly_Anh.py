import streamlit as st
import numpy as np
import cv2
import module.Chapter3 as c3
import module.Chapter4 as c4
import module.Chapter5 as c5
import module.Chapter9 as c9
import base64

st.set_page_config(initial_sidebar_state="expanded")

def get_image_download_link(img, filename, text):
    buffered = cv2.imencode('.jpg', img)[1].tobytes()
    b64 = base64.b64encode(buffered).decode()
    href = f'<a href="data:file/jpg;base64,{b64}" download="{filename}">{text}</a>'
    return href

st.write("## Xử lý ảnh")

uploaded_file = st.file_uploader("Chọn một ảnh...", type=["jpg", "jpeg", "png", "bmp", "tif", "gif"])

all_operations = {
    "Chapter 3": [
        "Negative",
        "Logarithmic",
        "Piecewise Linear",
        "Histogram",
        "Histogram Equalization",
        "Histogram Equalization Color",
        "Local Hist",
        "Hist Stat",
        "Box Filter",
        "Lowpass Gauss",
        "Threshold",
        "Median Filter",
        "Sharpen",
        "Gradient"
    ],
    "Chapter 4": [
        "Spectrum",
        "Frequency Filter",
        "Draw Notch Reject Filter",
        "Remove Moire"
    ],
    "Chapter 5": [
        "Create Motion Noise",
        "Denoise Motion",
        "Denoise Est Motion"
    ],
    "Chapter 9": [
        "Connected Component",
        "Count Rice"
    ]
}

operation_names = [(chapter, operation) for chapter, operations in all_operations.items() for operation in operations]
operation_names_display = [f"{chapter} - {operation}" for chapter, operation in operation_names]

selected_operation = st.sidebar.selectbox("Chọn kiểu xử lý ảnh", operation_names_display)
chapter, operation = selected_operation.split(" - ")

# imgout = None

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    imgin = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    imgin_color = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(imgin, caption='Ảnh đã tải lên', use_column_width=True)

    if st.sidebar.button("Xác nhận xử lý ảnh"):
        if chapter == "Chapter 3":
            if operation == "Negative":
                imgout = c3.Negative(imgin)
            elif operation == "Logarithmic":
                imgout = c3.Logarithm(imgin)
            elif operation == "Piecewise Linear":
                imgout = c3.PiecewiseLinear(imgin)
            elif operation == "Histogram":
                imgout = c3.Histogram(imgin)
            elif operation == "Histogram Equalization":
                imgout = c3.HistEqual(imgin)
            elif operation == "Histogram Equalization Color":
                imgout = c3.HistEqualColor(imgin_color)
            elif operation == "Local Hist":
                imgout = c3.LocalHist(imgin)
            elif operation == "Hist Stat":
                imgout = c3.HistStat(imgin)
            elif operation == "Box Filter":
                imgout = c3.BoxFilter(imgin)
            elif operation == "Lowpass Gauss":
                imgout = c3.LowpassGauss(imgin)
            elif operation == "Threshold":
                imgout = c3.Threshold(imgin)
            elif operation == "Median Filter":
                imgout = c3.MedianFilter(imgin)
            elif operation == "Sharpen":
                imgout = c3.Sharpen(imgin)
            elif operation == "Gradient":
                imgout = c3.Gradient(imgin)

        elif chapter == "Chapter 4":
            if operation == "Spectrum":
                imgout = c4.Spectrum(imgin)
            elif operation == "Frequency Filter":
                imgout = c4.FrequencyFilter(imgin)
            elif operation == "Draw Notch Reject Filter":
                imgout = c4.DrawNotchRejectFilter()
            elif operation == "Remove Moire":
                imgout = c4.RemoveMoire(imgin)

        elif chapter == "Chapter 5":
            if operation == "Create Motion Noise":
                imgout = c5.CreateMotionNoise(imgin)
            elif operation == "Denoise Motion":
                imgout = c5.DenoiseMotion(imgin)
            elif operation == "Denoise Est Motion":
                temp = cv2.medianBlur(imgin, 7)
                imgout = c5.DenoiseMotion(temp)

        elif chapter == "Chapter 9":         
            if operation == "Connected Component":
                imgout = c9.ConnectedComponent(imgin)
            elif operation == "Count Rice":
                imgout = c9.CountRice(imgin)

        st.image(imgout, caption=f'{operation} Image', use_column_width=True)

    # Thêm phần lưu ảnh sau khi xử lý xong
        if imgout is not None:
            st.sidebar.markdown(get_image_download_link(imgout, 'output.jpg', 'DownLoad_Image'), unsafe_allow_html=True)
