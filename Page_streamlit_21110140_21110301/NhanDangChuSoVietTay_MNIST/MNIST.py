import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, models
import cv2
from PIL import Image

OPTIMIZER = tf.keras.optimizers.Adam()

# Load model
model_architecture = "D:\\Page_XuLyAnh_Streamlit_21110140_21110301\\Page_streamlit_21110140_21110301\\NhanDangChuSoVietTay_MNIST\\digit_config.json"
model_weights = "D:\\Page_XuLyAnh_Streamlit_21110140_21110301\\Page_streamlit_21110140_21110301\\NhanDangChuSoVietTay_MNIST\\digit.weights.h5"
model = models.model_from_json(open(model_architecture).read())
model.load_weights(model_weights)

model.compile(loss="categorical_crossentropy", optimizer=OPTIMIZER,
              metrics=["accuracy"])

# Data: shuffled and split between train and test sets
(_, _), (X_test, _) = datasets.mnist.load_data()

# Reshape
X_test = X_test.reshape((10000, 28, 28, 1))

def generate_random_image(grid_size):
    digit_size = 28  # Fixed size for MNIST digits
    # Generate random indices
    index = np.random.randint(0, 9999, grid_size**2)
    sample = np.zeros((grid_size**2, digit_size, digit_size, 1))
    for i in range(grid_size**2):
        sample[i] = X_test[index[i]].reshape(digit_size, digit_size, 1)

    image_size = grid_size * digit_size
    image = np.zeros((image_size, image_size), dtype=np.uint8)
    k = 0
    for i in range(grid_size):
        for j in range(grid_size):
            if k < grid_size**2:
                image[i*digit_size:(i+1)*digit_size, j*digit_size:(j+1)*digit_size] = sample[k, :, :, 0]
                k += 1

    color_converted = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image_pil = Image.fromarray(color_converted)

    # Normalize and cast
    data = sample / 255.0
    data = data.astype('float32')
    
    return image_pil, data

def main():
    st.title("Nhận dạng chữ số viết tay MNIST")

    grid_size = st.slider('Kích thước lưới (số chữ số trên mỗi hàng/cột)', 2, 20, 10, 1)

    if 'data' not in st.session_state:
        st.session_state['image'], st.session_state['data'] = generate_random_image(grid_size)

    if st.button('Tạo ảnh'):
        st.session_state['image'], st.session_state['data'] = generate_random_image(grid_size)
    
    st.image(st.session_state['image'], caption='Random MNIST', width=300, use_column_width='auto')

    
    if st.button('Nhận dạng'):
        prediction = model.predict(st.session_state['data'], verbose=0)
        result = prediction.argmax(axis=1)
        s = ''
        dem = 0    
        for x in result:
            s += str(x) + ' '
            dem += 1
            if dem % grid_size == 0:
                s += '<br>'
        st.markdown(f"<p style='font-size:16px; margin-top:0px; letter-spacing: 10px;'>{s}</p>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
