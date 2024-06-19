import streamlit as st
from streamlit_option_menu import option_menu
import subprocess
import os

# Biến để theo dõi chương trình hiện tại
current_program = None
image_path = "D:/Page_XuLyAnh_Streamlit_21110140_21110301/Logo-DH-Su-Pham-Ky-Thuat-TP-Ho-Chi-Minh-HCMUTE-623x800.jpg"

def run_program(program_folder, program_file):
    global current_program
    # Đóng chương trình hiện tại (nếu có)
    if current_program and current_program.poll() is None:
        st.text("Đang đóng chương trình trước...")
        current_program.terminate()
    # Mở chương trình mới
    current_program = subprocess.Popen(['python', '-m', 'streamlit', 'run', os.path.join(program_folder, program_file)])

def main():

    # Hiển thị ảnh với các tùy chọn
    st.image(image_path, width=100)
    st.title("Đồ án môn học cuối kỳ")
    st.write("Môn: Xử lý Ảnh Số, Nhóm 04CLC Sáng T5")
    st.write("DIPR430685_23_2_04CLC")
    st.write("### Thông tin")
    
    # Sử dụng các cột để bố trí thông tin
    col1, col2 = st.columns(2)
    with col1:
        st.write("SV: 1")
        st.write(" Họ tên: Trần Ngọc Biên")
        st.write(" Mã số sinh viên: 21110140")
    
    with col2:
        st.write("SV: 2")
        st.write("Họ tên: Phùng Hửu Thành")
        st.write("Mã số sinh viên: 21110301")

    # Đường dẫn đến thư mục streamlit_final
    streamlit_final_folder = os.path.join(os.getcwd(), "D:/Page_XuLyAnh_Streamlit_21110140_21110301/Page_streamlit_21110140_21110301")
    if not os.path.exists(streamlit_final_folder):
        st.error("Không tìm thấy thư mục streamlit_final.")
        return

    # Danh sách các chương trình
    programs_info = {
        "Phần 1: Nhận diện khuôn mặt": {
            "folder_name": "NhanDangKhuonMat_onnx_Streamlit",
            "file_name": "nhan_dien_khuon_mat.py"
        },
        "Phần 2: Nhận diện trái cây": {
            "folder_name": "NhanDang5DoiTuong_yolov8",
            "file_name": "nhan_dang_trai_cay.py"
        },
        "Phần 3: Nhận dạng chữ số viết tay": {
            "folder_name": "NhanDangChuSoVietTay_MNIST",
            "file_name": "MNIST.py"
        },
        "Phần 4: Nội dung Xử lý ảnh số": {
            "folder_name": "XuLyAnh",
            "file_name": "Xu_Ly_Anh.py"
        }
    }

    # Sử dụng option_menu để tạo các nút với icon
    selected_program = option_menu(
        "Chọn Page Chương Trình",
        options=list(programs_info.keys()),
        icons=["bi-emoji-smile", "apple", "pencil", "image"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal"
    )

    # Lấy thông tin về chương trình được chọn
    selected_program_info = programs_info[selected_program]

    # Đường dẫn đến thư mục chứa chương trình
    program_folder = os.path.join(streamlit_final_folder, selected_program_info["folder_name"])

    # Kiểm tra xem thư mục chương trình có tồn tại không
    if not os.path.exists(program_folder) and selected_program != "Thông tin":
        st.error(f"Không tìm thấy thư mục chứa chương trình '{selected_program}'.")
        return

    # Đường dẫn và tên file của chương trình
    program_file = selected_program_info["file_name"]
    program_path = os.path.join(program_folder, program_file)

    # Kiểm tra xem file của chương trình có tồn tại không
    if not os.path.exists(program_path) and selected_program != "Thông tin":
        st.error(f"Không tìm thấy file '{program_file}' trong thư mục chứa chương trình '{selected_program}'.")
        return

    # Nút để chạy chương trình
    if st.button("Run Page Chương Trình"):
        run_program(program_folder, program_file)

if __name__ == "__main__":
    main()
