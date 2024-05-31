from PyQt5 import QtWidgets, uic, QtCore
import sys
import os
import time
import threading
import earnedvaluemanagement as evm
import pickle

class MyWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        uic.loadUi('tool.ui', self)  # Tải giao diện từ tệp .ui

        # Kết nối nút với phương thức
        self.myButton = self.findChild(QtWidgets.QPushButton, 'pushButton')  # Tìm nút Select File theo tên
        self.myButton.clicked.connect(self.open_file_dialog)

        self.runButton = self.findChild(QtWidgets.QPushButton, 'pushButton_2')  # Tìm nút Chạy mô hình theo tên
        self.runButton.clicked.connect(self.run_model)

        # Tìm QLabel và QProgressBar
        self.label = self.findChild(QtWidgets.QLabel, 'label')  # Tìm QLabel theo tên
        self.progressBar = self.findChild(QtWidgets.QProgressBar, 'progressBar')  # Tìm QProgressBar theo tên

        # Tạo đối tượng QTimer
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_progress)  # Kết nối timeout signal với phương thức update_progress

        #Xaác định mode, mode default là mode 1: dùng Model đã train, mode 2: train model mới
        self.mode1.clicked.connect(self.on_radio_button_clicked)
        self.mode2.clicked.connect(self.on_radio_button_clicked)
        self.selected_mode = 1

        # Biến trạng thái để xác định xem mô hình đã chạy xong hay chưa
        self.model_running = False

        # Kết nối sự kiện tắt cửa sổ với hàm xử lý
        self.closeEvent = self.close_window

    #Xử lý mode được chọn
    def on_radio_button_clicked(self):
        if self.mode1.isChecked():
            self.selected_mode = 1
        else:
            self.selected_mode = 2

    def open_file_dialog(self):
        options = QtWidgets.QFileDialog.Options()
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select File", "", "Excel Files (*.xlsx)", options=options)
        if file_name:
            base_name = os.path.basename(file_name)  # Lấy tên tệp (không bao gồm đường dẫn)
            self.label.setText(base_name)  # Cập nhật QLabel với tên tệp
            self.file_name = file_name  # Lưu tên tệp đầy đủ để sử dụng sau

    def run_model(self):
        if hasattr(self, 'file_name') and not self.model_running:
            self.on_radio_button_clicked()
            self.progressBar.setValue(0)  # Đặt giá trị ban đầu của thanh tiến trình là 0%
            self.total_steps = 18  # Số lần cập nhật tiến trình (18 lần * 5% = 90%)
            self.current_step = 0

            #mode 2: train model mới time lâu hơn rất nhiều
            if self.selected_mode == 2:
                self.timer.start(20000)
            else:
                self.timer.start(5000)  # Bắt đầu timer, kích hoạt mỗi 5 giây

            # Tạo một luồng riêng biệt để chạy mô hình
            self.model_thread = threading.Thread(target=self.run_evm_model, args=(self.file_name,))
            self.model_thread.start()
        elif self.model_running:
            QtWidgets.QMessageBox.warning(self, "Warning", "Model is already running!")
        else:
            QtWidgets.QMessageBox.warning(self, "Warning", "No file selected!")

    def run_evm_model(self, file_name):
        # Đánh dấu là mô hình đang chạy
        self.model_running = True

    
        # Giả lập quá trình xử lý dữ liệu
        inputName = file_name

        #mode 1: dùng Model đã train
        if self.selected_mode == 1:
            # Tải model
            with open('case2.pkl', 'rb') as f:
                case = pickle.load(f)

            # Tải sample của model
            with open('case_sample2.pkl', 'rb') as f:
                case_sample = pickle.load(f)
        
        #mode 2: train model mới
        if self.selected_mode == 2:
            PRJ = evm.project_reader(inputName)
            case = evm.build_model(PRJ)
            case_sample = evm.sample_model(case)

        # Thực hiện chạy mô hình
        evm.excel_posterior(case_sample, inputName)

        # Đánh dấu là mô hình đã chạy xong
        self.model_running = False

    def update_progress(self):
        if self.model_running:
            if int((self.current_step + 1) * 5) == 90:
                self.progressBar.setValue(90)
                self.label.setText(f"Tiến trình: {self.progressBar.value()}%")
            else:
                self.progressBar.setValue(int((self.current_step + 1) * 5))  # Tăng tiến trình lên 5%
                self.current_step += 1
                self.label.setText(f"Tiến trình: {self.progressBar.value()}%")
        else:
            if (1):
                self.label.setText("Tiến trình: Hoàn thành")
                self.progressBar.setValue(100)  # Đặt giá trị của thanh tiến trình thành 100%
                self.timer.stop()  # Dừng timer

    def close_window(self, event):
        if self.model_running:
            QtWidgets.QMessageBox.warning(self, "Warning", "Model is still running! Please wait for it to finish.")
            event.ignore()  # Không đóng cửa sổ
        else:
            event.accept()  # Đóng cửa sổ

app = QtWidgets.QApplication(sys.argv)
window = MyWindow()
window.show()
app.exec_()
