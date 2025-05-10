# 🔥 GPAPredictor
## 📖 Cách sử dụng 
### 1. Yêu cầu hệ thống
- Haddop HDFS
- Cài đặt các thư viện trong requirement.txt
### 2. Train model
```bash
   git clone https://github.com/Hatsuki9813/GPAPredictor.git
   ```
- Thực hiện uncomment các dòng để save model vào HDFS ( #mode.save(...) )
- Chỉnh sửa code để save / load model từ HDFS tùy vào cấu trúc thư mục HDFS của bạn
### 3. Sử dụng Streamlt UI
```bash
   streamlit run Home.py
   ```
- Chọn Model
- Chọn số học kì đã học
- Nhấn dự đoán
