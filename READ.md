# Project 5.4 – Wafer Spin-Coater Acceleration Control

## 1. Giới thiệu
Chương trình mô phỏng bằng Python quá trình điều khiển gia tốc góc $\alpha(t)$ của máy phủ quay wafer (spin-coater) trong sản xuất bán dẫn (Topic 5: Non-Uniform Angular Kinematics - General Physics 1 CC22). Chương trình khảo sát 3 quy luật gia tốc (Tuyến tính, S-curve, Hàm mũ) và đánh giá thời gian tăng tốc, độ giật (jerk), gia tốc tại mép wafer, mô-men xoắn, công suất và sai số số học.

## 2. Cài đặt môi trường
Yêu cầu cài đặt **Python 3.10 trở lên**. Mở Terminal tại thư mục gốc của project và chạy lệnh:
```bash
pip install -r requirements.txt