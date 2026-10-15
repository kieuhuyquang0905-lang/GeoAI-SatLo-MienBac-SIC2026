# 🛰️ HỆ THỐNG CẢNH BÁO VÀ PHÂN VÙNG NGUY CƠ SẠT LỞ ĐẤT MIỀN BẮC VIỆT NAM (GEO-AI SIC 2026)

**Kiến trúc:** `PU-Bagging + Stacking Ensemble (Random Forest, LightGBM, XGBoost)`  
**Hệ quy chiếu chuẩn hóa:** `EPSG:32648 (WGS 84 / UTM Zone 48N)` $\leftrightarrow$ `EPSG:4326 (GPS WGS84)`  
**Quy mô không gian:** `28.990 điểm lưới toàn Miền Bắc Việt Nam` | **Kiểm chứng thực địa:** `135 điểm sạt lở lịch sử (NASA GLC & Báo chí 2015–2025)`

---

## 📌 1. Tổng quan Quy trình 3 Giai đoạn (3-Stage Geo-AI Pipeline)

Dự án giải quyết bài toán dự báo không gian và cảnh báo sớm nguy cơ trượt lở đất tại khu vực Miền Bắc Việt Nam thông qua kiến trúc **3 giai đoạn khép kín**:

1. **Stage 1 – Tiền xử lý Không gian & Kiểm định Độc lập Tuyến tính (13 Nhân tố):**
   * Đồng bộ hóa 13 lớp dữ liệu địa hình, địa chất, thủy văn và viễn thám về hệ tọa độ phẳng `EPSG:32648`: `Slope` (Độ dốc), `RoadDist` (Khoảng cách đường giao thông), `DEM` (Độ cao), `FaultDist` (Khoảng cách đứt gãy), `Rainfall` (Lượng mưa), `RiverDist` (Khoảng cách sông suối), `TWI` (Chỉ số ẩm địa hình), `PhanCatNgang` (Độ phân cắt ngang), `NDVI` (Chỉ số thực vật), `PhanCatSau` (Độ phân cắt sâu), `Litho_Code` (Thạch học), `Aspect` (Hướng sườn), `LUIC` (Sử dụng đất).
   * Tính toán **Tỷ số tần suất ($FR$)** và **Mật độ xác suất Entropy ($P_{ij}$) theo tiếp cận `Slope-IOE`**, chứng minh các cấp độ dốc $< 15^\circ$ đều có $FR < 1.0$, làm cơ sở khoa học để lọc nhiễu bán kính Buffer 150m và xác định vùng lấy mẫu an toàn.
   * Kiểm định đa cộng tuyến đạt chuẩn: **$\text{VIF}_{\max} = 2.010 < 2.30$** (`LUIC = 2.010`, `NDVI = 1.905`) và hệ số tương quan cặp **Pearson $|r|_{\max} = 0.668 < 0.70$**.

2. **Stage 2 – Thực nghiệm Chéo 18 Kịch bản & Giải mã Mức độ Đóng góp 13 Nhân tố:**
   * Đánh giá toàn diện **18 kịch bản** (`3 chiến lược lấy mẫu âm: Random Sampling, Slope-IOE, PU-Bagging` $\times$ `6 cấu hình mô hình: RF, LightGBM, XGBoost, DNN, Stacking_Full4, Stacking_Trees3`).
   * Cấu hình tối ưu **`PU-Bagging + Stacking_Trees3 (RF + LightGBM + XGBoost)`** đạt hiệu năng cao nhất trên tập kiểm thử độc lập (30% Test): **$\text{AUC} = 0.950$**, **$\text{ACC} = 89.2\%$**, **$\text{Kappa} = 0.784$**, **$\text{Sensitivity} = 96.1\%$**.
   * Trích xuất tỷ trọng đóng góp của 13 nhân tố ở quy mô liên vùng Miền Bắc: **`Độ dốc (23.14%)`** $>$ **`Khoảng cách đường giao thông (13.51%)`** $>$ **`Độ cao DEM (11.56%)`** $>$ **`Khoảng cách đứt gãy (8.43%)`** $>$ **`Lượng mưa (8.36%)`** $>$ **`Khoảng cách sông suối (7.31%)`**.

3. **Stage 3 – Bản đồ Phân vùng Nguy cơ 5 Cấp độ & Hệ thống Web/App Định vị GPS Thời gian thực:**
   * Tích hợp **Bộ lọc Địa mạo – Thủy văn – Đô thị (`is_safe_geomorph`)**: Khử triệt để nhiễu độ dốc giả tạo (`SRTM DSM`) do nhà cao tầng tại các đô thị lớn (như TP. Hà Nội) và bẫy chỉ số thực vật âm (`NDVI < 0.12`) tại mặt nước sông hồ / hố khai thác ngập nước (như Hồ Tuyệt Tình Cốc – Hải Phòng).
   * Thành lập Bản đồ Phân vùng Nguy cơ Sạt lở 5 cấp độ chuẩn xuất bản khoa học `(A) Random_Stacking_Full4`, `(B) Slope-IOE_RF`, `(C) PU-Bagging_Stacking_Trees3`.
   * Triển khai ứng dụng Web tương tác (`index.html`) hỗ trợ định vị GPS thời gian thực, kết nối API thời tiết trực tiếp (`Open-Meteo`) và đối chiếu 135 điểm sạt lở lịch sử.

---

## 📂 2. Danh mục Cấu trúc File trên GitHub Repository

| Tên File | Vai trò trong Dự án |
| :--- | :--- |
| **`index.html`** | Giao diện Web Bản đồ Cảnh báo Sạt lở & Định vị GPS Thời gian thực (chạy trực tiếp trên **GitHub Pages**). |
| **`SIC2026_AI_Training_Clean.ipynb`** | Sổ tay Jupyter/Colab chứa toàn bộ mã nguồn và kết quả hiển thị sẵn của **Stage 1, Stage 2, Stage 3**. |
| **`sic2026_ai_training_clean.py`** | Script Python thuần (`.py`) đã chuẩn hóa để chạy trực tiếp qua Terminal/IDE. |
| **`MaTran_Train_SIC2026.csv`** | Bộ dữ liệu huấn luyện gốc gồm 13 nhân tố không gian. |
| **`DB_GeoAI_MienBac_GPS_ChuanHoa.csv`** | CSDL lưới `28.990 điểm` toàn Miền Bắc đã chuyển đổi tọa độ `GPS WGS84` và tích hợp xác suất dự báo. |
| **`Stage2_Bang_SoSanh_18_KichBan.csv`** | Bảng kết quả thực nghiệm chéo 18 kịch bản (`3 chiến lược lấy mẫu x 6 mô hình`). |
| **`Bang_DoiChieu_135_Diem_ChuanHoa_SIC2026.csv`** | Bảng kiểm chứng độc lập trên 135 điểm sạt lở thực tế (55 điểm NASA GLC & 80 điểm thực địa báo chí). |
| **`Stage1_Pearson_Correlation_Matrix.png`** | Biểu đồ nhiệt Ma trận tương quan Pearson 13 nhân tố (`|r|_max = 0.668 < 0.70`). |
| **`Feature_Importance_ChuanDiaChat_SIC2026.png`** | Biểu đồ mức độ đóng góp (%) của 13 nhân tố gây sạt lở tại Miền Bắc. |
| **`ROC_Curve_ChuanDiaChat_SIC2026.png`** | Biểu đồ đường cong ROC so sánh mô hình Stacking Ensemble và các mô hình đơn lẻ. |
| **`Hinh5_BanDo_LSM_Chuan_100_VJST.png`** | Bản đồ phân vùng nguy cơ sạt lở đất 5 cấp độ `(A), (B), (C)` kèm kim chỉ Bắc và thước tỷ lệ. |
| **`requirements.txt`** | Danh sách các thư viện Python cần thiết để tái lập môi trường chạy code. |

---

## 🚀 3. Hướng dẫn Chạy Dự án

1. **Trải nghiệm trực tiếp trên Web (GitHub Pages):**
   * Mở trực tiếp file `index.html` trên trình duyệt hoặc truy cập đường dẫn GitHub Pages của kho lưu trữ.
2. **Chạy Huấn luyện & Xuất Bản đồ trên Google Colab / Local:**
   ```bash
   pip install -r requirements.txt
   python sic2026_ai_training_clean.py
   ```
