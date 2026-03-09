import cv2
import numpy as np
import sqlite3
import os
from datetime import datetime
from backend import PlantDetector

class PlantDetectorV2(PlantDetector):
    def __init__(self):
        super().__init__()
        if not os.path.exists("pic"):
            os.makedirs("pic")
        self.init_db()

    def init_db(self):
        """Khởi tạo cấu trúc DB chỉ lưu Rộng và Dài. 
        Tự động làm sạch nếu bảng cũ không khớp cấu trúc."""
        try:
            conn = sqlite3.connect("thongtin.db")
            cursor = conn.cursor()
            
            # Kiểm tra số lượng cột hiện tại của bảng (nếu đã tồn tại)
            cursor.execute("PRAGMA table_info(thongtin)")
            columns = cursor.fetchall()
            
            # Cấu trúc mới cần đúng 7 cột: id, thoi_gian, stt, ten_vat_the, rong_m, dai_m, image_path
            # Nếu bảng cũ có 9 cột (do có thêm diện tích, thể tích), ta xóa đi để tạo lại
            if len(columns) > 0 and len(columns) != 7:
                cursor.execute("DROP TABLE thongtin")
                conn.commit()
                print("Đã xóa bảng cũ để cập nhật cấu trúc mới (Chỉ Rộng/Dài).")

            # Tạo bảng với cấu trúc mới
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS thongtin (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thoi_gian TEXT, 
                    stt INTEGER, 
                    ten_vat_the TEXT,
                    rong_m REAL, 
                    dai_m REAL, 
                    image_path TEXT
                )
            ''')
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Lỗi khởi tạo DB: {e}")

    def delete_record(self, record_id, image_path):
        """Xóa dữ liệu và file ảnh"""
        try:
            if os.path.exists(image_path):
                os.remove(image_path)
            conn = sqlite3.connect("thongtin.db")
            cursor = conn.cursor()
            cursor.execute("DELETE FROM thongtin WHERE id = ?", (record_id,))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Lỗi khi xóa: {e}")
            return False

    def get_detailed_info(self, image, detections, predictions, px_per_m=100.0):
        """Chỉ tính toán Rộng và Dài"""
        detailed_data = []
        scale = 1.0 / px_per_m
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det['box']
            mask = det['mask']
            label, prob = predictions[i] if i < len(predictions) else ("Unknown", 0.0)
            
            # Crop vùng vật thể để lưu ảnh
            margin = 15
            h, w, _ = image.shape
            cy1, cy2 = max(0, y1-margin), min(h, y2+margin)
            cx1, cx2 = max(0, x1-margin), min(w, x2+margin)
            crop_img = image[cy1:cy2, cx1:cx2].copy()
            
            # Vẽ khung lên ảnh crop
            cv2.rectangle(crop_img, (x1-cx1, y1-cy1), (x2-cx1, y2-cy1), (0, 255, 0), 2)
            pts = (mask.reshape((-1, 1, 2)) - [cx1, cy1]).astype(int)
            cv2.polylines(crop_img, [pts], True, (0, 0, 255), 2)

            # Tính kích thước thực tế (m)
            width_m = (x2 - x1) * scale
            height_m = (y2 - y1) * scale

            detailed_data.append({
                "STT": i + 1,
                "Tên vật thể": label,
                "Rộng (m)": round(width_m, 3),
                "Dài (m)": round(height_m, 3),
                "raw_crop": crop_img 
            })
        return detailed_data

    def save_to_sqlite(self, data_list):
        """Lưu vào SQLite (Chỉ 6 giá trị: thoi_gian, stt, ten, rong, dai, path)"""
        try:
            conn = sqlite3.connect("thongtin.db")
            cursor = conn.cursor()
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            for item in data_list:
                timestamp = datetime.now().strftime("%H%M%S")
                file_path = f"pic/obj_{item['STT']}_{timestamp}.jpg"
                cv2.imwrite(file_path, item["raw_crop"])
                
                # Cập nhật query INSERT chính xác cho 6 cột
                cursor.execute('''
                    INSERT INTO thongtin (thoi_gian, stt, ten_vat_the, rong_m, dai_m, image_path)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (now, item["STT"], item["Tên vật thể"], item["Rộng (m)"], 
                      item["Dài (m)"], file_path))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            # Nếu lỗi, in lỗi chi tiết ra màn hình đen (terminal) để kiểm tra
            print(f"LỖI LƯU DB CHI TIẾT: {e}")
            return False
