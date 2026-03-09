import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
from PIL import Image, ImageTk
import cv2
import threading
import os
import math
import sqlite3
from backend2 import PlantDetectorV2

def log(app, msg):
    app.log_box.configure(state='normal')
    app.log_box.insert(tk.END, f">>> {msg}\n")
    app.log_box.see(tk.END)
    app.log_box.configure(state='disabled')

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Plant Detector Pro - V3.0 (Thước Đo Chuẩn)")
        self.root.geometry("1200x850")
        self.root.configure(bg="#1e1e1e")
        
        self.detector = PlantDetectorV2()
        self.image = None
        self.detections = []
        self.detailed_results = []
        
        # Biến cho thước đo chuẩn
        self.calib_mode = tk.BooleanVar(value=False)
        self.start_x = None
        self.start_y = None
        self.temp_line = None
        self.px_per_m = 100.0

        self.notebook = ttk.Notebook(self.root)
        self.tab1 = tk.Frame(self.notebook, bg="#1e1e1e")
        self.tab2 = tk.Frame(self.notebook, bg="#1e1e1e")
        self.notebook.add(self.tab1, text="  XỬ LÝ  ")
        self.notebook.add(self.tab2, text="  LỊCH SỬ DỮ LIỆU  ")
        self.notebook.pack(expand=True, fill="both")

        self.setup_tab1()
        self.setup_tab2()
        self.notebook.bind("<<NotebookTabChanged>>", lambda e: self.load_db_to_table())

    def setup_tab1(self):
        # Khu vực hiển thị ảnh (Dùng Canvas thay vì Label để vẽ)
        self.left_f = tk.Frame(self.tab1, bg="#2d2d2d", width=750, height=550)
        self.left_f.place(x=15, y=15)
        self.canvas = tk.Canvas(self.left_f, bg="#2d2d2d", width=730, height=530, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)

        # Điều khiển bên phải
        self.right_f = tk.Frame(self.tab1, bg="#2d2d2d", width=380, height=550)
        self.right_f.place(x=780, y=15)

        # Chế độ thước đo chuẩn
        tk.Label(self.right_f, text="CẤU HÌNH ĐỘ ĐO", bg="#2d2d2d", fg="#00ff88", font=("Segoe UI", 10, "bold")).pack(pady=(20,5))
        
        self.check_calib = tk.Checkbutton(self.right_f, text="Chế độ Vẽ Thước Chuẩn", variable=self.calib_mode, 
                                          bg="#2d2d2d", fg="white", selectcolor="#1e1e1e", command=self.toggle_calib)
        self.check_calib.pack(pady=5)

        tk.Label(self.right_f, text="1 Mét = (px)", bg="#2d2d2d", fg="white").pack()
        self.scale_ent = tk.Entry(self.right_f, justify='center', font=("Segoe UI", 12))
        self.scale_ent.insert(0, "100")
        self.scale_ent.pack(pady=5)

        ttk.Button(self.right_f, text="1. CHỌN ẢNH", command=self.load_image).pack(fill="x", padx=30, pady=5)
        ttk.Button(self.right_f, text="2. PHÁT HIỆN", command=self.detect).pack(fill="x", padx=30, pady=5)
        ttk.Button(self.right_f, text="3. DỰ ĐOÁN & LƯU", command=self.predict_and_show).pack(fill="x", padx=30, pady=5)
        
        self.listbox = tk.Listbox(self.right_f, bg="#3a3a3a", fg="#00ff88", font=("Consolas", 10))
        self.listbox.pack(fill="both", expand=True, padx=30, pady=20)

        self.log_box = tk.Text(self.tab1, height=10, bg="#1a1a1a", fg="#00ff88", state='disabled')
        self.log_box.place(x=15, y=580, width=1145)

    # --- Xử lý vẽ trên Canvas ---
    def toggle_calib(self):
        if self.calib_mode.get():
            self.scale_ent.config(state='disabled')
            log(self, "Chế độ Thước Chuẩn: Bật. Hãy kéo chuột trên ảnh để vẽ 1 đoạn thẳng mẫu.")
        else:
            self.scale_ent.config(state='normal')
            log(self, "Đã tắt chế độ vẽ. Bạn có thể nhập px thủ công.")

    def on_canvas_click(self, event):
        if self.calib_mode.get() and self.image is not None:
            self.start_x, self.start_y = event.x, event.y

    def on_canvas_drag(self, event):
        if self.calib_mode.get() and self.start_x is not None:
            if self.temp_line:
                self.canvas.delete(self.temp_line)
            self.temp_line = self.canvas.create_line(self.start_x, self.start_y, event.x, event.y, fill="red", width=3, dash=(4,4))

    def on_canvas_release(self, event):
        if self.calib_mode.get() and self.start_x is not None:
            # Tính độ dài pixel của đoạn vừa vẽ
            dist_px = math.sqrt((event.x - self.start_x)**2 + (event.y - self.start_y)**2)
            
            if dist_px > 5:
                # Hiện popup hỏi độ dài thực tế
                real_m = simpledialog.askfloat("Cài đặt thước đo", f"Đoạn thẳng này ({int(dist_px)} px) dài bao nhiêu mét thực tế?", initialvalue=1.0)
                if real_m:
                    self.px_per_m = dist_px / real_m
                    self.scale_ent.config(state='normal')
                    self.scale_ent.delete(0, tk.END)
                    self.scale_ent.insert(0, str(round(self.px_per_m, 2)))
                    self.scale_ent.config(state='disabled')
                    log(self, f"Đã cập nhật: 1 mét = {round(self.px_per_m, 2)} px")
            
            self.start_x = None
            if self.temp_line:
                self.canvas.delete(self.temp_line)

    def load_image(self):
        path = filedialog.askopenfilename()
        if path:
            self.image = cv2.imread(path)
            self.show_img(self.image)
            log(self, f"Đã nạp ảnh: {os.path.basename(path)}")

    def show_img(self, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        # Tính toán tỷ lệ để fit canvas 730x530
        ratio = min(730/w, 530/h)
        new_w, new_h = int(w*ratio), int(h*ratio)
        resized = cv2.resize(rgb, (new_w, new_h))
        
        self.tk_p = ImageTk.PhotoImage(Image.fromarray(resized))
        self.canvas.delete("all")
        self.canvas.create_image(365, 265, image=self.tk_p, anchor="center")

    def detect(self):
        if self.image is not None:
            threading.Thread(target=self._det_thread, daemon=True).start()

    def _det_thread(self):
        self.detections = self.detector.detect_trees(self.image)
        self.listbox.delete(0, tk.END)
        for d in self.detections: 
            self.listbox.insert(tk.END, f"ID {d['index']}: Vùng chọn {d['area']} px")
        self.show_img(self.detector.draw_results(self.image, self.detections))
        log(self, f"Tìm thấy {len(self.detections)} vật thể.")

    def predict_and_show(self):
        if not self.detections: return
        try: px_m = float(self.scale_ent.get())
        except: px_m = 100.0
        
        preds = []
        for d in self.detections:
            crop = self.image[d['box'][1]:d['box'][3], d['box'][0]:d['box'][2]]
            preds.append(self.detector.classify_crop(crop))
        
        self.detailed_results = self.detector.get_detailed_info(self.image, self.detections, preds, px_per_m=px_m)
        self.show_img(self.detector.draw_results(self.image, self.detections, preds))
        self.show_current_data()

    def show_current_data(self):
        win = tk.Toplevel(self.root)
        win.title("KẾT QUẢ")
        win.geometry("600x450")
        cols = ("STT", "Vật thể", "Rộng (m)", "Dài (m)")
        tree = ttk.Treeview(win, columns=cols, show='headings')
        for c in cols: 
            tree.heading(c, text=c)
            tree.column(c, width=120, anchor="center")
        tree.pack(fill="both", expand=True, padx=10, pady=10)
        for i in self.detailed_results:
            tree.insert("", tk.END, values=(i["STT"], i["Tên vật thể"], i["Rộng (m)"], i["Dài (m)"]))

        def save():
            if self.detector.save_to_sqlite(self.detailed_results):
                messagebox.showinfo("OK", "Đã lưu vào DB!")
                win.destroy()
        ttk.Button(win, text="XÁC NHẬN LƯU", command=save).pack(pady=10)

    # --- Phần Lịch sử (Giữ nguyên logic cũ nhưng cập nhật cột) ---
    def setup_tab2(self):
        self.paned = tk.PanedWindow(self.tab2, orient=tk.HORIZONTAL, bg="#1e1e1e")
        self.paned.pack(fill="both", expand=True)
        self.tree_f = tk.Frame(self.paned, bg="#2d2d2d")
        self.paned.add(self.tree_f, width=750)
        
        cols = ("ID", "Thời gian", "Vật thể", "Rộng (m)", "Dài (m)", "Ảnh")
        self.db_tree = ttk.Treeview(self.tree_f, columns=cols, show='headings')
        for c in cols: 
            self.db_tree.heading(c, text=c)
            self.db_tree.column(c, width=110, anchor="center")
        self.db_tree.pack(fill="both", expand=True)
        self.db_tree.bind("<<TreeviewSelect>>", self.on_row_select)

        ttk.Button(self.tree_f, text="XÓA DÒNG", command=self.delete_selected).pack(pady=5)
        
        self.prev_f = tk.Frame(self.paned, bg="#1a1a1a")
        self.paned.add(self.prev_f, width=430)
        self.prev_label = tk.Label(self.prev_f, bg="#1a1a1a")
        self.prev_label.pack(fill="both", expand=True)

    def load_db_to_table(self):
        for r in self.db_tree.get_children(): self.db_tree.delete(r)
        if os.path.exists("thongtin.db"):
            conn = sqlite3.connect("thongtin.db")
            query = "SELECT id, thoi_gian, ten_vat_the, rong_m, dai_m, image_path FROM thongtin ORDER BY id DESC"
            try:
                for r in conn.execute(query): self.db_tree.insert("", tk.END, values=r)
            except: pass
            conn.close()

    def on_row_select(self, e):
        sel = self.db_tree.selection()
        if sel:
            path = self.db_tree.item(sel[0])['values'][5]
            if os.path.exists(path):
                img = cv2.resize(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB), (400, 400))
                self.tk_prev = ImageTk.PhotoImage(Image.fromarray(img))
                self.prev_label.config(image=self.tk_prev)

    def delete_selected(self):
        sel = self.db_tree.selection()
        if sel and messagebox.askyesno("Xác nhận", "Xóa mục này?"):
            item = self.db_tree.item(sel[0])
            if self.detector.delete_record(item['values'][0], item['values'][5]):
                self.load_db_to_table()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
