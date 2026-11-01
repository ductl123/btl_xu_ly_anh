"""
gui.py - Giao diện đồ họa người dùng (GUI) cho hệ thống phát hiện & nhận dạng biển số xe
Sử dụng Tkinter (chuẩn Python) và Pillow để hiển thị trực quan:
- Chọn ảnh giao thông tùy ý từ máy tính hoặc danh sách có sẵn.
- Hiển thị song song ảnh gốc và ảnh kết quả có vẽ Bounding Box kèm số biển.
- Xem chi tiết từng bước trung gian (Grayscale, Gaussian Blur, Sobel-X, Otsu, Closing, Opening, CCL).
- Hiển thị danh sách các biển số nhận diện được (Biển Số Xe, Loại biển, Tọa độ, Aspect Ratio, Confidence Score).
- Nút bấm xuất file Excel (.xlsx), nút xử lý hàng loạt và nút xem các đồ thị so sánh thực nghiệm.
"""

import sys
import os
import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk

# Đảm bảo import được các module trong dự án
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import config
from main import process_single_image, process_batch
from compare import run_all_comparisons
from utils.file_utils import get_image_paths, create_run_folders, export_results_to_excel


class LicensePlateApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Hệ Thống Phát Hiện & Nhận Dạng Biển Số Xe - Digital Image Processing")
        self.root.geometry("1240x840")
        self.root.minsize(1050, 720)

        # Cấu hình màu sắc giao diện
        self.bg_color = "#F0F2F5"
        self.panel_color = "#FFFFFF"
        self.root.configure(bg=self.bg_color)

        self.current_image_path = None
        self.current_candidates = []
        self.output_dirs = create_run_folders(config.OUTPUT_DIR)

        self._setup_ui()
        self._load_default_sample()

    def _setup_ui(self):
        # 1. Header Toolbar
        header_frame = tk.Frame(self.root, bg="#1E293B", height=60)
        header_frame.pack(fill=tk.X, side=tk.TOP)

        title_lbl = tk.Label(
            header_frame,
            text="NHẬN DIỆN BIỂN SỐ XE TỪ ẢNH GIAO THÔNG (SOBEL - MORPHOLOGY - TEMPLATE MATCHING)",
            fg="#F8FAFC",
            bg="#1E293B",
            font=("Segoe UI", 12, "bold"),
            padx=15,
            pady=12
        )
        title_lbl.pack(side=tk.LEFT)

        btn_style = {"font": ("Segoe UI", 9, "bold"), "padx": 10, "pady": 5, "cursor": "hand2"}

        btn_open = tk.Button(
            header_frame, text="📁 Chọn Ảnh", bg="#3B82F6", fg="white",
            relief=tk.FLAT, command=self._on_choose_image, **btn_style
        )
        btn_open.pack(side=tk.RIGHT, padx=5, pady=10)

        btn_batch = tk.Button(
            header_frame, text="⚡ Xử Lý Hàng Loạt", bg="#10B981", fg="white",
            relief=tk.FLAT, command=self._on_batch_process, **btn_style
        )
        btn_batch.pack(side=tk.RIGHT, padx=5, pady=10)

        btn_excel = tk.Button(
            header_frame, text="📗 Xuất Excel (.xlsx)", bg="#059669", fg="white",
            relief=tk.FLAT, command=self._on_export_excel, **btn_style
        )
        btn_excel.pack(side=tk.RIGHT, padx=5, pady=10)

        btn_compare = tk.Button(
            header_frame, text="📊 So Sánh Thực Nghiệm", bg="#8B5CF6", fg="white",
            relief=tk.FLAT, command=self._on_run_comparisons, **btn_style
        )
        btn_compare.pack(side=tk.RIGHT, padx=5, pady=10)

        btn_folder = tk.Button(
            header_frame, text="📂 Thư Mục Output", bg="#64748B", fg="white",
            relief=tk.FLAT, command=self._open_output_folder, **btn_style
        )
        btn_folder.pack(side=tk.RIGHT, padx=5, pady=10)

        # 2. Main Content
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Cột Trái: Tab xem Ảnh Kết Quả & Các Bước Trung Gian
        left_frame = tk.Frame(main_paned, bg=self.panel_color, relief=tk.SOLID, bd=1)
        main_paned.add(left_frame, weight=3)

        self.tab_control = ttk.Notebook(left_frame)
        self.tab_control.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.tab_detected = tk.Frame(self.tab_control, bg="#1E1E1E")
        self.tab_steps = tk.Frame(self.tab_control, bg="#1E1E1E")
        self.tab_sobel = tk.Frame(self.tab_control, bg="#1E1E1E")
        self.tab_morph = tk.Frame(self.tab_control, bg="#1E1E1E")

        self.tab_control.add(self.tab_detected, text="🎯 Kết Quả Phát Hiện (BBox + Biển Số)")
        self.tab_control.add(self.tab_steps, text="🧩 Lưới 8 Bước Xử Lý")
        self.tab_control.add(self.tab_sobel, text="📐 Cạnh Dọc Sobel-X")
        self.tab_control.add(self.tab_morph, text="⬛ Nhị Phân & Closing")

        self.lbl_detected_img = tk.Label(self.tab_detected, bg="#1E1E1E")
        self.lbl_detected_img.pack(fill=tk.BOTH, expand=True)

        self.lbl_steps_img = tk.Label(self.tab_steps, bg="#1E1E1E")
        self.lbl_steps_img.pack(fill=tk.BOTH, expand=True)

        self.lbl_sobel_img = tk.Label(self.tab_sobel, bg="#1E1E1E")
        self.lbl_sobel_img.pack(fill=tk.BOTH, expand=True)

        self.lbl_morph_img = tk.Label(self.tab_morph, bg="#1E1E1E")
        self.lbl_morph_img.pack(fill=tk.BOTH, expand=True)

        # Cột Phải: Thông tin chi tiết & danh sách biển số phát hiện được
        right_frame = tk.Frame(main_paned, bg=self.panel_color, relief=tk.SOLID, bd=1, width=400)
        main_paned.add(right_frame, weight=1)

        info_title = tk.Label(
            right_frame, text="DANH SÁCH BIỂN SỐ NHẬN DIỆN",
            font=("Segoe UI", 11, "bold"), bg=self.panel_color, fg="#1E293B", pady=8
        )
        info_title.pack(fill=tk.X)

        self.tree = ttk.Treeview(
            right_frame,
            columns=("idx", "plate_text", "type", "ar", "score"),
            show="headings",
            height=6
        )
        self.tree.heading("idx", text="#")
        self.tree.heading("plate_text", text="Biển Số Xe")
        self.tree.heading("type", text="Loại Biển")
        self.tree.heading("ar", text="Tỷ Lệ w/h")
        self.tree.heading("score", text="Độ Tin Cậy")

        self.tree.column("idx", width=30, anchor="center")
        self.tree.column("plate_text", width=110, anchor="center")
        self.tree.column("type", width=80, anchor="center")
        self.tree.column("ar", width=65, anchor="center")
        self.tree.column("score", width=75, anchor="center")
        self.tree.pack(fill=tk.X, padx=10, pady=5)
        self.tree.bind("<<TreeviewSelect>>", self._on_select_plate)

        crop_lbl = tk.Label(
            right_frame, text="VÙNG BIỂN SỐ ĐÃ CẮT (CROPPED)",
            font=("Segoe UI", 10, "bold"), bg=self.panel_color, fg="#334155", pady=6
        )
        crop_lbl.pack(fill=tk.X)

        crop_container = tk.Frame(right_frame, bg="#E2E8F0", height=130)
        crop_container.pack(fill=tk.X, padx=15, pady=5)
        crop_container.pack_propagate(False)

        self.lbl_cropped_preview = tk.Label(crop_container, bg="#E2E8F0", text="Chưa chọn biển số")
        self.lbl_cropped_preview.pack(fill=tk.BOTH, expand=True)

        self.detail_text = tk.Text(right_frame, height=13, bg="#F8FAFC", fg="#0F172A", font=("Consolas", 9), relief=tk.FLAT)
        self.detail_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        # 3. Status Bar
        self.status_var = tk.StringVar(value="Sẵn sàng. Hãy chọn một ảnh để xử lý.")
        status_bar = tk.Label(
            self.root, textvariable=self.status_var,
            bd=1, relief=tk.SUNKEN, anchor=tk.W, bg="#E2E8F0", fg="#334155",
            font=("Segoe UI", 9), padx=10, pady=3
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _load_default_sample(self):
        samples = get_image_paths(config.INPUT_DIR)
        if samples:
            self._process_and_show(samples[0])

    def _on_choose_image(self):
        file_path = filedialog.askopenfilename(
            initialdir=str(config.INPUT_DIR),
            title="Chọn ảnh giao thông",
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")]
        )
        if file_path:
            self._process_and_show(Path(file_path))

    def _process_and_show(self, image_path: Path):
        self.current_image_path = image_path
        self.status_var.set(f"Đang xử lý ảnh: {image_path.name}...")
        self.root.update_idletasks()

        records = process_single_image(
            image_path,
            self.output_dirs,
            save_intermediate=True,
            show_plot=False
        )
        self.current_candidates = records

        # Cập nhật danh sách bảng
        for item in self.tree.get_children():
            self.tree.delete(item)

        detected_records = [r for r in records if r["plate_index"] > 0]

        for r in detected_records:
            type_str = "1 Dòng" if r["plate_type"] == "single_row" else "2 Dòng"
            plate_text = r.get("plate_text", "KXD")
            self.tree.insert(
                "", tk.END,
                values=(r["plate_index"], plate_text, type_str, f"{r['aspect_ratio']:.2f}", f"{r['confidence_score']:.2f}"),
                tags=(str(r["plate_index"]),)
            )

        # Hiển thị ảnh Detected lên giao diện
        stem = image_path.stem
        det_img_path = self.output_dirs["detected"] / f"{stem}_detected.png"
        self._display_image_on_label(det_img_path, self.lbl_detected_img)

        # Hiển thị ảnh Lưới 8 bước
        grid_img_path = self.output_dirs["intermediate"] / stem / "08_pipeline_summary_grid.png"
        self._display_image_on_label(grid_img_path, self.lbl_steps_img)

        # Hiển thị ảnh Sobel-X
        sobel_img_path = self.output_dirs["intermediate"] / stem / "03_sobel_x.png"
        self._display_image_on_label(sobel_img_path, self.lbl_sobel_img)

        # Hiển thị ảnh Closing
        closing_img_path = self.output_dirs["intermediate"] / stem / "05_morph_closing.png"
        self._display_image_on_label(closing_img_path, self.lbl_morph_img)

        # Chọn hàng đầu tiên nếu có
        children = self.tree.get_children()
        if children:
            self.tree.selection_set(children[0])
            self.tree.focus(children[0])
            self._on_select_plate(None)
        else:
            self.lbl_cropped_preview.configure(image="", text="Không phát hiện biển số phù hợp")
            self.detail_text.delete("1.0", tk.END)
            self.detail_text.insert(tk.END, "Không tìm thấy biển số nào thỏa mãn tiêu chuẩn kích thước và tỷ lệ.")

        self.status_var.set(f"Đã xử lý xong ảnh: {image_path.name} | Phát hiện {len(detected_records)} biển số.")

    def _on_select_plate(self, event):
        selected = self.tree.selection()
        if not selected:
            return
        item = self.tree.item(selected[0])
        idx = int(item["values"][0])

        matching = next((r for r in self.current_candidates if r["plate_index"] == idx), None)
        if matching:
            crop_path = Path(matching["cropped_image_path"])
            if crop_path.is_file():
                self._display_image_on_label(crop_path, self.lbl_cropped_preview, max_size=(320, 120))

            info_str = (
                f"--- THÔNG SỐ BIỂN SỐ #{idx} ---\n"
                f"★ BIỂN SỐ NHẬN DIỆN: {matching.get('plate_text', 'KXD')}\n"
                f"• Phân loại:         {matching['plate_type']}\n"
                f"• Điểm tin cậy BBox: {matching['confidence_score']:.3f}\n"
                f"• Độ tin cậy ký tự:  {matching.get('ocr_confidence', 0.0):.3f}\n"
                f"• Tọa độ gốc x,y:    ({matching['orig_x']}, {matching['orig_y']})\n"
                f"• Kích thước w,h:    {matching['orig_w']} x {matching['orig_h']} px\n"
                f"• Tỷ lệ w/h:         {matching['aspect_ratio']:.3f}\n"
                f"• Mật độ trắng:      {matching['white_density']:.3f}\n"
                f"• Số pixel cạnh:     {matching['pixel_area']} px\n"
                f"• File ảnh cắt:\n  {crop_path.name}\n"
            )
            self.detail_text.delete("1.0", tk.END)
            self.detail_text.insert(tk.END, info_str)

    def _display_image_on_label(self, img_path: Path, label_widget: tk.Label, max_size=(780, 700)):
        if not img_path.is_file():
            return
        try:
            im = Image.open(img_path)
            im.thumbnail(max_size, Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(im)
            label_widget.configure(image=photo, text="")
            label_widget.image = photo
        except Exception as e:
            label_widget.configure(text=f"Lỗi hiển thị ảnh: {e}")

    def _on_export_excel(self):
        if not self.current_candidates:
            messagebox.showwarning("Cảnh báo", "Chưa có kết quả nhận diện để xuất Excel!")
            return

        excel_path = self.output_dirs["main_results"] / "detection_results.xlsx"
        export_results_to_excel(self.current_candidates, excel_path)
        messagebox.showinfo("Thành công", f"Đã xuất file Excel kết quả thành công tại:\n{excel_path}")

    def _on_batch_process(self):
        images = get_image_paths(config.INPUT_DIR)
        if not images:
            messagebox.showwarning("Cảnh báo", "Thư mục input/images đang trống!")
            return

        confirm = messagebox.askyesno(
            "Xác nhận", f"Xử lý hàng loạt toàn bộ {len(images)} bức ảnh trong thư mục input?\n(Quá trình này có thể mất 20-30 giây)"
        )
        if not confirm:
            return

        def _worker():
            self.status_var.set(f"Đang xử lý hàng loạt {len(images)} ảnh...")
            process_batch(images, self.output_dirs, save_intermediate=True)
            self.status_var.set(f"Đã hoàn thành xử lý hàng loạt {len(images)} ảnh! Đã lưu file Excel tại output/results/.")
            messagebox.showinfo("Thành công", f"Đã xử lý xong {len(images)} bức ảnh!\nFile Excel (.xlsx) đã được cập nhật tại thư mục output/results/.")

        threading.Thread(target=_worker, daemon=True).start()

    def _on_run_comparisons(self):
        samples = get_image_paths(config.INPUT_DIR)
        target = self.current_image_path if self.current_image_path else (samples[0] if samples else None)
        if not target:
            messagebox.showwarning("Cảnh báo", "Chưa có ảnh để chạy so sánh!")
            return

        def _worker():
            self.status_var.set(f"Đang tạo đồ thị so sánh thực nghiệm...")
            run_all_comparisons(target, self.output_dirs["main_results"])
            self.status_var.set(f"Đã lưu các đồ thị so sánh thực nghiệm vào output/results/!")
            messagebox.showinfo("Hoàn tất", f"Đã xuất 4 biểu đồ so sánh chất lượng cao vào:\n{self.output_dirs['main_results']}")

        threading.Thread(target=_worker, daemon=True).start()

    def _open_output_folder(self):
        out_path = config.OUTPUT_DIR.resolve()
        if sys.platform == "win32":
            os.system(f'explorer "{out_path}"')
        else:
            messagebox.showinfo("Thư mục", str(out_path))


def launch_gui():
    root = tk.Tk()
    app = LicensePlateApp(root)
    root.mainloop()


if __name__ == "__main__":
    launch_gui()
