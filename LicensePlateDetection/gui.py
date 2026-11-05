"""
gui.py - Giao diện đồ họa người dùng (GUI) cho hệ thống phát hiện & nhận dạng biển số xe
Sử dụng Tkinter (chuẩn Python) và Pillow để hiển thị trực quan:
- Chọn ảnh giao thông tùy ý từ máy tính hoặc danh sách có sẵn (chuyển ảnh tức thì, xử lý đa luồng ngầm không đơ giật UI).
- Hiển thị song song ảnh gốc và ảnh kết quả có vẽ Bounding Box kèm số biển.
- Xem chi tiết từng bước trung gian (Grayscale, Gaussian Blur, Sobel-X, Otsu, Closing, Opening, CCL).
- Hiển thị danh sách các biển số nhận diện được (Biển Số Xe, Loại biển, Tọa độ, Aspect Ratio, Confidence Score).
- Hỗ trợ xử lý hàng loạt chuyên nghiệp với thanh tiến trình trực quan, chạy nền không đơ, không bị thoát ứng dụng.
- Nút bấm xuất file Excel (.xlsx), nút xem các đồ thị so sánh thực nghiệm.
"""

import sys
import os
import threading
import time
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import numpy as np
from PIL import Image, ImageTk

# Đảm bảo import được các module trong dự án
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import config
from main import process_single_image, process_batch
from compare import run_all_comparisons
from utils.file_utils import get_image_paths, create_run_folders, export_results_to_excel, export_results_to_csv


class LicensePlateApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Hệ Thống Phát Hiện & Nhận Dạng Biển Số Xe - Digital Image Processing")
        self.root.geometry("1260x860")
        self.root.minsize(1050, 720)

        # Cấu hình màu sắc giao diện
        self.bg_color = "#F0F2F5"
        self.panel_color = "#FFFFFF"
        self.root.configure(bg=self.bg_color)

        self.current_image_path = None
        self.current_candidates = []
        self.output_dirs = create_run_folders(config.OUTPUT_DIR)
        self.is_busy = False

        self._setup_ui()
        self._load_default_sample()

    def _setup_ui(self):
        # 1. Header Toolbar
        header_frame = tk.Frame(self.root, bg="#1E293B", height=60)
        header_frame.pack(fill=tk.X, side=tk.TOP)

        title_lbl = tk.Label(
            header_frame,
            text="NHẬN DIỆN BIỂN SỐ: XỬ LÝ ẢNH + OCR HỌC SÂU",
            fg="#F8FAFC",
            bg="#1E293B",
            font=("Segoe UI", 12, "bold"),
            padx=15,
            pady=12
        )
        title_lbl.pack(side=tk.LEFT)

        btn_style = {"font": ("Segoe UI", 9, "bold"), "padx": 10, "pady": 5, "cursor": "hand2"}

        self.btn_open = tk.Button(
            header_frame, text="📁 Chọn Ảnh", bg="#3B82F6", fg="white",
            relief=tk.FLAT, command=self._on_choose_image, **btn_style
        )
        self.btn_open.pack(side=tk.RIGHT, padx=5, pady=10)

        self.btn_batch = tk.Button(
            header_frame, text="⚡ Xử Lý Hàng Loạt", bg="#10B981", fg="white",
            relief=tk.FLAT, command=self._on_batch_process, **btn_style
        )
        self.btn_batch.pack(side=tk.RIGHT, padx=5, pady=10)

        self.btn_excel = tk.Button(
            header_frame, text="📗 Xuất Excel (.xlsx)", bg="#059669", fg="white",
            relief=tk.FLAT, command=self._on_export_excel, **btn_style
        )
        self.btn_excel.pack(side=tk.RIGHT, padx=5, pady=10)

        self.btn_compare = tk.Button(
            header_frame, text="📊 So Sánh Thực Nghiệm", bg="#8B5CF6", fg="white",
            relief=tk.FLAT, command=self._on_run_comparisons, **btn_style
        )
        self.btn_compare.pack(side=tk.RIGHT, padx=5, pady=10)

        self.btn_folder = tk.Button(
            header_frame, text="📂 Thư Mục Output", bg="#64748B", fg="white",
            relief=tk.FLAT, command=self._open_output_folder, **btn_style
        )
        self.btn_folder.pack(side=tk.RIGHT, padx=5, pady=10)

        learning_bar = tk.Frame(self.root,bg="#E2E8F0")
        learning_bar.pack(fill=tk.X)
        tk.Label(learning_bar,text="Bộ đọc:",bg="#E2E8F0").pack(side=tk.LEFT,padx=(12,4))
        self.reader_choice = tk.StringVar(value="Học sâu CRNN" if config.OCR_BACKEND=='crnn' else "Truyền thống")
        self.reader_combo = ttk.Combobox(learning_bar,textvariable=self.reader_choice,
                                       values=["Học sâu CRNN","Truyền thống"],state="readonly",width=22)
        self.reader_combo.pack(side=tk.LEFT,padx=4,pady=5)
        self.reader_combo.bind('<<ComboboxSelected>>',lambda _:setattr(config,'OCR_BACKEND','crnn' if self.reader_choice.get()=="Học sâu CRNN" else 'template'))
        self.btn_learning = ttk.Button(learning_bar,text="Gán nhãn & Huấn luyện",command=self._open_learning)
        self.btn_learning.pack(side=tk.LEFT,padx=12)
        self.restore_choice = tk.BooleanVar(value=config.USE_WIENER_RESTORATION)
        self.restore_check = ttk.Checkbutton(learning_bar,text="Thử khôi phục Wiener ảnh mờ",
            variable=self.restore_choice,command=lambda:setattr(config,'USE_WIENER_RESTORATION',self.restore_choice.get()))
        self.restore_check.pack(side=tk.LEFT,padx=10)

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
        right_frame = tk.Frame(main_paned, bg=self.panel_color, relief=tk.SOLID, bd=1, width=420)
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
        self.tree.heading("score", text="Điểm Vùng Biển")

        self.tree.column("idx", width=30, anchor="center")
        self.tree.column("plate_text", width=120, anchor="center")
        self.tree.column("type", width=75, anchor="center")
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

    def _set_buttons_state(self, enabled: bool):
        """Khóa hoặc mở lại các nút thao tác chính khi đang xử lý nền."""
        state = tk.NORMAL if enabled else tk.DISABLED
        self.btn_open.config(state=state)
        self.btn_batch.config(state=state)
        self.btn_excel.config(state=state)
        self.btn_compare.config(state=state)
        self.btn_learning.config(state=state)
        self.reader_combo.config(state='readonly' if enabled else 'disabled')
        self.restore_check.config(state=state)

    def _open_learning(self):
        if self.is_busy:return
        if getattr(self,'training_window',None) is not None and self.training_window.winfo_exists():
            self.training_window.lift();return
        from learning.training_ui import TrainingWindow
        self.training_window=TrainingWindow(self)

    def _load_default_sample(self):
        """Khởi động hiển thị ảnh mẫu đầu tiên nhanh chóng."""
        samples = get_image_paths(config.INPUT_DIR)
        if samples:
            self._start_processing_image(samples[0])

    def _on_choose_image(self):
        if self.is_busy:
            messagebox.showinfo("Thông báo", "Hệ thống đang bận xử lý, vui lòng đợi trong giây lát!")
            return

        file_path = filedialog.askopenfilename(
            initialdir=str(config.INPUT_DIR),
            title="Chọn ảnh giao thông",
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")]
        )
        if file_path:
            self._start_processing_image(Path(file_path))

    def _start_processing_image(self, image_path: Path):
        """
        TỐI ƯU HÓA TỐC ĐỘ:
        1. Hiển thị ảnh gốc đã chọn NGAY LẬP TỨC (< 30ms) lên giao diện.
        2. Chạy quy trình nhận diện nặng ở luồng phụ (Thread) ngầm.
        3. Cập nhật kết quả BBox & ký tự lên giao diện an toàn khi hoàn thành.
        """
        self.current_image_path = image_path
        self.is_busy = True
        self._set_buttons_state(False)

        # 1. Hiển thị ngay ảnh gốc đã chọn để người dùng không phải chờ đợi
        self._display_image_on_label(image_path, self.lbl_detected_img)
        self.tab_control.select(self.tab_detected)

        # Xóa bảng cũ và hiển thị thông báo đang phân tích
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.tree.insert("", tk.END, values=("...", "Đang nhận diện...", "...", "...", "..."))

        self.lbl_cropped_preview.configure(image="", text="Đang phân tích...")
        self.detail_text.delete("1.0", tk.END)
        self.detail_text.insert(tk.END, f"Đang áp dụng quy trình DIP trên ảnh {image_path.name}...\n- Gaussian Blur\n- Sobel-X Edge\n- Otsu Threshold\n- Morphological Closing & Opening\n- BFS Connected Components\n- Template Matching OCR")

        self.status_var.set(f"⏳ Đang xử lý ảnh: {image_path.name}... Vui lòng đợi trong giây lát.")

        # 2. Đẩy công việc nặng sang luồng phụ
        def _worker():
            try:
                records = process_single_image(
                    image_path,
                    self.output_dirs,
                    save_intermediate=True,
                    show_plot=False
                )
            except Exception as e:
                print(f"[!] Lỗi khi xử lý ảnh {image_path}: {e}")
                records = []
            
            # Đưa kết quả về luồng chính an toàn qua root.after
            self.root.after(0, lambda: self._apply_single_results(image_path, records))

        threading.Thread(target=_worker, daemon=True).start()

    def _apply_single_results(self, image_path: Path, records: list):
        """Cập nhật toàn bộ kết quả lên giao diện trên Main Thread."""
        self.current_candidates = records
        self.is_busy = False
        self._set_buttons_state(True)

        # Cập nhật danh sách bảng
        for item in self.tree.get_children():
            self.tree.delete(item)

        detected_records = [r for r in records if r["plate_index"] > 0]

        for r in detected_records:
            type_str = "1 Dòng" if r["plate_type"] == "single_row" else "2 Dòng"
            plate_text = r.get("plate_text", "KXD")
            if r.get("recognition_status") == "unreadable":
                plate_text = "Chưa đọc rõ"
            elif r.get("needs_review", True):
                plate_text += " (?)"
            self.tree.insert(
                "", tk.END,
                values=(r["plate_index"], plate_text, type_str, f"{r['aspect_ratio']:.2f}", f"{r['confidence_score']:.2f}"),
                tags=(str(r["plate_index"]),)
            )

        # Hiển thị ảnh Detected (có BBox + Text) lên giao diện
        stem = image_path.stem
        det_img_path = self.output_dirs["detected"] / f"{stem}_detected.png"
        if det_img_path.is_file():
            self._display_image_on_label(det_img_path, self.lbl_detected_img)
        else:
            self._display_image_on_label(image_path, self.lbl_detected_img)

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
            self.detail_text.insert(tk.END, "Chưa tìm thấy vùng biển có đủ bằng chứng trong ảnh này.")

        self.status_var.set(f"✅ Đã xử lý xong ảnh: {image_path.name} | Phát hiện {len(detected_records)} biển số.")

    def _on_select_plate(self, event):
        selected = self.tree.selection()
        if not selected:
            return
        item = self.tree.item(selected[0])
        val = item["values"][0]
        try:
            idx = int(val)
        except (ValueError, TypeError):
            return

        matching = next((r for r in self.current_candidates if r["plate_index"] == idx), None)
        if matching:
            crop_path = Path(matching["cropped_image_path"])
            if crop_path.is_file():
                self._display_image_on_label(crop_path, self.lbl_cropped_preview, max_size=(320, 120))

            review = matching.get("needs_review", False)
            review_status = "Cần đối chiếu lại ký tự mơ hồ" if review else "Chưa thấy ký tự có điểm sát nhau"
            if matching.get("recognition_status") == "unreadable":
                review_status = "Đã tìm thấy vùng biển nhưng chưa đọc rõ ký tự"
            positions = ", ".join(str(i+1) for i in matching.get("uncertain_positions", []))
            ocr_score = f"{matching.get('ocr_confidence', 0.0):.3f}"
            if matching.get('recognition_status') == 'confirmed':
                review_status = 'Đã đối chiếu với nhãn xác nhận của ảnh cắt trùng khớp'
                ocr_score = '— (dùng nhãn đã xác nhận)'
            info_str = (
                f"• Kiểm tra kết quả: {review_status}\n"
                f"• Vị trí cần xem:   {positions or '—'}\n"
                f"--- THÔNG SỐ BIỂN SỐ #{idx} ---\n"
                f"★ BIỂN SỐ NHẬN DIỆN: {matching.get('plate_text', 'KXD')}\n"
                f"• Phân loại:         {matching['plate_type']}\n"
                f"• Điểm vùng biển: {matching['confidence_score']:.3f}\n"
                f"• Điểm khớp ký tự:   {ocr_score}\n"
                f"• Bộ đọc:             {matching.get('recognizer', 'unknown')}\n"
                f"• Lý do kiểm tra:     {'; '.join(matching.get('uncertainty_reasons', [])) or '—'}\n"
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
            with Image.open(img_path) as im:
                im_copy = im.copy()
            im_copy.thumbnail(max_size, Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(im_copy)
            label_widget.configure(image=photo, text="")
            label_widget.image = photo  # Giữ tham chiếu tránh bị GC xóa
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
        """
        XỬ LÝ HÀNG LOẠT KHÔNG ĐƠ, KHÔNG CRASH:
        - Sử dụng cửa sổ tiến trình chuyên biệt Toplevel.
        - Chạy luồng phụ ngầm, cập nhật % tiến độ từng ảnh lên Main Thread.
        - Cho phép người dùng dừng sớm nếu muốn.
        - Lưu kết quả đầy đủ ra Excel và CSV khi hoàn tất.
        """
        if self.is_busy:
            messagebox.showinfo("Thông báo", "Hệ thống đang bận xử lý, vui lòng đợi!")
            return

        images = get_image_paths(config.INPUT_DIR)
        if not images:
            messagebox.showwarning("Cảnh báo", "Thư mục input/images đang trống!")
            return

        confirm = messagebox.askyesno(
            "Xác nhận xử lý hàng loạt",
            f"Tìm thấy {len(images)} bức ảnh trong thư mục input.\nBạn có muốn bắt đầu xử lý hàng loạt không?"
        )
        if not confirm:
            return

        total_images = len(images)
        self.is_busy = True
        self._set_buttons_state(False)

        # Tạo Cửa sổ Tiến trình Xử lý hàng loạt (Progress Dialog)
        prog_win = tk.Toplevel(self.root)
        prog_win.title("Tiến Trình Xử Lý Hàng Loạt Biển Số Xe")
        prog_win.geometry("620x420")
        prog_win.minsize(500, 350)
        prog_win.transient(self.root)
        prog_win.grab_set()

        lbl_batch_title = tk.Label(
            prog_win, text=f"ĐANG XỬ LÝ HÀNG LOẠT ({total_images} ẢNH)",
            font=("Segoe UI", 11, "bold"), fg="#1E293B", pady=10
        )
        lbl_batch_title.pack(fill=tk.X)

        status_lbl_var = tk.StringVar(value=f"Đang chuẩn bị xử lý...")
        lbl_current = tk.Label(prog_win, textvariable=status_lbl_var, font=("Segoe UI", 9), fg="#475569")
        lbl_current.pack(fill=tk.X, padx=15, pady=2)

        pb_var = tk.DoubleVar(value=0.0)
        pb = ttk.Progressbar(prog_win, variable=pb_var, maximum=total_images, mode='determinate')
        pb.pack(fill=tk.X, padx=20, pady=8)

        lbl_percent_var = tk.StringVar(value=f"0% (0/{total_images})")
        lbl_percent = tk.Label(prog_win, textvariable=lbl_percent_var, font=("Segoe UI", 9, "bold"), fg="#2563EB")
        lbl_percent.pack()

        # Khung nhật ký ghi lại các biển số phát hiện được trong thời gian thực
        log_box = scrolledtext.ScrolledText(prog_win, height=10, font=("Consolas", 9), bg="#F8FAFC", fg="#0F172A")
        log_box.pack(fill=tk.BOTH, expand=True, padx=15, pady=8)

        cancel_flag = threading.Event()

        def _on_cancel():
            if messagebox.askyesno("Dừng lại", "Bạn có chắc muốn dừng xử lý hàng loạt sau bức ảnh hiện tại?"):
                cancel_flag.set()
                btn_cancel.config(state=tk.DISABLED, text="Đang dừng...")

        btn_cancel = tk.Button(prog_win, text="Dừng lại", bg="#EF4444", fg="white", font=("Segoe UI", 9, "bold"), relief=tk.FLAT, command=_on_cancel, padx=15, pady=4)
        btn_cancel.pack(side=tk.BOTTOM, pady=10)

        def _batch_worker():
            all_records = []
            start_time = time.time()

            for i, img_p in enumerate(images, start=1):
                if cancel_flag.is_set():
                    def _log_cancel():
                        log_box.insert(tk.END, f"\n[!] Người dùng đã yêu cầu dừng lại tại ảnh {i-1}/{total_images}.\n")
                        log_box.see(tk.END)
                    self.root.after(0, _log_cancel)
                    break

                # Cập nhật nhãn trạng thái trước khi xử lý
                def _pre_update(curr=i, name=img_p.name):
                    status_lbl_var.set(f"Đang phân tích [{curr}/{total_images}]: {name}")
                self.root.after(0, _pre_update)

                try:
                    records = process_single_image(img_p, self.output_dirs, save_intermediate=False, show_plot=False)
                    all_records.extend(records)
                except Exception as e:
                    records = []
                    print(f"[!] Lỗi xử lý ảnh {img_p.name}: {e}")

                # Cập nhật tiến độ sau khi hoàn thành ảnh
                def _post_update(curr=i, name=img_p.name, recs=records):
                    pb_var.set(curr)
                    pct = int(round(curr / total_images * 100))
                    lbl_percent_var.set(f"{pct}% ({curr}/{total_images})")

                    plates = [r.get('plate_text', 'KXD') for r in recs if r.get('plate_index', 0) > 0]
                    plates_str = ", ".join(plates) if plates else "Không có biển số"
                    log_box.insert(tk.END, f"[{curr:02d}/{total_images}] {name[:22]:<22} -> Biển: {plates_str}\n")
                    log_box.see(tk.END)
                self.root.after(0, _post_update)

            # Xuất kết quả tổng hợp ra CSV & Excel
            if all_records:
                csv_path = self.output_dirs["main_results"] / "detection_results.csv"
                xlsx_path = self.output_dirs["main_results"] / "detection_results.xlsx"
                export_results_to_csv(all_records, csv_path)
                export_results_to_excel(all_records, xlsx_path)

            elapsed = time.time() - start_time

            # Hoàn tất tiến trình an toàn trên Main Thread
            def _on_done():
                self.is_busy = False
                self._set_buttons_state(True)
                btn_cancel.config(text="Đóng", bg="#10B981", state=tk.NORMAL, command=prog_win.destroy)
                lbl_batch_title.config(text="HOÀN TẤT XỬ LÝ HÀNG LOẠT!", fg="#059669")
                status_lbl_var.set(f"Đã xử lý xong trong {elapsed:.1f} giây. Kết quả đã lưu vào Excel!")
                self.status_var.set(f"✅ Đã xử lý xong hàng loạt! File Excel tại: {self.output_dirs['main_results'] / 'detection_results.xlsx'}")

                # Tải ảnh cuối cùng lên giao diện chính
                if images:
                    self._start_processing_image(images[0])

                messagebox.showinfo(
                    "Hoàn tất xử lý hàng loạt",
                    f"Đã xử lý xong các bức ảnh trong {elapsed:.1f} giây!\n\nFile kết quả chi tiết đã được cập nhật tại:\n{self.output_dirs['main_results'] / 'detection_results.xlsx'}"
                )

            self.root.after(0, _on_done)

        threading.Thread(target=_batch_worker, daemon=True).start()

    def _on_run_comparisons(self):
        if self.is_busy:
            messagebox.showinfo("Thông báo", "Hệ thống đang bận xử lý tác vụ khác!")
            return

        samples = get_image_paths(config.INPUT_DIR)
        target = self.current_image_path if self.current_image_path else (samples[0] if samples else None)
        if not target:
            messagebox.showwarning("Cảnh báo", "Chưa có ảnh để chạy so sánh!")
            return

        self.is_busy = True
        self._set_buttons_state(False)
        self.status_var.set("⏳ Đang tạo 4 biểu đồ so sánh thực nghiệm DIP... Vui lòng đợi trong giây lát.")

        def _worker():
            try:
                run_all_comparisons(target, self.output_dirs["main_results"])
                success = True
                err_msg = ""
            except Exception as e:
                success = False
                err_msg = str(e)

            def _done():
                self.is_busy = False
                self._set_buttons_state(True)
                if success:
                    self.status_var.set("✅ Đã lưu các đồ thị so sánh thực nghiệm vào output/results/!")
                    messagebox.showinfo("Hoàn tất", f"Đã xuất 4 biểu đồ so sánh chất lượng cao vào:\n{self.output_dirs['main_results']}")
                else:
                    self.status_var.set(f"❌ Lỗi khi chạy so sánh: {err_msg}")
                    messagebox.showerror("Lỗi", f"Không thể tạo đồ thị so sánh:\n{err_msg}")

            self.root.after(0, _done)

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
