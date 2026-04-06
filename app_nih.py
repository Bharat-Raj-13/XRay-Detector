import customtkinter as ctk
from tkinter import filedialog, messagebox
import tensorflow as tf
import numpy as np
from PIL import Image
import threading
import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.units import inch

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Load both models
print("Loading models...")
model_quick = tf.keras.models.load_model('model/xray_model.h5')
model_full = tf.keras.models.load_model('model/xray_nih_model.h5')
LABELS = list(np.load('model/labels.npy', allow_pickle=True))
print("Models loaded!")

RECOMMENDATIONS = {
    'Atelectasis': [
        "🔴  Consult a pulmonologist immediately",
        "🔴  Deep breathing exercises recommended",
        "🔴  Avoid smoking and air pollutants",
        "🔴  Physical therapy may be required",
        "🔴  Follow up chest X-ray after treatment",
    ],
    'Cardiomegaly': [
        "🔴  Consult a cardiologist urgently",
        "🔴  Monitor blood pressure daily",
        "🔴  Reduce salt intake immediately",
        "🔴  Avoid heavy physical activity",
        "🔴  Take prescribed heart medications",
    ],
    'Effusion': [
        "🔴  Seek emergency medical attention",
        "🔴  Avoid physical exertion",
        "🔴  Monitor breathing difficulty closely",
        "🔴  Drainage procedure may be needed",
        "🔴  Follow up with specialist urgently",
    ],
    'Infiltration': [
        "🔴  Consult a doctor immediately",
        "🔴  Get blood culture test done",
        "🔴  Take prescribed antibiotics",
        "🔴  Rest completely and stay hydrated",
        "🔴  Monitor oxygen levels regularly",
    ],
    'Mass': [
        "🔴  Consult an oncologist immediately",
        "🔴  CT scan or MRI recommended",
        "🔴  Do not panic — not always cancer",
        "🔴  Biopsy may be required",
        "🔴  Follow up every 3-6 months",
    ],
    'Nodule': [
        "🔴  Consult a pulmonologist",
        "🔴  CT scan recommended for clarity",
        "🔴  Monitor size change over time",
        "🔴  Avoid smoking immediately",
        "🔴  Regular follow-up required",
    ],
    'Pneumonia': [
        "🔴  Consult a doctor immediately",
        "🔴  Get a blood test done",
        "🔴  Take prescribed antibiotics",
        "🔴  Stay hydrated and rest completely",
        "🔴  Monitor oxygen levels regularly",
    ],
    'Pneumothorax': [
        "🔴  EMERGENCY — Go to hospital now",
        "🔴  Do not delay treatment",
        "🔴  Avoid flying or diving",
        "🔴  Chest tube may be required",
        "🔴  Call emergency services immediately",
    ],
    'Consolidation': [
        "🔴  Consult a doctor urgently",
        "🔴  Antibiotic treatment likely needed",
        "🔴  Rest and stay hydrated",
        "🔴  Monitor breathing carefully",
        "🔴  Follow up X-ray after treatment",
    ],
    'Edema': [
        "🔴  Consult a cardiologist urgently",
        "🔴  Reduce fluid and salt intake",
        "🔴  Elevate legs when resting",
        "🔴  Take prescribed diuretics",
        "🔴  Monitor weight daily",
    ],
    'Emphysema': [
        "🔴  Stop smoking immediately",
        "🔴  Consult a pulmonologist",
        "🔴  Use prescribed inhaler regularly",
        "🔴  Avoid air pollution and dust",
        "🔴  Pulmonary rehabilitation recommended",
    ],
    'Fibrosis': [
        "🔴  Consult a pulmonologist urgently",
        "🔴  Avoid lung irritants and dust",
        "🔴  Oxygen therapy may be needed",
        "🔴  Anti-fibrotic medication available",
        "🔴  Lung transplant evaluation possible",
    ],
    'Pleural_Thickening': [
        "🔴  Consult a pulmonologist",
        "🔴  Avoid asbestos exposure",
        "🔴  Regular breathing tests needed",
        "🔴  Monitor lung function closely",
        "🔴  Anti-inflammatory treatment possible",
    ],
    'Hernia': [
        "🔴  Consult a surgeon immediately",
        "🔴  Avoid heavy lifting completely",
        "🔴  Surgical repair may be needed",
        "🔴  Maintain healthy body weight",
        "🔴  Do not delay treatment",
    ],
    'No Finding': [
        "🟢  Your X-ray appears normal",
        "🟢  Maintain a healthy lifestyle",
        "🟢  Annual checkup recommended",
        "🟢  Exercise regularly — 30 mins daily",
        "🟢  Avoid smoking and alcohol",
    ],
    'PNEUMONIA': [
        "🔴  Consult a doctor immediately",
        "🔴  Get a complete blood count test",
        "🔴  Take prescribed antibiotics",
        "🔴  Stay hydrated and rest completely",
        "🔴  Monitor oxygen levels regularly",
        "🔴  Avoid cold environments",
    ],
    'NORMAL': [
        "🟢  Your X-ray appears normal",
        "🟢  Maintain a healthy lifestyle",
        "🟢  Annual checkup recommended",
        "🟢  Exercise regularly — 30 mins daily",
        "🟢  Avoid smoking and alcohol",
        "🟢  Stay hydrated and eat well",
    ]
}

BG_PRIMARY = "#080810"
BG_SIDEBAR = "#0d0d18"
BG_CARD = "#13131f"
BG_HOVER = "#1a1a28"
BG_INPUT = "#1e1e2e"
ACCENT_BLUE = "#3b82f6"
ACCENT_RED = "#ef4444"
ACCENT_GREEN = "#22c55e"
TEXT_PRIMARY = "#f1f5f9"
TEXT_SECONDARY = "#64748b"
BORDER = "#1e1e2e"

history_log = []

class XRayApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("XRay AI — Advanced Medical Detector")
        self.geometry("1200x800")
        self.resizable(False, False)
        self.configure(fg_color=BG_PRIMARY)
        self.current_image_path = None
        self.current_results = None
        self.current_mode = None
        self.build_ui()

    def build_ui(self):
        self.main_frame = ctk.CTkFrame(self, fg_color=BG_PRIMARY)
        self.main_frame.pack(fill="both", expand=True)

        # Sidebar
        self.sidebar = ctk.CTkFrame(self.main_frame, width=220,
            fg_color=BG_SIDEBAR, corner_radius=0)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        logo_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent", height=80)
        logo_frame.pack(fill="x")
        logo_frame.pack_propagate(False)

        icon = ctk.CTkFrame(logo_frame, fg_color="#1d4ed8",
            width=42, height=42, corner_radius=10)
        icon.place(x=16, y=19)
        ctk.CTkLabel(icon, text="☩", font=ctk.CTkFont(size=20),
            text_color="white").place(relx=0.5, rely=0.5, anchor="center")

        ctk.CTkLabel(logo_frame, text="XRay AI",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=TEXT_PRIMARY).place(x=68, y=20)
        ctk.CTkLabel(logo_frame, text="Medical Detector",
            font=ctk.CTkFont(size=11),
            text_color=TEXT_SECONDARY).place(x=68, y=44)

        ctk.CTkFrame(self.sidebar, height=1,
            fg_color=BORDER).pack(fill="x", padx=16, pady=4)

        ctk.CTkLabel(self.sidebar, text="NAVIGATION",
            font=ctk.CTkFont(size=10),
            text_color=TEXT_SECONDARY).pack(anchor="w", padx=20, pady=(12,6))

        self.btn_detector = self.nav_button("  Detector", "detector", "🔬")
        self.btn_history = self.nav_button("  History", "history", "📋")

        ctk.CTkFrame(self.sidebar, height=1,
            fg_color=BORDER).pack(fill="x", padx=16, pady=12)

        ctk.CTkLabel(self.sidebar, text="MODELS",
            font=ctk.CTkFont(size=10),
            text_color=TEXT_SECONDARY).pack(anchor="w", padx=20, pady=(0,8))

        # Quick scan model info
        quick_card = ctk.CTkFrame(self.sidebar, fg_color=BG_HOVER, corner_radius=8)
        quick_card.pack(fill="x", padx=12, pady=3)
        ctk.CTkLabel(quick_card, text="⚡ Quick Scan",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=ACCENT_BLUE).pack(anchor="w", padx=10, pady=(8,2))
        ctk.CTkLabel(quick_card, text="VGG16 • 2 Classes • 95% acc",
            font=ctk.CTkFont(size=10),
            text_color=TEXT_SECONDARY).pack(anchor="w", padx=10, pady=(0,8))

        # Full scan model info
        full_card = ctk.CTkFrame(self.sidebar, fg_color=BG_HOVER, corner_radius=8)
        full_card.pack(fill="x", padx=12, pady=3)
        ctk.CTkLabel(full_card, text="🔬 Full Scan",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=ACCENT_GREEN).pack(anchor="w", padx=10, pady=(8,2))
        ctk.CTkLabel(full_card, text="DenseNet121 • 14 Classes • 54% acc",
            font=ctk.CTkFont(size=10),
            text_color=TEXT_SECONDARY).pack(anchor="w", padx=10, pady=(0,8))

        status_frame = ctk.CTkFrame(self.sidebar, fg_color=BG_HOVER, corner_radius=8)
        status_frame.pack(fill="x", padx=16, pady=16, side="bottom")
        ctk.CTkLabel(status_frame, text="🟢  Both Models Ready",
            font=ctk.CTkFont(size=12),
            text_color=ACCENT_GREEN).pack(pady=10)

        # Content
        self.content = ctk.CTkFrame(self.main_frame, fg_color=BG_PRIMARY)
        self.content.pack(side="right", fill="both", expand=True)

        self.detector_frame = ctk.CTkFrame(self.content, fg_color=BG_PRIMARY)
        self.history_frame_outer = ctk.CTkFrame(self.content, fg_color=BG_PRIMARY)

        self.build_detector_page()
        self.build_history_page()
        self.show_page("detector")

    def nav_button(self, text, page, icon):
        btn = ctk.CTkButton(self.sidebar, text=f"{icon}{text}",
            font=ctk.CTkFont(size=13),
            fg_color="transparent", hover_color=BG_HOVER,
            text_color=TEXT_SECONDARY, anchor="w", height=40,
            corner_radius=8, command=lambda: self.show_page(page))
        btn.pack(fill="x", padx=10, pady=2)
        return btn

    def show_page(self, page):
        if page == "detector":
            self.history_frame_outer.pack_forget()
            self.detector_frame.pack(fill="both", expand=True)
            self.btn_detector.configure(fg_color=BG_HOVER, text_color=TEXT_PRIMARY)
            self.btn_history.configure(fg_color="transparent", text_color=TEXT_SECONDARY)
        else:
            self.detector_frame.pack_forget()
            self.history_frame_outer.pack(fill="both", expand=True)
            self.btn_history.configure(fg_color=BG_HOVER, text_color=TEXT_PRIMARY)
            self.btn_detector.configure(fg_color="transparent", text_color=TEXT_SECONDARY)

    def build_detector_page(self):
        frame = self.detector_frame

        topbar = ctk.CTkFrame(frame, fg_color=BG_CARD, height=55, corner_radius=0)
        topbar.pack(fill="x")
        topbar.pack_propagate(False)
        ctk.CTkLabel(topbar, text="Chest X-Ray Detector",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=TEXT_PRIMARY).pack(side="left", padx=20, pady=15)
        self.status = ctk.CTkLabel(topbar,
            text="Ready — Upload an X-ray to begin",
            font=ctk.CTkFont(size=11), text_color=TEXT_SECONDARY)
        self.status.pack(side="right", padx=20)

        # Patient form
        form = ctk.CTkFrame(frame, fg_color=BG_CARD, corner_radius=12)
        form.pack(fill="x", padx=15, pady=(12,8))
        ctk.CTkLabel(form, text="Patient Details",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=TEXT_SECONDARY).pack(anchor="w", padx=16, pady=(10,6))

        fields = ctk.CTkFrame(form, fg_color="transparent")
        fields.pack(fill="x", padx=16, pady=(0,10))

        ctk.CTkLabel(fields, text="Name:", font=ctk.CTkFont(size=12),
            text_color=TEXT_SECONDARY).grid(row=0, column=0, padx=(0,6), sticky="w")
        self.patient_name = ctk.CTkEntry(fields, width=180,
            placeholder_text="Full name", fg_color=BG_INPUT,
            border_color=BORDER, font=ctk.CTkFont(size=12))
        self.patient_name.grid(row=0, column=1, padx=(0,16))

        ctk.CTkLabel(fields, text="Age:", font=ctk.CTkFont(size=12),
            text_color=TEXT_SECONDARY).grid(row=0, column=2, padx=(0,6), sticky="w")
        self.patient_age = ctk.CTkEntry(fields, width=70,
            placeholder_text="Age", fg_color=BG_INPUT,
            border_color=BORDER, font=ctk.CTkFont(size=12))
        self.patient_age.grid(row=0, column=3, padx=(0,16))

        ctk.CTkLabel(fields, text="Gender:", font=ctk.CTkFont(size=12),
            text_color=TEXT_SECONDARY).grid(row=0, column=4, padx=(0,6), sticky="w")
        self.patient_gender = ctk.CTkOptionMenu(fields,
            values=["Male", "Female", "Other"], width=110,
            fg_color=BG_INPUT, button_color=ACCENT_BLUE,
            font=ctk.CTkFont(size=12))
        self.patient_gender.grid(row=0, column=5, padx=(0,16))

        ctk.CTkLabel(fields, text="Date:", font=ctk.CTkFont(size=12),
            text_color=TEXT_SECONDARY).grid(row=0, column=6, padx=(0,6), sticky="w")
        self.patient_date = ctk.CTkEntry(fields, width=120,
            fg_color=BG_INPUT, border_color=BORDER,
            font=ctk.CTkFont(size=12))
        self.patient_date.grid(row=0, column=7)
        self.patient_date.insert(0, datetime.datetime.now().strftime("%d/%m/%Y"))

        # Main panels
        panels = ctk.CTkFrame(frame, fg_color="transparent")
        panels.pack(fill="both", expand=True, padx=15, pady=(0,15))

        # Left
        left = ctk.CTkFrame(panels, fg_color=BG_CARD, corner_radius=12)
        left.pack(side="left", fill="both", expand=True, padx=(0,8))

        ctk.CTkLabel(left, text="X-Ray Image",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=TEXT_PRIMARY).pack(pady=(14,6))

        self.image_label = ctk.CTkLabel(left,
            text="No image selected\n\nClick below to upload",
            width=400, height=290, fg_color=BG_HOVER, corner_radius=10,
            font=ctk.CTkFont(size=13), text_color=TEXT_SECONDARY)
        self.image_label.pack(padx=14, pady=6)

        # Scan buttons
        scan_label = ctk.CTkLabel(left, text="Select Scan Type:",
            font=ctk.CTkFont(size=12),
            text_color=TEXT_SECONDARY)
        scan_label.pack(pady=(6,4))

        scan_btns = ctk.CTkFrame(left, fg_color="transparent")
        scan_btns.pack(fill="x", padx=14, pady=(0,6))

        ctk.CTkButton(scan_btns, text="⚡ Quick Scan\n95% Accuracy • 2 Diseases",
            height=50, font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=ACCENT_BLUE, hover_color="#2563eb",
            corner_radius=10, command=self.quick_scan).pack(
            side="left", expand=True, padx=(0,6))

        ctk.CTkButton(scan_btns, text="🔬 Full Scan\n54% Accuracy • 14 Diseases",
            height=50, font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#166534", hover_color="#14532d",
            corner_radius=10, command=self.full_scan).pack(
            side="right", expand=True, padx=(6,0))

        ctk.CTkButton(left, text="Save PDF Report", height=38,
            font=ctk.CTkFont(size=12),
            fg_color=BG_HOVER, hover_color=BG_INPUT,
            corner_radius=10, command=self.save_pdf).pack(
            fill="x", padx=14, pady=(0,10))

        # Right
        right = ctk.CTkFrame(panels, fg_color=BG_CARD, corner_radius=12, width=430)
        right.pack(side="right", fill="both", padx=(8,0))
        right.pack_propagate(False)

        ctk.CTkLabel(right, text="Detected Conditions",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=TEXT_PRIMARY).pack(pady=(14,6))

        self.results_scroll = ctk.CTkScrollableFrame(right,
            fg_color=BG_HOVER, corner_radius=10, height=180)
        self.results_scroll.pack(fill="x", padx=14, pady=6)

        self.results_placeholder = ctk.CTkLabel(self.results_scroll,
            text="Upload an X-ray and select scan type",
            font=ctk.CTkFont(size=12), text_color=TEXT_SECONDARY)
        self.results_placeholder.pack(pady=20)

        ctk.CTkLabel(right, text="Recommendations",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=TEXT_PRIMARY).pack(anchor="w", padx=16, pady=(8,4))

        self.rec_scroll = ctk.CTkScrollableFrame(right,
            fg_color=BG_HOVER, corner_radius=10)
        self.rec_scroll.pack(fill="both", expand=True, padx=14, pady=(0,14))

        self.rec_placeholder = ctk.CTkLabel(self.rec_scroll,
            text="Recommendations will appear here",
            font=ctk.CTkFont(size=12), text_color=TEXT_SECONDARY)
        self.rec_placeholder.pack(pady=20)

    def build_history_page(self):
        frame = self.history_frame_outer
        topbar = ctk.CTkFrame(frame, fg_color=BG_CARD, height=55, corner_radius=0)
        topbar.pack(fill="x")
        topbar.pack_propagate(False)
        ctk.CTkLabel(topbar, text="Analysis History",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=TEXT_PRIMARY).pack(side="left", padx=20, pady=15)

        self.history_scroll = ctk.CTkScrollableFrame(frame, fg_color=BG_PRIMARY)
        self.history_scroll.pack(fill="both", expand=True, padx=15, pady=15)

        self.history_empty = ctk.CTkLabel(self.history_scroll,
            text="No analyses yet\n\nGo to Detector and upload an X-ray",
            text_color=TEXT_SECONDARY, font=ctk.CTkFont(size=14))
        self.history_empty.pack(pady=80)

    def upload_and_scan(self, mode):
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not path:
            return
        self.current_image_path = path
        self.current_mode = mode
        self.status.configure(text=f"Running {mode} analysis...")
        threading.Thread(target=self.analyze, args=(path, mode), daemon=True).start()

    def quick_scan(self):
        self.upload_and_scan("Quick")

    def full_scan(self):
        self.upload_and_scan("Full")

    def analyze(self, path, mode):
        img = Image.open(path).convert("RGB")
        display_img = img.resize((400, 290))
        ctk_img = ctk.CTkImage(light_image=display_img, size=(400, 290))
        self.image_label.configure(image=ctk_img, text="")
        self.image_label.image = ctk_img

        img_array = np.array(img.resize((224, 224))) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        results = []

        if mode == "Quick":
            prediction = model_quick.predict(img_array, verbose=0)[0][0]
            if prediction > 0.5:
                results = [("PNEUMONIA", float(prediction) * 100)]
            else:
                results = [("NORMAL", (1 - float(prediction)) * 100)]

        else:
            predictions = model_full.predict(img_array, verbose=0)[0]
            for i, label in enumerate(LABELS):
                confidence = float(predictions[i]) * 100
                if confidence > 15:
                    results.append((label, confidence))
            results.sort(key=lambda x: x[1], reverse=True)
            if not results:
                results = [("No Finding", 85.0)]

        self.current_results = results

        # Update results panel
        for widget in self.results_scroll.winfo_children():
            widget.destroy()

        mode_label = "⚡ Quick Scan" if mode == "Quick" else "🔬 Full Scan"
        ctk.CTkLabel(self.results_scroll,
            text=mode_label,
            font=ctk.CTkFont(size=11),
            text_color=ACCENT_BLUE if mode == "Quick" else ACCENT_GREEN).pack(
            anchor="w", padx=8, pady=(6,2))

        for disease, conf in results:
            is_disease = disease not in ["NORMAL", "No Finding"]
            color = ACCENT_RED if is_disease else ACCENT_GREEN

            row = ctk.CTkFrame(self.results_scroll, fg_color=BG_INPUT, corner_radius=8)
            row.pack(fill="x", pady=3, padx=4)

            ctk.CTkLabel(row, text=disease,
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color=color).pack(side="left", padx=10, pady=8)

            bar_frame = ctk.CTkFrame(row, fg_color="transparent")
            bar_frame.pack(side="right", padx=10, pady=8)
            ctk.CTkLabel(bar_frame, text=f"{conf:.1f}%",
                font=ctk.CTkFont(size=12),
                text_color=color).pack(side="right", padx=(6,0))
            bar = ctk.CTkProgressBar(bar_frame, width=100, height=6,
                progress_color=color, fg_color=BG_HOVER)
            bar.pack(side="right")
            bar.set(min(conf/100, 1.0))

        # Update recommendations
        for widget in self.rec_scroll.winfo_children():
            widget.destroy()

        top_disease = results[0][0]
        recs = RECOMMENDATIONS.get(top_disease, RECOMMENDATIONS['No Finding'])
        for rec in recs:
            ctk.CTkLabel(self.rec_scroll, text=rec,
                font=ctk.CTkFont(size=12), text_color=TEXT_PRIMARY,
                wraplength=370, justify="left", anchor="w").pack(
                fill="x", padx=12, pady=4, anchor="w")

        self.status.configure(
            text=f"{mode} scan complete — {len(results)} condition(s) detected")
        self.add_to_history(results, mode)

    def add_to_history(self, results, mode):
        name = self.patient_name.get() or "Unknown"
        age = self.patient_age.get() or "—"
        date = self.patient_date.get()
        history_log.append({
            "name": name, "age": age, "date": date,
            "results": results, "mode": mode
        })
        self.history_empty.pack_forget()

        card = ctk.CTkFrame(self.history_scroll, fg_color=BG_CARD, corner_radius=10)
        card.pack(fill="x", pady=5)

        top = ctk.CTkFrame(card, fg_color="transparent")
        top.pack(fill="x", padx=16, pady=(12,4))
        ctk.CTkLabel(top, text=f"👤  {name}  |  Age: {age}",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=TEXT_PRIMARY).pack(side="left")
        ctk.CTkLabel(top, text=f"{date} • {mode} Scan",
            font=ctk.CTkFont(size=11),
            text_color=TEXT_SECONDARY).pack(side="right")

        diseases = ", ".join([f"{d} ({c:.0f}%)" for d, c in results[:3]])
        color = ACCENT_RED if results[0][0] not in ["NORMAL", "No Finding"] else ACCENT_GREEN
        ctk.CTkLabel(card, text=diseases,
            font=ctk.CTkFont(size=12),
            text_color=color).pack(anchor="w", padx=16, pady=(0,12))

    def save_pdf(self):
        if not self.current_results:
            messagebox.showwarning("No Result", "Please analyze an X-ray first!")
            return

        name = self.patient_name.get() or "Unknown"
        age = self.patient_age.get() or "—"
        gender = self.patient_gender.get()
        date = self.patient_date.get()
        mode = self.current_mode or "Unknown"

        save_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")],
            initialfile=f"XRay_Report_{name}_{date.replace('/', '-')}.pdf")
        if not save_path:
            return

        doc = SimpleDocTemplate(save_path, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph("XRay AI — Medical Report", styles['Title']))
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph(
            f"Generated: {datetime.datetime.now().strftime('%d %B %Y, %H:%M')}  |  Scan Type: {mode} Scan",
            styles['Normal']))
        story.append(Spacer(1, 0.3*inch))

        story.append(Paragraph("Patient Information", styles['Heading2']))
        t = Table([
            ["Name", name], ["Age", age],
            ["Gender", gender], ["Date", date]
        ], colWidths=[2*inch, 4*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (0,-1), colors.lightblue),
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,0), (-1,-1), 11),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('PADDING', (0,0), (-1,-1), 8),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.3*inch))

        story.append(Paragraph("Detected Conditions", styles['Heading2']))
        result_data = [["Disease", "Confidence"]]
        for disease, conf in self.current_results:
            result_data.append([disease, f"{conf:.1f}%"])
        rt = Table(result_data, colWidths=[3*inch, 3*inch])
        rt.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.darkblue),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,0), (-1,-1), 11),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('PADDING', (0,0), (-1,-1), 8),
        ]))
        story.append(rt)
        story.append(Spacer(1, 0.3*inch))

        story.append(Paragraph("Recommendations", styles['Heading2']))
        top_disease = self.current_results[0][0]
        recs = RECOMMENDATIONS.get(top_disease, RECOMMENDATIONS['No Finding'])
        for rec in recs:
            clean = rec.replace("🔴  ", "• ").replace("🟢  ", "• ")
            story.append(Paragraph(clean, styles['Normal']))
            story.append(Spacer(1, 0.05*inch))
        story.append(Spacer(1, 0.2*inch))

        story.append(Paragraph("X-Ray Image", styles['Heading2']))
        if self.current_image_path:
            story.append(RLImage(self.current_image_path,
                width=3*inch, height=3*inch))
        story.append(Spacer(1, 0.2*inch))

        story.append(Paragraph(
            "<i>Disclaimer: This report is AI-generated and should not replace "
            "professional medical diagnosis. Consult a qualified doctor.</i>",
            styles['Normal']))

        doc.build(story)
        messagebox.showinfo("Saved!", f"PDF Report saved!\n{save_path}")

app = XRayApp()
app.mainloop()