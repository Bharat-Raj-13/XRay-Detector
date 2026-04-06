import customtkinter as ctk
from tkinter import filedialog, messagebox
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import threading
import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, KeepTogether
from reportlab.lib.units import inch

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

print("Loading model...")
model = tf.keras.models.load_model('model/xray_model.h5')
print("Model loaded!")

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

RECOMMENDATIONS = {
    "PNEUMONIA": [
        "🔴  Consult a doctor or pulmonologist immediately",
        "🔴  Get a complete blood count (CBC) test done",
        "🔴  Take prescribed antibiotics without skipping",
        "🔴  Rest completely — avoid all physical exertion",
        "🔴  Stay hydrated — drink 8-10 glasses daily",
        "🔴  Monitor oxygen levels with a pulse oximeter",
        "🔴  Avoid cold environments and air conditioning",
        "🔴  Follow up with a chest X-ray after treatment",
    ],
    "NORMAL": [
        "🟢  Your X-ray appears completely normal",
        "🟢  Maintain a healthy and balanced diet",
        "🟢  Exercise regularly — at least 30 minutes daily",
        "🟢  Get an annual chest checkup as a precaution",
        "🟢  Avoid smoking and alcohol consumption",
        "🟢  Stay hydrated — drink 8 glasses of water daily",
        "🟢  Practice deep breathing exercises daily",
        "🟢  Keep your vaccinations up to date",
    ]
}

history_log = []

def generate_gradcam(img_array, model):
    try:
        base_model = model.layers[0]
        last_conv_layer_name = "block5_conv3"
        last_conv_layer = base_model.get_layer(last_conv_layer_name)

        grad_model = tf.keras.models.Model(
            inputs=base_model.input,
            outputs=[last_conv_layer.output, base_model.output]
        )

        img_tensor = tf.cast(img_array, tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            conv_outputs, base_output = grad_model(img_tensor)
            x = base_output
            for layer in model.layers[1:]:
                x = layer(x)
            loss = x[:, 0]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0].numpy()
        pooled_grads = pooled_grads.numpy()

        for i in range(pooled_grads.shape[-1]):
            conv_outputs[:, :, i] *= pooled_grads[i]

        heatmap = np.mean(conv_outputs, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap = heatmap / (np.max(heatmap) + 1e-8)
        return heatmap

    except Exception as e:
        print(f"Gradcam error: {e}")
        return None

def apply_heatmap(original_img, heatmap):
    try:
        img_array = np.array(original_img.resize((224, 224)))
        heatmap_resized = cv2.resize(heatmap, (224, 224))
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        superimposed = cv2.addWeighted(img_array, 0.6, heatmap_colored, 0.4, 0)
        return Image.fromarray(superimposed)
    except Exception as e:
        print(f"Heatmap apply error: {e}")
        return original_img

class XRayApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("XRay AI — Pneumonia Detection System")
        self.geometry("1300x820")
        self.resizable(True, True)
        self.state('zoomed')
        self.configure(fg_color=BG_PRIMARY)
        self.current_image_path = None
        self.current_result = None
        self.current_confidence = None
        self.heatmap_image = None
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
        ctk.CTkLabel(logo_frame, text="Pneumonia Detector",
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

        ctk.CTkLabel(self.sidebar, text="MODEL INFO",
            font=ctk.CTkFont(size=10),
            text_color=TEXT_SECONDARY).pack(anchor="w", padx=20, pady=(0,8))

        info_items = [
            ("Architecture", "VGG16"),
            ("Dataset", "5,216 X-Rays"),
            ("Classes", "Normal / Pneumonia"),
            ("Accuracy", "~95%"),
            ("Explainability", "Grad-CAM"),
        ]
        for label, value in info_items:
            row = ctk.CTkFrame(self.sidebar, fg_color="transparent")
            row.pack(fill="x", padx=16, pady=2)
            ctk.CTkLabel(row, text=label,
                font=ctk.CTkFont(size=11),
                text_color=TEXT_SECONDARY).pack(side="left")
            ctk.CTkLabel(row, text=value,
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color=TEXT_PRIMARY).pack(side="right")

        ctk.CTkFrame(self.sidebar, height=1,
            fg_color=BORDER).pack(fill="x", padx=16, pady=12)

        ctk.CTkLabel(self.sidebar, text="HEATMAP LEGEND",
            font=ctk.CTkFont(size=10),
            text_color=TEXT_SECONDARY).pack(anchor="w", padx=20, pady=(0,8))

        legend_items = [
            ("High anomaly", ACCENT_RED),
            ("Moderate", "#f59e0b"),
            ("Normal region", ACCENT_BLUE),
        ]
        for meaning, color in legend_items:
            row = ctk.CTkFrame(self.sidebar, fg_color="transparent")
            row.pack(fill="x", padx=16, pady=3)
            dot = ctk.CTkFrame(row, fg_color=color,
                width=12, height=12, corner_radius=6)
            dot.pack(side="left", padx=(0,8))
            ctk.CTkLabel(row, text=meaning,
                font=ctk.CTkFont(size=11),
                text_color=TEXT_PRIMARY).pack(side="left")

        status_frame = ctk.CTkFrame(self.sidebar, fg_color=BG_HOVER,
            corner_radius=8)
        status_frame.pack(fill="x", padx=16, pady=16, side="bottom")
        ctk.CTkLabel(status_frame, text="🟢  Model Ready",
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
        ctk.CTkLabel(topbar, text="Chest X-Ray Pneumonia Detector",
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

        ctk.CTkLabel(fields, text="Name:",
            font=ctk.CTkFont(size=12),
            text_color=TEXT_SECONDARY).grid(row=0, column=0, padx=(0,6), sticky="w")
        self.patient_name = ctk.CTkEntry(fields, width=190,
            placeholder_text="Full name", fg_color=BG_INPUT,
            border_color=BORDER, font=ctk.CTkFont(size=12))
        self.patient_name.grid(row=0, column=1, padx=(0,16))

        ctk.CTkLabel(fields, text="Age:",
            font=ctk.CTkFont(size=12),
            text_color=TEXT_SECONDARY).grid(row=0, column=2, padx=(0,6), sticky="w")
        self.patient_age = ctk.CTkEntry(fields, width=70,
            placeholder_text="Age", fg_color=BG_INPUT,
            border_color=BORDER, font=ctk.CTkFont(size=12))
        self.patient_age.grid(row=0, column=3, padx=(0,16))

        ctk.CTkLabel(fields, text="Gender:",
            font=ctk.CTkFont(size=12),
            text_color=TEXT_SECONDARY).grid(row=0, column=4, padx=(0,6), sticky="w")
        self.patient_gender = ctk.CTkOptionMenu(fields,
            values=["Male", "Female", "Other"], width=110,
            fg_color=BG_INPUT, button_color=ACCENT_BLUE,
            font=ctk.CTkFont(size=12))
        self.patient_gender.grid(row=0, column=5, padx=(0,16))

        ctk.CTkLabel(fields, text="Date:",
            font=ctk.CTkFont(size=12),
            text_color=TEXT_SECONDARY).grid(row=0, column=6, padx=(0,6), sticky="w")
        self.patient_date = ctk.CTkEntry(fields, width=120,
            fg_color=BG_INPUT, border_color=BORDER,
            font=ctk.CTkFont(size=12))
        self.patient_date.grid(row=0, column=7)
        self.patient_date.insert(0, datetime.datetime.now().strftime("%d/%m/%Y"))

        # Main panels
        panels = ctk.CTkFrame(frame, fg_color="transparent")
        panels.pack(fill="both", expand=True, padx=15, pady=(0,15))

        # Left panel
        left = ctk.CTkFrame(panels, fg_color=BG_CARD, corner_radius=12)
        left.pack(side="left", fill="both", expand=True, padx=(0,6))

        ctk.CTkLabel(left, text="Original X-Ray",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=TEXT_PRIMARY).pack(pady=(14,6))

        self.image_label = ctk.CTkLabel(left,
            text="No image selected\n\nClick Browse to upload",
            width=310, height=310, fg_color=BG_HOVER, corner_radius=10,
            font=ctk.CTkFont(size=13), text_color=TEXT_SECONDARY)
        self.image_label.pack(padx=14, pady=6)

        btns = ctk.CTkFrame(left, fg_color="transparent")
        btns.pack(fill="x", padx=14, pady=10)
        ctk.CTkButton(btns, text="Browse X-Ray", height=42,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=ACCENT_BLUE, hover_color="#2563eb",
            corner_radius=10, command=self.upload_image).pack(
            side="left", expand=True, padx=(0,6))
        ctk.CTkButton(btns, text="Save PDF", height=42,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#166534", hover_color="#14532d",
            corner_radius=10, command=self.save_pdf).pack(
            side="right", expand=True, padx=(6,0))

        # Middle panel
        middle = ctk.CTkFrame(panels, fg_color=BG_CARD, corner_radius=12)
        middle.pack(side="left", fill="both", expand=True, padx=(6,6))

        ctk.CTkLabel(middle, text="Grad-CAM Heatmap",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=TEXT_PRIMARY).pack(pady=(14,6))

        self.heatmap_label = ctk.CTkLabel(middle,
            text="Heatmap will appear\nafter analysis",
            width=310, height=310, fg_color=BG_HOVER, corner_radius=10,
            font=ctk.CTkFont(size=13), text_color=TEXT_SECONDARY)
        self.heatmap_label.pack(padx=14, pady=6)

        ctk.CTkLabel(middle,
            text="Red = High anomaly region\nBlue = Normal region",
            font=ctk.CTkFont(size=11),
            text_color=TEXT_SECONDARY).pack(pady=8)

        # Right panel
        right = ctk.CTkFrame(panels, fg_color=BG_CARD, corner_radius=12, width=300)
        right.pack(side="right", fill="both", padx=(6,0))
        right.pack_propagate(False)

        ctk.CTkLabel(right, text="Analysis Result",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=TEXT_PRIMARY).pack(pady=(14,6))

        result_card = ctk.CTkFrame(right, fg_color=BG_HOVER,
            corner_radius=12, height=110)
        result_card.pack(fill="x", padx=14, pady=6)
        result_card.pack_propagate(False)

        self.result_label = ctk.CTkLabel(result_card, text="—",
            font=ctk.CTkFont(size=34, weight="bold"),
            text_color=TEXT_SECONDARY)
        self.result_label.place(relx=0.5, rely=0.38, anchor="center")

        self.confidence_label = ctk.CTkLabel(result_card,
            text="Upload an X-ray to begin",
            font=ctk.CTkFont(size=11), text_color=TEXT_SECONDARY)
        self.confidence_label.place(relx=0.5, rely=0.75, anchor="center")

        ctk.CTkLabel(right, text="Confidence Score",
            font=ctk.CTkFont(size=11),
            text_color=TEXT_SECONDARY).pack(anchor="w", padx=16, pady=(6,2))

        self.progress = ctk.CTkProgressBar(right, height=8,
            corner_radius=4, progress_color=ACCENT_BLUE, fg_color=BG_HOVER)
        self.progress.pack(fill="x", padx=14, pady=(0,10))
        self.progress.set(0)

        ctk.CTkLabel(right, text="Recommendations",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=TEXT_PRIMARY).pack(anchor="w", padx=16, pady=(4,6))

        self.rec_scroll = ctk.CTkScrollableFrame(right,
            fg_color=BG_HOVER, corner_radius=10)
        self.rec_scroll.pack(fill="both", expand=True, padx=14, pady=(0,14))

        ctk.CTkLabel(self.rec_scroll,
            text="Recommendations will\nappear here",
            font=ctk.CTkFont(size=12), text_color=TEXT_SECONDARY).pack(pady=20)

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

    def upload_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not path:
            return
        self.current_image_path = path
        self.status.configure(text="Analyzing and generating heatmap...")
        self.result_label.configure(text="...", text_color=TEXT_SECONDARY)
        threading.Thread(target=self.analyze, args=(path,), daemon=True).start()

    def analyze(self, path):
        img = Image.open(path).convert("RGB")

        display_img = img.resize((310, 310))
        ctk_img = ctk.CTkImage(light_image=display_img, size=(310, 310))
        self.image_label.configure(image=ctk_img, text="")
        self.image_label.image = ctk_img

        img_array = np.array(img.resize((224, 224))) / 255.0
        img_array_batch = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array_batch, verbose=0)[0][0]

        if prediction > 0.5:
            label = "PNEUMONIA"
            confidence = float(prediction)
            color = ACCENT_RED
        else:
            label = "NORMAL"
            confidence = 1 - float(prediction)
            color = ACCENT_GREEN

        self.current_result = label
        self.current_confidence = confidence * 100

        heatmap = generate_gradcam(img_array_batch, model)
        if heatmap is not None:
            heatmap_img = apply_heatmap(img, heatmap)
            self.heatmap_image = heatmap_img
            heatmap_display = heatmap_img.resize((310, 310))
            ctk_heatmap = ctk.CTkImage(light_image=heatmap_display, size=(310, 310))
            self.heatmap_label.configure(image=ctk_heatmap, text="")
            self.heatmap_label.image = ctk_heatmap
        else:
            self.heatmap_label.configure(text="Heatmap unavailable")

        self.result_label.configure(text=label, text_color=color)
        self.confidence_label.configure(
            text=f"Confidence: {self.current_confidence:.1f}%",
            text_color=color)
        self.progress.configure(progress_color=color)
        self.progress.set(confidence)

        for widget in self.rec_scroll.winfo_children():
            widget.destroy()
        for rec in RECOMMENDATIONS[label]:
            ctk.CTkLabel(self.rec_scroll, text=rec,
                font=ctk.CTkFont(size=11), text_color=TEXT_PRIMARY,
                wraplength=250, justify="left", anchor="w").pack(
                fill="x", padx=10, pady=3, anchor="w")

        self.status.configure(
            text=f"Analysis complete — {label} detected with {self.current_confidence:.1f}% confidence")
        self.add_to_history(label, self.current_confidence)

    def add_to_history(self, result, confidence):
        name = self.patient_name.get() or "Unknown"
        age = self.patient_age.get() or "—"
        date = self.patient_date.get()
        history_log.append({
            "name": name, "age": age, "date": date,
            "result": result, "confidence": confidence
        })
        self.history_empty.pack_forget()
        color = ACCENT_RED if result == "PNEUMONIA" else ACCENT_GREEN

        card = ctk.CTkFrame(self.history_scroll, fg_color=BG_CARD, corner_radius=10)
        card.pack(fill="x", pady=5)

        top = ctk.CTkFrame(card, fg_color="transparent")
        top.pack(fill="x", padx=16, pady=(12,4))
        ctk.CTkLabel(top, text=f"👤  {name}  |  Age: {age}",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=TEXT_PRIMARY).pack(side="left")
        ctk.CTkLabel(top, text=date,
            font=ctk.CTkFont(size=11),
            text_color=TEXT_SECONDARY).pack(side="right")
        ctk.CTkLabel(card,
            text=f"Result: {result}  —  Confidence: {confidence:.1f}%",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=color).pack(anchor="w", padx=16, pady=(0,12))

    def save_pdf(self):
        if not self.current_result:
            messagebox.showwarning("No Result", "Please analyze an X-ray first!")
            return

        name = self.patient_name.get() or "Unknown"
        age = self.patient_age.get() or "—"
        gender = self.patient_gender.get()
        date = self.patient_date.get()

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
            f"Generated: {datetime.datetime.now().strftime('%d %B %Y, %H:%M')}",
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

        story.append(Paragraph("Analysis Result", styles['Heading2']))
        result_color = "red" if self.current_result == "PNEUMONIA" else "green"
        story.append(Paragraph(
            f'<font color="{result_color}" size="18"><b>{self.current_result}</b></font>'
            f'  —  Confidence: {self.current_confidence:.1f}%',
            styles['Normal']))
        story.append(Spacer(1, 0.3*inch))

        story.append(Paragraph("Recommendations", styles['Heading2']))
        for rec in RECOMMENDATIONS[self.current_result]:
            clean = rec.replace("🔴  ", "• ").replace("🟢  ", "• ")
            story.append(Paragraph(clean, styles['Normal']))
            story.append(Spacer(1, 0.05*inch))
        story.append(Spacer(1, 0.2*inch))


        img_data = []
        if self.current_image_path:
            img_data.append(RLImage(self.current_image_path,
                width=2.5*inch, height=2.5*inch))
        else:
            img_data.append(Paragraph("Original not available", styles['Normal']))

        if self.heatmap_image:
            heatmap_path = "model/temp_heatmap.png"
            self.heatmap_image.save(heatmap_path)
            img_data.append(RLImage(heatmap_path,
                width=2.5*inch, height=2.5*inch))
        else:
            img_data.append(Paragraph("Heatmap not available", styles['Normal']))

        img_table = Table(
            [img_data, ["Original X-Ray", "Grad-CAM Heatmap"]],
            colWidths=[3*inch, 3*inch],
            rowHeights=[2.6*inch, 0.3*inch])
        img_table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,0), 'MIDDLE'),
            ('VALIGN', (0,1), (-1,1), 'TOP'),
            ('FONTNAME', (0,1), (-1,1), 'Helvetica'),
            ('FONTSIZE', (0,1), (-1,1), 10),
            ('TEXTCOLOR', (0,1), (-1,1), colors.grey),
            ('TOPPADDING', (0,1), (-1,1), 4),
            ('KEEPWITHNEXT', (0,0), (-1,0), 1),
        ]))
        story.append(KeepTogether([
    Paragraph("X-Ray Analysis Images", styles['Heading2']),
    Spacer(1, 0.1*inch),
    img_table
]))
        story.append(Spacer(1, 0.2*inch))

        story.append(Paragraph(
            "<i>Disclaimer: This report is AI-generated and should not replace "
            "professional medical diagnosis. Consult a qualified doctor.</i>",
            styles['Normal']))

        doc.build(story)
        messagebox.showinfo("Saved!", f"PDF Report saved!\n{save_path}")

app = XRayApp()
app.mainloop()