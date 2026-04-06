import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
import threading

# App settings
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

model = tf.keras.models.load_model('model/xray_model.h5')

RECOMMENDATIONS = {
    "PNEUMONIA": [
        "Consult a doctor immediately",
        "Get a blood test done",
        "Avoid cold environments",
        "Stay hydrated and rest",
        "Take prescribed antibiotics"
    ],
    "NORMAL": [
        "Your X-ray appears normal",
        "Maintain a healthy lifestyle",
        "Annual checkup recommended",
        "Exercise regularly",
        "Avoid smoking"
    ]
}

class XRayApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("XRay AI — Anomaly Detector")
        self.geometry("900x650")
        self.resizable(False, False)
        self.build_ui()

    def build_ui(self):
        # Header
        header = ctk.CTkFrame(self, height=60, corner_radius=0)
        header.pack(fill="x")
        ctk.CTkLabel(header, text="🏥  XRay AI — Anomaly Detector",
            font=ctk.CTkFont(size=20, weight="bold")).pack(side="left", padx=20, pady=15)
        ctk.CTkLabel(header, text="Powered by Deep Learning",
            font=ctk.CTkFont(size=12), text_color="gray").pack(side="right", padx=20)

        # Main area
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=20, pady=20)

        # Left panel
        left = ctk.CTkFrame(main, width=420)
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))
        left.pack_propagate(False)

        ctk.CTkLabel(left, text="Upload X-Ray Image",
            font=ctk.CTkFont(size=15, weight="bold")).pack(pady=(15, 5))

        self.image_label = ctk.CTkLabel(left, text="No image selected\n\nClick below to upload",
            width=380, height=350, fg_color="#1a1a2e", corner_radius=10,
            font=ctk.CTkFont(size=13), text_color="gray")
        self.image_label.pack(padx=15, pady=10)

        ctk.CTkButton(left, text="Browse X-Ray Image", height=40,
            font=ctk.CTkFont(size=14), command=self.upload_image).pack(pady=10, padx=15, fill="x")

        # Right panel
        right = ctk.CTkFrame(main, width=400)
        right.pack(side="right", fill="both", expand=True, padx=(10, 0))
        right.pack_propagate(False)

        ctk.CTkLabel(right, text="Analysis Result",
            font=ctk.CTkFont(size=15, weight="bold")).pack(pady=(15, 5))

        self.result_label = ctk.CTkLabel(right, text="—",
            font=ctk.CTkFont(size=28, weight="bold"), text_color="gray")
        self.result_label.pack(pady=10)

        self.confidence_label = ctk.CTkLabel(right, text="Confidence: —",
            font=ctk.CTkFont(size=13), text_color="gray")
        self.confidence_label.pack()

        self.progress = ctk.CTkProgressBar(right, width=350)
        self.progress.pack(pady=10, padx=15)
        self.progress.set(0)

        ctk.CTkLabel(right, text="Recommendations",
            font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(15, 5))

        self.rec_frame = ctk.CTkFrame(right, fg_color="#1a1a2e", corner_radius=10)
        self.rec_frame.pack(fill="both", expand=True, padx=15, pady=(0, 15))

        self.rec_label = ctk.CTkLabel(self.rec_frame,
            text="Upload an X-ray to see recommendations",
            font=ctk.CTkFont(size=12), text_color="gray", wraplength=340)
        self.rec_label.pack(pady=20, padx=15)

        # Status bar
        self.status = ctk.CTkLabel(self, text="Ready",
            font=ctk.CTkFont(size=11), text_color="gray")
        self.status.pack(pady=5)

    def upload_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not path:
            return
        self.status.configure(text="Analyzing...")
        self.result_label.configure(text="Analyzing...", text_color="gray")
        threading.Thread(target=self.analyze, args=(path,), daemon=True).start()

    def analyze(self, path):
        img = Image.open(path).convert("RGB")
        display_img = img.resize((380, 350))
        ctk_img = ctk.CTkImage(light_image=display_img, size=(380, 350))
        self.image_label.configure(image=ctk_img, text="")
        self.image_label.image = ctk_img

        img_array = np.array(img.resize((224, 224))) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array, verbose=0)[0][0]

        if prediction > 0.5:
            label = "PNEUMONIA"
            confidence = prediction
            color = "#E24B4A"
        else:
            label = "NORMAL"
            confidence = 1 - prediction
            color = "#1D9E75"

        conf_pct = confidence * 100
        self.result_label.configure(text=label, text_color=color)
        self.confidence_label.configure(text=f"Confidence: {conf_pct:.1f}%")
        self.progress.set(confidence)

        recs = RECOMMENDATIONS[label]
        rec_text = "\n\n".join([f"• {r}" for r in recs])
        self.rec_label.configure(text=rec_text, text_color="white")
        self.status.configure(text=f"Analysis complete — {label} detected")

app = XRayApp()
app.mainloop()