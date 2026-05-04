from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "plant_disease_model.h5"
LABELS_PATH = ROOT / "class_labels.txt"
SAMPLES_DIR = ROOT / "samples"

model = tf.keras.models.load_model(MODEL_PATH)
class_names = LABELS_PATH.read_text(encoding="utf-8").splitlines()

_, H, W, _ = model.input_shape
sample_images = sorted(
    [
        str(path)
        for path in SAMPLES_DIR.glob("*")
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    ]
)


def prettify(label: str) -> str:
    plant, _, disease = label.partition("___")
    return f"{plant.replace('_', ' ').strip()} — {disease.replace('_', ' ').strip()}"


def sobel_edges(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    mag = np.clip(mag / max(mag.max(), 1e-6) * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(mag, cv2.COLOR_GRAY2RGB)


def sharpen(rgb: np.ndarray) -> np.ndarray:
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(rgb, -1, kernel)


def detect_blobs(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 0
    params.filterByArea = True
    params.minArea = 30
    params.maxArea = 5000
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(gray)
    return cv2.drawKeypoints(
        rgb,
        keypoints,
        None,
        (255, 0, 0),
        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )


def process(image: Image.Image):
    if image is None:
        return None, None, None, {}
    rgb = np.asarray(image.convert("RGB"))
    edges = sobel_edges(rgb)
    sharpened = sharpen(rgb)
    blobs = detect_blobs(rgb)

    resized = image.convert("RGB").resize((W, H))
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    probs = model.predict(arr[None, ...], verbose=0)[0]
    labels = {prettify(class_names[i]): float(probs[i]) for i in range(len(class_names))}
    return edges, sharpened, blobs, labels


with gr.Blocks(title="Plant Disease Detection") as demo:
    gr.Markdown(
        "# Plant Disease Detection\n"
        "Upload a leaf image. Classical preprocessing steps "
        "(Sobel edges, sharpening, blob detection) are shown alongside the model prediction."
    )
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="Leaf image")
            run_btn = gr.Button("Analyze", variant="primary")
        out_label = gr.Label(num_top_classes=3, label="Prediction")
    with gr.Row():
        out_edges = gr.Image(label="Sobel edges", height=240)
        out_sharp = gr.Image(label="Sharpened", height=240)
        out_blobs = gr.Image(label="Blob detection", height=240)
    if sample_images:
        gr.Examples(examples=sample_images, inputs=inp)

    outputs = [out_edges, out_sharp, out_blobs, out_label]
    run_btn.click(process, inputs=inp, outputs=outputs)
    inp.change(process, inputs=inp, outputs=outputs)


if __name__ == "__main__":
    demo.launch()
