import os
import cv2
import numpy as np
import tensorflow as tf
import base64
import time
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model

app = Flask(__name__, static_folder='static', template_folder='templates')

MODEL_PATH = "models/autoencoder_noise.h5"
autoencoder = load_model(MODEL_PATH)

def calculate_snr(image):
    signal = np.mean(image)
    noise = np.std(image)
    if noise == 0:
        return float('inf')
    return 10 * np.log10(signal / noise)

def calculate_denoised_snr(original, denoised):
    signal_power = np.mean(original ** 2)
    noise_power = np.mean((original - denoised) ** 2)
    if noise_power == 0:
        return float('inf')
    return 10 * np.log10(signal_power / noise_power)

def load_image(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("error")
    img = img.astype(np.float32)
    img = cv2.resize(img, (200, 200))
    img = np.repeat(img[..., np.newaxis], 3, -1)
    img = img / 255.0
    return img

def denoise_image(model, image):
    image_expanded = np.expand_dims(image, axis=0)
    denoised_image = model.predict(image_expanded)
    return np.squeeze(denoised_image)

def image_to_base64(image):
    _, buffer = cv2.imencode('.png', (image * 255).astype(np.uint8))
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/api/images", methods=["GET"])
def get_images():
    noise_dir = os.path.join("Image", "Noise")
    if not os.path.exists(noise_dir):
        return jsonify({})
    categories = os.listdir(noise_dir)
    data = {}
    for cat in categories:
        if os.path.isdir(os.path.join(noise_dir, cat)):
            files = [f for f in os.listdir(os.path.join(noise_dir, cat)) if not f.startswith('.')]
            data[cat] = [f for f in files]
    return jsonify(data)

@app.route("/api/process", methods=["POST"])
def process():
    time.sleep(5)
    req = request.json
    category = req.get("category")
    noise_file = req.get("file")
    
    noise_path = os.path.join("Image", "Noise", category, noise_file)
    
    if not os.path.exists(noise_path):
        return jsonify({"error": "File not found"}), 404
        
    noise_img = load_image(noise_path)
    noise_denoised = denoise_image(autoencoder, noise_img)
    
    noise_snr_orig = calculate_snr(noise_img[..., 0])
    noise_snr_denoised = calculate_denoised_snr(noise_img[..., 0], noise_denoised[..., 0])
    
    return jsonify({
        "noise_b64": image_to_base64(noise_img),
        "noise_denoised_b64": image_to_base64(noise_denoised),
        "noise_orig_snr": float(round(noise_snr_orig, 2)),
        "noise_denoised_snr": float(round(noise_snr_denoised, 2)),
        "noise_improvement": float(round(noise_snr_denoised - noise_snr_orig, 2))
    })

if __name__ == "__main__":
    app.run(debug=True)
