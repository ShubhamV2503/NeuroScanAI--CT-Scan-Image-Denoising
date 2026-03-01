import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

MODEL_PATH = "models/autoencoder_noise.h5"
IMAGE_DIR = "Image"
OUTPUT_DIR = "Output"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

if os.path.exists(MODEL_PATH):
    print(f"Loading model from {MODEL_PATH}...")
    autoencoder = load_model(MODEL_PATH)
else:
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found. Please check the path.")

def calculate_original_snr(image):
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
    # Read the image in grayscale
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {filepath}")
    
    img = img.astype(np.float32)
    img = cv2.resize(img, (200, 200))
    img = np.repeat(img[..., np.newaxis], 3, -1)
    img = img / 255.0
    return img

def denoise_image(model, image):
    image_expanded = np.expand_dims(image, axis=0)
    denoised_image = model.predict(image_expanded)
    return np.squeeze(denoised_image)

def process_images():
    print(f"\nProcessing images from directory: {IMAGE_DIR}")
    print("-" * 50)
    
    if not os.path.exists(IMAGE_DIR):
        print(f"Error: Directory '{IMAGE_DIR}' not found.")
        return
        
    for filename in os.listdir(IMAGE_DIR):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.dcm')):
            continue
            
        filepath = os.path.join(IMAGE_DIR, filename)
        print(f"Processing: {filename}")
        
        try:
            # Load and preprocess
            original_image = load_image(filepath)
            
            # Denoise
            denoised_image = denoise_image(autoencoder, original_image)
            
            # Calculate SNRs
            original_snr = calculate_original_snr(original_image[..., 0])
            denoised_snr = calculate_denoised_snr(original_image[..., 0], denoised_image[..., 0])
            
            print(f"  -> Original SNR: {original_snr:.2f} dB")
            print(f"  -> Denoised SNR: {denoised_snr:.2f} dB")
            
            # Save results (side by side)
            orig_display = (original_image * 255).astype(np.uint8)
            denoised_display = (denoised_image * 255).astype(np.uint8)
            
            combined = np.hstack((orig_display, denoised_display))
            
            out_path = os.path.join(OUTPUT_DIR, f"result_{filename}")
            cv2.imwrite(out_path, combined)
            print(f"  -> Saved comparison to {out_path}\n")
            
        except Exception as e:
            print(f"  -> Error processing {filename}: {str(e)}\n")

if __name__ == "__main__":
    process_images()
    print(f"All done! Check the '{OUTPUT_DIR}' directory for visual results.")
