import os
import cv2
import numpy as np
import tensorflow as tf
import gradio as gr
from tensorflow.keras.models import load_model

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

def process_image(input_img):
    if input_img is None:
        return None, "0.00 dB", "0.00 dB", "0.00 dB"
        
    img = cv2.cvtColor(input_img, cv2.COLOR_RGB2GRAY)
    img = img.astype(np.float32)
    img = cv2.resize(img, (200, 200))
    img = np.repeat(img[..., np.newaxis], 3, -1)
    img = img / 255.0
    
    image_expanded = np.expand_dims(img, axis=0)
    denoised_image = autoencoder.predict(image_expanded)
    denoised_image = np.squeeze(denoised_image)
    
    noise_snr_orig = calculate_snr(img[..., 0])
    noise_snr_denoised = calculate_denoised_snr(img[..., 0], denoised_image[..., 0])
    improvement = noise_snr_denoised - noise_snr_orig
    
    out_img = (denoised_image * 255).astype(np.uint8)
    
    return out_img, f"{noise_snr_orig:.2f} dB", f"{noise_snr_denoised:.2f} dB", f"{improvement:+.2f} dB"

with gr.Blocks(theme=gr.themes.Base(primary_hue="blue", neutral_hue="slate")) as demo:
    gr.Markdown(
        """
        # 🧠 NeuroScanAI
        ### Advanced Deep Learning Denoising Engine
        Upload a Brain CT or MRI scan to automatically reconstruct and remove artifact noise using our custom-trained Autoencoder model. Developed by Shubham Vishwakarma.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="numpy", label="Upload Noisy Scan")
            process_btn = gr.Button("Analyze Scan", variant="primary")
            
        with gr.Column(scale=1):
            image_output = gr.Image(type="numpy", label="Output: Reconstructed")
            
            with gr.Row():
                orig_snr_text = gr.Textbox(label="Original SNR", interactive=False)
                denoised_snr_text = gr.Textbox(label="Denoised SNR", interactive=False)
                improvement_text = gr.Textbox(label="Improvement", interactive=False)
                
    process_btn.click(
        fn=process_image,
        inputs=image_input,
        outputs=[image_output, orig_snr_text, denoised_snr_text, improvement_text]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
