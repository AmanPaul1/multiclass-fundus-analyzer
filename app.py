import gradio as gr
import tensorflow as tf
import numpy as np
import os
import random
import zipfile
import requests
from PIL import Image
from huggingface_hub import hf_hub_download, login

# 1. READ-ONLY AUTHENTICATION LOCK (MALARIA COMPLIANT PROTOCOL)
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN:
    try:
        login(token=HF_TOKEN)
        print("🔓 Read-Only Authentication Session Active!")
    except Exception as e:
        print(f"⚠️ Session Notice: {e}")

MODEL_REPO = "paulaman1/multiclass_fundus_project" 
MODEL_FILENAME = "inception_edgenet_distilled_best.keras"
DATASET_REPO = "paulaman1/retinal-fundus-test-samples" 
ZIP_FILENAME = "Testdata.zip"  # Teri uploaded private zip file

print("⏳ Fetching private distilled intelligence graph...")
try:
    secure_model_path = hf_hub_download(
        repo_id=MODEL_REPO, filename=MODEL_FILENAME, token=HF_TOKEN, repo_type="model"
    )
    model = tf.keras.models.load_model(secure_model_path, compile=False)
    print("🎯 Private Pro Model loaded successfully!")
except Exception as e:
    print(f"⚠️ Model loading barrier: {e}")

# Compilation lookup mapping
short_classes = ['dr', 'glaucoma', 'maculopathy', 'normal', 'optic_atrophy']

clinical_labels = {
    'dr': 'Diabetic Retinopathy (DR)',
    'glaucoma': 'Glaucoma Stage',
    'maculopathy': 'Maculopathy Disease',
    'normal': 'Normal Healthy Retina',
    'optic_atrophy': 'Optic Atrophy (OA)'
}

# 2. NESTED ZIP RESOLVER WITH RAW SECURE HTTP STREAMING
def load_random_from_private_zip():
    local_download_target = "transient_testdata.zip"
    
    try:
        # Authenticated secure HTTP stream bypassing Hub context crashes
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        url = f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/main/{ZIP_FILENAME}"
        
        response = requests.get(url, headers=headers, stream=True)
        if response.status_code != 200:
            return None, f"Dataset Connection Interrupted: HTTP {response.status_code} Verification Barrier."
            
        with open(local_download_target, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    
        with zipfile.ZipFile(local_download_target, 'r') as archive:
            # Absolute image indexing: filter metadata artifacts and system logs
            all_files = [
                f for f in archive.namelist() 
                if f.lower().endswith(('.png', '.jpg', '.jpeg')) and '__macosx' not in f.lower() and not f.split('/')[-1].startswith('.')
            ]
            
            if not all_files:
                return None, "Error: No valid clinical images located in the specified structural layout inside zip."
                
            selected_file = random.choice(all_files)
            
            # Streaming target asset from inner nested buffer straight into processing pipeline
            with archive.open(selected_file) as file_stream:
                img = Image.open(file_stream)
                img.load()  
                img = img.convert('RGB')
                
            # DYNAMIC DEEP NESTED FOLDER PARSING ENGINE
            normalized_path = selected_file.lower()
            detected_truth = "Simulated Patient Workspace Active"
            
            if 'optic atrophy' in normalized_path or 'atrophy' in normalized_path or 'oa' in normalized_path:
                detected_truth = "Target Ground Truth Class: OPTIC ATROPHY (OA)"
            elif 'glaucoma' in normalized_path:
                detected_truth = "Target Ground Truth Class: GLAUCOMA STAGE"
            elif 'maculopathy' in normalized_path:
                detected_truth = "Target Ground Truth Class: MACULOPATHY DISEASE"
            elif 'normal' in normalized_path:
                detected_truth = "Target Ground Truth Class: NORMAL HEALTHY RETINA"
            elif 'diabetic' in normalized_path or 'retinopathy' in normalized_path or 'dr' in normalized_path:
                detected_truth = "Target Ground Truth Class: DIABETIC RETINOPATHY (DR)"
            else:
                # Failsafe tracker to show folder topology if match drifts
                path_segments = selected_file.split('/')
                # If nested inside Testdata/Folder_Name/image.jpg -> grab the middle folder name
                clean_name = path_segments[-2] if len(path_segments) >= 2 else path_segments[0]
                detected_truth = f"Target Ground Truth Source: {clean_name.upper()}"
                
            # Internal housekeeping storage swipe
            if os.path.exists(local_download_target):
                os.remove(local_download_target)
                
            return img, detected_truth
            
    except Exception as e:
        if os.path.exists(local_download_target):
            os.remove(local_download_target)
        return None, f"Runtime Data Retrieval Barrier: {str(e)}"

def predict_fundus(image):
    if image is None:
        return None
    
    img = image.resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array, verbose=0)[0]
    confidences = {clinical_labels[short_classes[i]]: float(predictions[i]) for i in range(len(short_classes))}
    return confidences

# 3. CLINICAL CONTROL PANEL BUILD
with gr.Blocks() as demo:
    gr.Markdown("# 🔬 Fundus Image Classifer")
    
    with gr.Group():
        gr.Markdown(
            """
            ### ⚠️ **CRITICAL MEDICAL DISCLAIMER & DEPLOYMENT PROTOCOL**
            * **Scope of Analysis:** This edge-optimized student network is explicitly engineered and trained *exclusively* for the evaluation of high-resolution **Retinal Fundus Optical Images**.
            * **Strict Constraint Matrix:** Processing or uploading external, non-fundus, or arbitrary real-world imagery is **strictly prohibited**.
            * **Target Clinical Evaluation Classes:** Diabetic Retinopathy (DR), Glaucoma Stage, Maculopathy Disease, Normal Healthy Retina, Optic Atrophy (OA).
            * **🔴 MANDATORY CLINICAL PROTOCOL:** The diagnostic inference and ground truth detection simulated by this AI framework are strictly for academic evaluation and primary screening reference. Users must **not solely rely on these automated predictions**. A formal, conclusive diagnosis must always be conducted via physical clinical examination by a certified **Medical Ophthalmologist**.
            """
        )
    
    gr.Markdown("---")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(type="pil", label="Patient Fundus Scanning Workspace")
            gr.Markdown("<br>") 
            status_text = gr.Textbox(label="Simulation Streaming Status (Ground Truth Verification)", placeholder="Awaiting trigger protocol...")
            random_btn = gr.Button("🎲 Load Random Frame From Private Test Dataset", variant="primary")
            submit_btn = gr.Button("🔬 Run Diagnostic Inference", variant="secondary")
            
        with gr.Column(scale=1):
            output_lbl = gr.Label(num_top_classes=3, label="Clinical Diagnostic Map (Confidence Scores)")
            
    random_btn.click(fn=load_random_from_private_zip, inputs=None, outputs=[input_img, status_text])
    submit_btn.click(fn=predict_fundus, inputs=input_img, outputs=output_lbl)

if __name__ == "__main__":
    demo.launch()