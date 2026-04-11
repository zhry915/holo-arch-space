"""
Holo-Arch Recognition System - Hugging Face Spaces Deployment
Main application with Gradio + FastAPI integration.
"""

import os
import sys
import json
import base64
import warnings
from pathlib import Path
from io import BytesIO

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
import h5py

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse
from gradio.mount_gradio_app import MountGradioAPP
import gradio as gr

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from holo_mobilenet import get_holo_mobilenet_v3
from holo_pgf_engine import HoloPGFEngine, holo_cross_entropy
from aesthetic_fft_sketch import generate_sketch_from_pil

# ==================== Configuration ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = None
CLASS_MAP = {}
ID_TO_NAME = {}
NAME_TO_EN = {}

# Paths
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "mobilenet_v3_arch_spectral_best.pth"
H5_PATH = BASE_DIR / "data" / "arch_dataset_v1_cleaned.h5"
HTML_PATH = BASE_DIR / "index.html"
CSV_PATH = BASE_DIR / "src" / "spectral_landscape_adamw_log.csv"

# ==================== Model Loading ====================
def load_model():
    global MODEL, CLASS_MAP, ID_TO_NAME, NAME_TO_EN
    
    num_classes = 21
    CLASS_MAP = {}
    
    # Try to load from H5 file if exists
    if H5_PATH.exists():
        try:
            with h5py.File(H5_PATH, 'r') as f:
                if 'class_map' in f.attrs:
                    CLASS_MAP = json.loads(f.attrs['class_map'])
                    num_classes = len(CLASS_MAP)
                    for k, v in CLASS_MAP.items():
                        ID_TO_NAME[int(k)] = v.get('chinese_name', f"Style {k}")
                        NAME_TO_EN[v.get('chinese_name', '')] = v.get('english_name', f"Style {k}")
        except Exception as e:
            print(f"Warning: Could not load class map from H5: {e}")
    
    # If no class map, use default Chinese architectural styles
    if not CLASS_MAP:
        default_styles = [
            "北京官式琉璃瓦", "北京四合院", "内蒙古传统毡房", "福建客家土楼", 
            "江西赣南围屋", "山东胶东海草房", "河南地坑院", "湖南湘西吊脚楼",
            "广东岭南骑楼", "广西三江风雨桥", "云南傣族竹楼", "陕西陕北窑洞",
            "安徽徽派民居", "四川川西碉堡", "西藏藏式碉房", "苏式园林建筑",
            "新疆阿以旺式民居", "山西晋商大院", "湖北武当山古建筑群", "东北民居",
            "台湾闽南古厝"
        ]
        for i, name in enumerate(default_styles):
            CLASS_MAP[str(i)] = {
                'chinese_name': name,
                'english_name': f"Style {i+1}"
            }
            ID_TO_NAME[i] = name
            NAME_TO_EN[name] = f"Style {i+1}"
    
    # Load model
    MODEL = get_holo_mobilenet_v3(num_classes=num_classes)
    
    if MODEL_PATH.exists():
        try:
            MODEL.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            print(f"Model loaded successfully from {MODEL_PATH}")
        except Exception as e:
            print(f"Warning: Could not load model weights: {e}")
            print("Using randomly initialized model for demo.")
    else:
        print(f"Warning: Model file not found at {MODEL_PATH}")
        print("Using randomly initialized model for demo.")
    
    MODEL = MODEL.to(DEVICE)
    MODEL.eval()

# ==================== FastAPI App ====================
app_fastapi = FastAPI(title="Holo-Arch Engine API")

@app_fastapi.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve the high-end HTML UI."""
    if HTML_PATH.exists():
        with open(HTML_PATH, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    else:
        return HTMLResponse(content="""
        <html><body style="font-family: sans-serif; padding: 50px; text-align: center;">
            <h1>Holo-Arch Recognition Engine</h1>
            <p>Welcome to the Holo-Arch Chinese Architectural Recognition System.</p>
            <p>Please ensure all required files are uploaded.</p>
        </body></html>
        """)

@app_fastapi.get("/api/styles")
async def get_styles():
    """Returns the list of 21 architectural styles with representative images."""
    results = []
    
    if H5_PATH.exists():
        try:
            with h5py.File(H5_PATH, 'r') as f:
                class_map = json.loads(f.attrs['class_map'])
                labels = f['labels'][:]
                images = f['images']
                
                for class_id_str, class_info in class_map.items():
                    class_id = int(class_id_str)
                    style_name_zh = class_info.get('chinese_name', 'Unknown Style')
                    style_name_en = class_info.get('english_name', f"Style {class_id}")
                    
                    idx = np.where(labels == class_id)[0]
                    if len(idx) > 0:
                        img_idx = idx[0]
                        if "北京" in style_name_zh and len(idx) > 10:
                            img_idx = idx[9]
                        elif "四合院" in style_name_zh and len(idx) > 8:
                            img_idx = idx[7]
                        elif "蒙古毡房" in style_name_zh and len(idx) > 5:
                            img_idx = idx[4]
                        elif "徽派" in style_name_zh and len(idx) > 6:
                            img_idx = idx[5]
                        else:
                            img_idx = idx[1 % len(idx)]
                            
                        img_array = images[img_idx]
                        img = Image.fromarray(img_array.astype('uint8'), 'RGB')
                        buffered = BytesIO()
                        img.save(buffered, format="JPEG", quality=85)
                        img_str = base64.b64encode(buffered.getvalue()).decode()
                        
                        results.append({
                            "id": class_id,
                            "zh": style_name_zh,
                            "en": style_name_en,
                            "image_b64": f"data:image/jpeg;base64,{img_str}"
                        })
                return {"styles": results}
        except Exception as e:
            print(f"Error loading styles from H5: {e}")
    
    # Fallback: return styles without images
    for i in range(21):
        name_zh = ID_TO_NAME.get(i, f"风格 {i+1}")
        name_en = NAME_TO_EN.get(name_zh, f"Style {i+1}")
        results.append({
            "id": i,
            "zh": name_zh,
            "en": name_en,
            "image_b64": ""
        })
    return {"styles": results}

@app_fastapi.post("/api/predict")
async def predict_image(file: UploadFile = File(...)):
    """Predict architectural style and calculate analytical stress."""
    if MODEL is None:
        return {"error": "Model not loaded"}
        
    try:
        contents = await file.read()
        img = Image.open(BytesIO(contents)).convert('RGB')
        
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        
        input_tensor = preprocess(img).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            output = MODEL(input_tensor)
            
            temperature = 2.5 
            scaled_logits = output[0] / temperature
            probs = torch.nn.functional.softmax(scaled_logits, dim=0)
            pred_idx = torch.argmax(probs).item()
            confidence = probs[pred_idx].item()
            
            style_zh = ID_TO_NAME.get(pred_idx, "Unknown")
            style_en = NAME_TO_EN.get(style_zh, "Unknown")
        
        # Extract Psi4 stress
        engine = HoloPGFEngine(MODEL, M=8, eta=1e-3)
        input_tensor.requires_grad = True
        target_label = torch.tensor([pred_idx]).to(DEVICE)
        
        try:
            psi_dict = engine.extract_spectrum(input_tensor, target_label, nn.CrossEntropyLoss())
            psi4_val = float(psi_dict[4].real)
        except Exception as e:
            print(f"Engine extraction failed: {e}")
            psi4_val = 0.0
            
        return {
            "prediction": {
                "id": pred_idx,
                "zh": style_zh,
                "en": style_en,
                "confidence": float(confidence)
            },
            "psi4_stress": psi4_val,
            "message": "success"
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

@app_fastapi.post("/api/sketch")
async def generate_sketch(file: UploadFile = File(...)):
    """Generate FFT-based architectural sketch."""
    try:
        contents = await file.read()
        img = Image.open(BytesIO(contents)).convert('RGB')
        
        sketch_b64 = generate_sketch_from_pil(img, low_cutoff=30, high_cutoff=150)
        
        return {
            "sketch_b64": sketch_b64,
            "message": "success"
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

@app_fastapi.get("/api/analytics/landscape")
async def get_landscape_data():
    """Compute 3D spectral landscape visualization data."""
    if not CSV_PATH.exists():
        return {"error": f"Log file not found at {CSV_PATH}"}
        
    try:
        df = pd.read_csv(CSV_PATH)
        df = df[df['step'] != 'EPOCH_END']
        df = df.rename(columns={'train_loss': 'loss'})
        for col in ['loss', 'psi1', 'psi2', 'psi3', 'psi4']:
            df[col] = df[col].astype(float)
        df = df.reset_index(drop=True)
        
        num_points = len(df)
        traj_x = np.zeros(num_points)
        traj_y = np.zeros(num_points)
        
        curr_theta = 0.0
        psi1_vals = np.abs(df['psi1'].values)
        psi1_norm = (psi1_vals - psi1_vals.min()) / (psi1_vals.max() - psi1_vals.min() + 1e-8)
        
        loss_vals = df['loss'].values
        r_vals = 0.1 + 0.8 * (loss_vals - loss_vals.min()) / (loss_vals.max() - loss_vals.min() + 1e-8)
        
        for i in range(num_points):
            curr_r = r_vals[i]
            steering_force = psi1_norm[i] * 0.3 
            curr_theta += 0.15 + steering_force 
            traj_x[i] = curr_r * np.cos(curr_theta)
            traj_y[i] = curr_r * np.sin(curr_theta)
            
        res = 60
        grid_range = 1.2
        gx = np.linspace(-grid_range, grid_range, res)
        gy = np.linspace(-grid_range, grid_range, res)
        GX, GY = np.meshgrid(gx, gy)
        
        Z_local_sum = np.zeros_like(GX)
        W_sum = np.zeros_like(GX)
        
        init_row = df.iloc[0]
        p0_init = np.log10(init_row['loss'] + 1e-12)
        p1_init, p2_init, p4_init = init_row['psi1'], init_row['psi2'], init_row['psi4']
        
        plateau_base = p0_init + np.log1p(abs(p4_init)) * 0.02 
        
        DX_bg = GX - traj_x[0]
        DY_bg = GY - traj_y[0]
        R_bg = np.sqrt(DX_bg**2 + DY_bg**2)
        Z_background = p0_init + (p1_init * R_bg + 0.5 * p2_init * (R_bg**2)) * 0.1
        Z_background = np.clip(Z_background, plateau_base - 0.5, plateau_base + 1.0)
        
        base_sigma_tangent = 0.15 
        base_sigma_normal = 0.04  
        
        dx_dt = np.gradient(traj_x)
        dy_dt = np.gradient(traj_y)
        
        for i, row in df.iterrows():
            cx, cy = traj_x[i], traj_y[i]
            tx, ty = dx_dt[i], dy_dt[i]
            t_norm = np.sqrt(tx**2 + ty**2 + 1e-10)
            tx, ty = tx/t_norm, ty/t_norm
            nx, ny = -ty, tx
            
            p0 = np.log10(row['loss'] + 1e-12)
            p1, p2, p3, p4 = row['psi1'], row['psi2'], row['psi3'], row['psi4']
            
            stability_factor = 1.0 / (1.0 + np.log1p(abs(p2)))
            curr_sigma_tangent = base_sigma_tangent * (0.5 + 1.5 * stability_factor)
            
            ruggedness_impact = np.log1p(abs(p4)) / 20.0 
            curr_sigma_normal = base_sigma_normal / (1.0 + ruggedness_impact)
            
            DX = GX - cx
            DY = GY - cy
            R = np.sqrt(DX**2 + DY**2)
            
            dist_t = DX * tx + DY * ty
            dist_n = DX * nx + DY * ny
            
            r_ref = 0.4
            raw_wall_val = abs(p1*r_ref + 0.5*p2*(r_ref**2) + (1/6)*p3*(r_ref**3) + (1/24)*p4*(r_ref**4))
            scale = 0.8 / (raw_wall_val + 1e-6)
            scale = min(scale, 0.5) 
            
            local_potential = scale * (p1*R + 0.5*p2*(R**2) + (1/6)*p3*(R**3) + (1/24)*p4*(R**4))
            weight = np.exp(-(dist_t**2 / (2 * curr_sigma_tangent**2) + dist_n**2 / (2 * curr_sigma_normal**2)))
            
            Z_local_sum += (p0 + local_potential) * weight
            W_sum += weight
            
        Z_blended = Z_local_sum / (W_sum + 1e-10)
        confidence = 1.0 - np.exp(-W_sum * 0.5)
        Z_final = Z_background * (1 - confidence) + Z_blended * confidence
        
        traj_z_glued = []
        for tx, ty in zip(traj_x, traj_y):
            ix = np.argmin(np.abs(gx - tx))
            iy = np.argmin(np.abs(gy - ty))
            traj_z_glued.append(Z_final[iy, ix])
            
        psi4_vals = np.abs(df['psi4'].values)
        psi4_norm = np.log1p(psi4_vals) / np.log1p(psi4_vals.max() + 1e-8)
        
        return {
            "x": gx.tolist(),
            "y": gy.tolist(),
            "z": Z_final.tolist(),
            "traj_x": traj_x.tolist(),
            "traj_y": traj_y.tolist(),
            "traj_z": traj_z_glued,
            "loss": loss_vals.tolist(),
            "psi4_norm": psi4_norm.tolist()
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

@app_fastapi.get("/api/analytics/corridor")
async def get_corridor_data():
    """Returns analytical corridor data."""
    steps = 15
    t_values = np.linspace(0, 1, steps).tolist()
    
    tas_linear = [5.0 + 3.0 * np.exp(-((t - 0.5) ** 2) / 0.05) + np.random.normal(0, 0.2) for t in t_values]
    tas_corridor = [5.0 + 0.8 * np.exp(-((t - 0.5) ** 2) / 0.1) + np.random.normal(0, 0.1) for t in t_values]
    
    return {
        "t": t_values,
        "tas_linear": tas_linear,
        "tas_corridor": tas_corridor,
        "source": "simulated"
    }

# ==================== Gradio Interface ====================
def gradio_predict(image):
    """Gradio interface for prediction."""
    if image is None:
        return {"error": "No image provided"}
    
    try:
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        
        input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            output = MODEL(input_tensor)
            
            temperature = 2.5 
            scaled_logits = output[0] / temperature
            probs = torch.nn.functional.softmax(scaled_logits, dim=0)
            pred_idx = torch.argmax(probs).item()
            confidence = probs[pred_idx].item()
            
            style_zh = ID_TO_NAME.get(pred_idx, "Unknown")
            style_en = NAME_TO_EN.get(style_zh, "Unknown")
        
        # Extract Psi4 stress
        engine = HoloPGFEngine(MODEL, M=8, eta=1e-3)
        input_tensor.requires_grad = True
        target_label = torch.tensor([pred_idx]).to(DEVICE)
        
        try:
            psi_dict = engine.extract_spectrum(input_tensor, target_label, nn.CrossEntropyLoss())
            psi4_val = float(psi_dict[4].real)
        except:
            psi4_val = 0.0
        
        result = f"""
## 识别结果 / Recognition Result

**建筑风格 / Style:** {style_zh} ({style_zh})

**英文名 / English Name:** {style_en}

**置信度 / Confidence:** {confidence * 100:.2f}%

**解析应力 / Analytical Stress (Ψ₄):** {psi4_val:.2e}
"""
        return result
        
    except Exception as e:
        return f"Error: {str(e)}"

def gradio_sketch(image):
    """Gradio interface for sketch generation."""
    if image is None:
        return {"error": "No image provided"}
    
    try:
        sketch_b64 = generate_sketch_from_pil(image, low_cutoff=30, high_cutoff=150)
        from PIL import Image as PILImage
        import re
        
        # Extract base64 data
        match = re.search(r'base64,(.*)', sketch_b64)
        if match:
            img_data = base64.b64decode(match.group(1))
            return PILImage.open(BytesIO(img_data))
        return sketch_b64
        
    except Exception as e:
        return f"Error: {str(e)}"

# Gradio Blocks interface
with gr.Blocks(title="Holo-Arch Recognition Engine", theme=gr.themes.Soft()) as gradio_app:
    gr.Markdown("""
    # Holo-Arch 识别引擎 / Recognition Engine
    
    基于全纯逻辑同构 (Holomorphic Logic Isomorphism) 的中国古建筑风格识别系统。
    
    **Supported Styles:** 21 Chinese Architectural Styles
    """)
    
    with gr.Tabs():
        with gr.TabItem("全息雷达 / Holo Radar"):
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(type="pil", label="上传建筑图片 / Upload Architecture Image")
                    predict_btn = gr.Button("识别 / Recognize", variant="primary")
                with gr.Column():
                    result_output = gr.Textbox(label="结果 / Result", lines=10)
            
            predict_btn.click(fn=gradio_predict, inputs=image_input, outputs=result_output)
        
        with gr.TabItem("水墨界画 / Sketch"):
            with gr.Row():
                with gr.Column():
                    sketch_input = gr.Image(type="pil", label="上传图片 / Upload Image")
                    sketch_btn = gr.Button("生成界画 / Generate Sketch", variant="primary")
                with gr.Column():
                    sketch_output = gr.Image(label="界画结果 / Sketch Result")
            
            sketch_btn.click(fn=gradio_sketch, inputs=sketch_input, outputs=sketch_output)
        
        with gr.TabItem("风格列表 / Style List"):
            gr.Markdown("""
            ## 支持的21种建筑风格 / 21 Supported Architectural Styles
            
            1. 北京官式琉璃瓦 (Beijing Official Glazed Tile)
            2. 北京四合院 (Beijing Courtyard House)
            3. 内蒙古传统毡房 (Mongolian Yurt)
            4. 福建客家土楼 (Fujian Hakka Tulou)
            5. 江西赣南围屋 (Jiangxi Gan-nan Weiwu)
            6. 山东胶东海草房 (Shandong Seaweed House)
            7. 河南地坑院 (Henan Cave Courtyard)
            8. 湖南湘西吊脚楼 (Hunan Stilted House)
            9. 广东岭南骑楼 (Guangdong Arcade House)
            10. 广西三江风雨桥 (Guangxi Wind-Rain Bridge)
            11. 云南傣族竹楼 (Yunnan Dai Bamboo House)
            12. 陕西陕北窑洞 (Shaanxi Loess Cave)
            13. 安徽徽派民居 (Anhui Huizhou Style)
            14. 四川川西碉堡 (Sichuan Tibetan-style Tower)
            15. 西藏藏式碉房 (Tibetan Stone House)
            16. 苏式园林建筑 (Suzhou Garden Architecture)
            17. 新疆阿以旺式民居 (Xinjiang Ayiwan House)
            18. 山西晋商大院 (Shanxi Merchant Manor)
            19. 湖北武当山古建筑群 (Wudang Mountain Complex)
            20. 东北民居 (Northeast Chinese House)
            21. 台湾闽南古厝 (Taiwan Minnan Traditional House)
            """)

# ==================== Main Entry Point ====================
print("[Holo-Arch] Initializing model...")
load_model()
print("[Holo-Arch] Model loaded successfully.")

# Mount Gradio app to FastAPI
app = MountGradioAPP(app_fastapi, gradio_app, "/gradio")

if __name__ == "__main__":
    import uvicorn
    print("Starting Holo-Arch High-End UI Server...")
    print("Please open your browser at: http://127.0.0.1:7860")
    uvicorn.run(app, host="0.0.0.0", port=7860)
