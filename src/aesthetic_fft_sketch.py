"""
Aesthetic FFT Sketch (Jiehua/White Painting) Generator
Modified for Hugging Face Spaces deployment.
"""

import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import cv2
import tempfile
import os

def create_fft_sketch(image_path, output_path, low_cutoff=20, high_cutoff=150):
    """
    Convert an image to a traditional Chinese architectural sketch using FFT.
    """
    img = Image.open(image_path).convert('L')
    
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.5)
    
    img_array = np.array(img)
    
    f = np.fft.fft2(img_array)
    fshift = np.fft.fftshift(f)
    
    rows, cols = img_array.shape
    crow, ccol = rows // 2, cols // 2
    
    x = np.linspace(-ccol, ccol - 1, cols) if cols % 2 == 0 else np.linspace(-ccol, ccol, cols)
    y = np.linspace(-crow, crow - 1, rows) if rows % 2 == 0 else np.linspace(-crow, crow, rows)
    X, Y = np.meshgrid(x, y)
    
    D = np.sqrt(X**2 + Y**2)
    
    n = 2
    mask_high = 1 / (1 + (low_cutoff / (D + 1e-5))**(2 * n))
    mask_low = 1 / (1 + (D / (high_cutoff + 1e-5))**(2 * n))
    mask = mask_high * mask_low
    
    fshift_filtered = fshift * mask
    
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    
    img_back = np.abs(img_back)
    
    img_back = (img_back - np.min(img_back)) / (np.max(img_back) - np.min(img_back)) * 255
    img_back = img_back.astype(np.uint8)
    
    sketch = ImageOps.invert(Image.fromarray(img_back))
    sketch_np = np.array(sketch)
    
    sketch_smooth = cv2.bilateralFilter(sketch_np, d=5, sigmaColor=75, sigmaSpace=75)
    
    sketch_binary = cv2.adaptiveThreshold(
        sketch_smooth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 15, 8
    )
    
    kernel = np.ones((2, 2), np.uint8)
    sketch_connected = cv2.erode(sketch_binary, kernel, iterations=1)
    
    final_np = cv2.GaussianBlur(sketch_connected, (3, 3), 0)
    
    final_sketch = Image.fromarray(final_np)
    final_sketch.save(output_path)

def generate_sketch_from_pil(img: Image.Image, low_cutoff=30, high_cutoff=150):
    """
    Generate sketch directly from PIL Image and return as base64 string.
    """
    import base64
    from io import BytesIO
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_in:
        img.save(tmp_in.name)
        tmp_in_path = tmp_in.name
    
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_out:
        tmp_out_path = tmp_out.name
    
    try:
        create_fft_sketch(tmp_in_path, tmp_out_path, low_cutoff, high_cutoff)
        
        with open(tmp_out_path, 'rb') as f:
            encoded_string = base64.b64encode(f.read()).decode('utf-8')
        
        return f"data:image/jpeg;base64,{encoded_string}"
    finally:
        if os.path.exists(tmp_in_path):
            os.remove(tmp_in_path)
        if os.path.exists(tmp_out_path):
            os.remove(tmp_out_path)
