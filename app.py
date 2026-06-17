import streamlit as st
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import numpy as np
import cv2
import io

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="AI Image Editor",
    page_icon="🖼️",
    layout="wide"
)

st.title("🖼️ AI Image Editor")
st.markdown("*Upload, edit, and enhance your images with AI-powered tools*")

# ============================================
# SIDEBAR: TOOLS
# ============================================
with st.sidebar:
    st.header("🎛️ Editing Tools")
    
    uploaded_file = st.file_uploader(
        "Upload Image",
        type=['jpg', 'jpeg', 'png', 'bmp', 'webp']
    )
    
    st.markdown("---")
    
    tool = st.radio(
        "Select Tool",
        [
            "Original",
            "Grayscale",
            "Sepia",
            "Blur",
            "Sharpen",
            "Edge Detection",
            "Cartoonify",
            "Brightness/Contrast",
            "Super Resolution"
        ]
    )
    
    st.markdown("---")
    
    if tool == "Blur":
        blur_amount = st.slider("Blur Intensity", 1, 10, 3)
    elif tool == "Sharpen":
        sharpen_amount = st.slider("Sharpen Intensity", 1.0, 5.0, 2.0)
    elif tool == "Brightness/Contrast":
        brightness = st.slider("Brightness", 0.5, 2.0, 1.0, 0.1)
        contrast = st.slider("Contrast", 0.5, 2.0, 1.0, 0.1)

# ============================================
# IMAGE PROCESSING FUNCTIONS
# ============================================
def apply_grayscale(image):
    return ImageOps.grayscale(image)

def apply_sepia(image):
    grayscale = ImageOps.grayscale(image)
    sepia = Image.merge(
        'RGB',
        (
            Image.eval(grayscale, lambda x: min(int(x * 1.2), 255)),
            Image.eval(grayscale, lambda x: min(int(x * 0.9), 255)),
            Image.eval(grayscale, lambda x: min(int(x * 0.6), 255))
        )
    )
    return sepia

def apply_blur(image, amount):
    return image.filter(ImageFilter.GaussianBlur(radius=amount))

def apply_sharpen(image, amount):
    enhancer = ImageEnhance.Sharpness(image)
    return enhancer.enhance(amount)

def apply_edge_detection(image):
    img_cv = np.array(image)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return Image.fromarray(edges)

def apply_cartoonify(image):
    img_cv = np.array(image)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(img_cv, 9, 300, 300)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return Image.fromarray(cartoon)

def apply_brightness_contrast(image, brightness, contrast):
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast)
    return image

def apply_super_resolution(image):
    width, height = image.size
    return image.resize((width * 2, height * 2), Image.Resampling.LANCZOS)

# ============================================
# MAIN APP LOGIC
# ============================================
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    original = image.copy()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📷 Original Image")
        st.image(original, use_container_width=True)
        st.caption(f"Size: {original.size[0]} x {original.size[1]} pixels")
    
    with col2:
        st.markdown("### 🎨 Edited Image")
        
        try:
            edited_image = original.copy()
            
            if tool == "Original":
                edited_image = original.copy()
            elif tool == "Grayscale":
                edited_image = apply_grayscale(original)
            elif tool == "Sepia":
                edited_image = apply_sepia(original)
            elif tool == "Blur":
                edited_image = apply_blur(original, blur_amount)
            elif tool == "Sharpen":
                edited_image = apply_sharpen(original, sharpen_amount)
            elif tool == "Edge Detection":
                edited_image = apply_edge_detection(original)
            elif tool == "Cartoonify":
                edited_image = apply_cartoonify(original)
            elif tool == "Brightness/Contrast":
                edited_image = apply_brightness_contrast(original, brightness, contrast)
            elif tool == "Super Resolution":
                edited_image = apply_super_resolution(original)
            
            st.image(edited_image, use_container_width=True)
            
            img_bytes = io.BytesIO()
            edited_image.save(img_bytes, format='PNG')
            img_bytes = img_bytes.getvalue()
            
            st.download_button(
                "💾 Download Edited Image",
                img_bytes,
                file_name="edited_image.png",
                mime="image/png"
            )
            
        except Exception as e:
            st.error(f"Error: {str(e)}")

else:
    st.info("👈 Upload an image from the sidebar to get started!")

st.markdown("---")
st.caption("🖼️ AI Image Editor - Powered by OpenCV, PIL, and Streamlit")