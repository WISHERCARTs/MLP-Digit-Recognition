# ==================== MLP Digit Recognition - Streamlit App ====================
# Web Application สำหรับทำนายตัวเลขด้วย Neural Network
# รันด้วย: streamlit run app.py

import streamlit as st
import numpy as np
from PIL import Image
import pickle
import os
from scipy import ndimage
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
from streamlit_drawable_canvas import st_canvas

# ==================== Page Config ====================
st.set_page_config(
    page_title="🔢 MLP Digit Recognition",
    page_icon="🔢",
    layout="wide"
)

# ==================== Custom CSS ====================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        color: white;
        font-size: 4rem;
        font-weight: bold;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.4);
    }
    .info-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ==================== Load or Train Model ====================
@st.cache_resource
def load_model():
    """โหลด model จากไฟล์ หรือ train ใหม่ถ้าไม่มี"""
    model_path = 'mlp_model.pkl'
    
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        return model_data['model'], model_data['accuracy']
    else:
        # Train model ใหม่ - ใช้ fetch_openml แทน loadmat
        with st.spinner('🚀 กำลังดาวน์โหลด MNIST และ Train Model... (ครั้งแรกเท่านั้น)'):
            # ดาวน์โหลด MNIST จาก OpenML
            mnist = fetch_openml('mnist_784', version=1, as_frame=False)
            x = mnist.data.astype('float32') / 255.0  # Normalize 0-1
            y = mnist.target.astype('int')
            
            # Shuffle
            shuffle_idx = np.random.permutation(len(x))
            x, y = x[shuffle_idx], y[shuffle_idx]
            
            # Split
            x_train, x_test = x[:60000], x[60000:]
            y_train, y_test = y[:60000], y[60000:]
            
            # Train
            model = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=20, random_state=42)
            model.fit(x_train, y_train)
            
            # Evaluate
            y_pred = model.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred) * 100
            
            # Save model
            with open(model_path, 'wb') as f:
                pickle.dump({'model': model, 'accuracy': accuracy}, f)
            
            return model, accuracy

# ==================== Process Image ====================
def preprocess_image(image):
    """แปลงรูปให้เป็น format ที่ model ต้องการ (784 pixels)
    - จัดการ RGBA → Grayscale อย่างถูกต้อง
    - Center ตัวเลขตาม bounding box (เหมือน MNIST)
    - Normalize ค่า pixel ให้อยู่ 0-1
    """
    # Convert RGBA to grayscale properly
    img_array = np.array(image)
    
    if img_array.ndim == 3 and img_array.shape[2] == 4:
        # Canvas RGBA: ใช้ RGB channels เพื่อดึงเส้นที่วาด
        # (alpha = 255 ทั้งพื้นและเส้น จึงใช้แยกไม่ได้)
        # พื้นดำ(0) + เส้นขาว(255) = ตรงกับ MNIST format เลย
        r = img_array[:, :, 0].astype(np.float32)
        g = img_array[:, :, 1].astype(np.float32)
        b = img_array[:, :, 2].astype(np.float32)
        gray = 0.299 * r + 0.587 * g + 0.114 * b
    else:
        # รูปปกติ: convert to grayscale
        img = image.convert('L')
        gray = np.array(img, dtype=np.float32)
        # Invert: MNIST = ขาวบนดำ, รูปปกติ = ดำบนขาว
        gray = 255.0 - gray
    
    # ---- Center digit using bounding box (เหมือนวิธี MNIST) ----
    # หา bounding box ของตัวเลข
    threshold = 30
    coords = np.where(gray > threshold)
    
    if len(coords[0]) == 0:
        # ไม่มีเส้นที่วาด → return blank
        img_final = np.zeros((28, 28), dtype=np.float32)
        return img_final.flatten().reshape(1, -1), img_final.astype(np.uint8)
    
    # Crop ตัวเลขออกมา
    top, bottom = coords[0].min(), coords[0].max()
    left, right = coords[1].min(), coords[1].max()
    digit = gray[top:bottom+1, left:right+1]
    
    # Resize ให้พอดี 20x20 (MNIST มี digit 20x20 อยู่กลางรูป 28x28)
    h, w = digit.shape
    if h > w:
        new_h = 20
        new_w = max(1, int(w * (20.0 / h)))
    else:
        new_w = 20
        new_h = max(1, int(h * (20.0 / w)))
    
    digit_pil = Image.fromarray(digit.astype(np.uint8))
    digit_resized = digit_pil.resize((new_w, new_h), Image.LANCZOS)
    digit_resized = np.array(digit_resized, dtype=np.float32)
    
    # วางตรงกลางรูป 28x28 (padding 4px รอบๆ)
    img_28 = np.zeros((28, 28), dtype=np.float32)
    pad_top = (28 - new_h) // 2
    pad_left = (28 - new_w) // 2
    img_28[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = digit_resized
    
    # Shift ไปที่ center of mass (เทคนิคเดียวกับ MNIST)
    cy, cx = ndimage.center_of_mass(img_28)
    shift_y = 14 - cy
    shift_x = 14 - cx
    img_28 = ndimage.shift(img_28, [shift_y, shift_x], mode='constant', cval=0)
    
    # Normalize 0-1 (ให้ตรงกับ training data)
    img_28 = img_28 / 255.0
    
    # สำหรับแสดงผล
    img_display = (img_28 * 255).astype(np.uint8)
    
    # Flatten to 784
    img_flat = img_28.flatten().reshape(1, -1)
    return img_flat, img_display

# ==================== Main App ====================
def main():
    # Header
    st.markdown('<h1 class="main-header">🔢 MLP Digit Recognition</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">ระบบจดจำตัวเลขด้วย Neural Network (Multi-Layer Perceptron)</p>', unsafe_allow_html=True)
    
    # Load model
    model, accuracy = load_model()
    
    # Metrics row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🎯 Model Accuracy", f"{accuracy:.2f}%")
    with col2:
        st.metric("🧠 Hidden Layers", "2 (256, 128)")
    with col3:
        st.metric("📊 Training Data", "60,000 images")
    
    st.markdown("---")
    
    # Main content
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.subheader("✏️ วาดตัวเลข หรือ อัปโหลดรูป")
        
        tab1, tab2 = st.tabs(["🖌️ วาดตัวเลข", "📁 อัปโหลดรูป"])
        
        with tab1:
            st.info("วาดตัวเลข 0-9 ในกรอบด้านล่าง")
            
            # Drawing canvas
            canvas_result = st_canvas(
                fill_color="rgba(0, 0, 0, 0)",
                stroke_width=15,
                stroke_color="white",
                background_color="#000000",
                height=280,
                width=280,
                drawing_mode="freedraw",
                key="canvas",
            )
            
            if st.button("🔮 ทำนาย", key="predict_draw", use_container_width=True):
                if canvas_result.image_data is not None:
                    # Convert canvas to PIL Image
                    img = Image.fromarray(canvas_result.image_data.astype('uint8'))
                    img_processed, img_display = preprocess_image(img)
                    
                    # Predict
                    prediction = model.predict(img_processed)[0]
                    confidence = np.max(model.predict_proba(img_processed)) * 100
                    
                    st.session_state['prediction'] = int(prediction)
                    st.session_state['confidence'] = confidence
                    st.session_state['img_display'] = img_display
        
        with tab2:
            uploaded_file = st.file_uploader("เลือกรูปตัวเลข", type=['png', 'jpg', 'jpeg'])
            
            if uploaded_file is not None:
                img = Image.open(uploaded_file)
                st.image(img, caption="รูปที่อัปโหลด", width=200)
                
                if st.button("🔮 ทำนาย", key="predict_upload", use_container_width=True):
                    img_processed, img_display = preprocess_image(img)
                    
                    prediction = model.predict(img_processed)[0]
                    confidence = np.max(model.predict_proba(img_processed)) * 100
                    
                    st.session_state['prediction'] = int(prediction)
                    st.session_state['confidence'] = confidence
                    st.session_state['img_display'] = img_display
    
    with col_right:
        st.subheader("🎯 ผลการทำนาย")
        
        if 'prediction' in st.session_state:
            # Show prediction
            st.markdown(f'''
            <div class="prediction-box">
                {st.session_state['prediction']}
            </div>
            ''', unsafe_allow_html=True)
            
            st.markdown(f"<p style='text-align: center; font-size: 1.2rem; margin-top: 1rem;'>Confidence: <b>{st.session_state['confidence']:.1f}%</b></p>", unsafe_allow_html=True)
            
            # Show processed image
            if 'img_display' in st.session_state:
                st.markdown("##### 🖼️ รูปที่ประมวลผล (28x28)")
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(3, 3))
                ax.imshow(st.session_state['img_display'], cmap='gray')
                ax.axis('off')
                st.pyplot(fig)
        else:
            st.markdown('''
            <div style="text-align: center; padding: 3rem; color: #888;">
                <p style="font-size: 4rem;">❓</p>
                <p>วาดหรืออัปโหลดรูปตัวเลข<br>แล้วกดปุ่ม "ทำนาย"</p>
            </div>
            ''', unsafe_allow_html=True)
    
    # About section
    st.markdown("---")
    with st.expander("ℹ️ เกี่ยวกับ MLP (Multi-Layer Perceptron)"):
        st.markdown("""
        ### 🧠 MLP คืออะไร?
        
        **Multi-Layer Perceptron (MLP)** คือโครงข่ายประสาทเทียมแบบหลายชั้น ประกอบด้วย:
        
        1. **Input Layer** - รับข้อมูลเข้า (784 pixels จากรูป 28×28)
        2. **Hidden Layers** - ประมวลผลและเรียนรู้ patterns
        3. **Output Layer** - ผลลัพธ์สุดท้าย (10 neurons สำหรับตัวเลข 0-9)
        
        ### 🔄 Flow การทำงาน
        ```
        รูปตัวเลข (28×28) → Flatten (784) → MLP → ทำนายตัวเลข 0-9
        ```
        
        ### 📊 Dataset
        - **MNIST** - ชุดข้อมูลรูปตัวเลขเขียนมือ 70,000 รูป
        - Train: 60,000 รูป | Test: 10,000 รูป
        """)

if __name__ == "__main__":
    main()
