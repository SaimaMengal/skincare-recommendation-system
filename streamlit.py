import streamlit as st
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Load the trained model
model = load_model('neural_mobilenet.h5')

# Corrected categories (alphabetical order as per LabelEncoder)
CATEGORIES = ['acne', 'dry', 'oily']  # Alphabetical order for correct mapping
PRODUCT_DIR = r'D:\python\skincare products'  # Update this path as needed

def get_product_images(skin_type):
    """Retrieve product images for the predicted skin type"""
    product_path = os.path.join(PRODUCT_DIR, f"{skin_type.lower()}skinproducts")
    if os.path.exists(product_path):
        return [os.path.join(product_path, f) for f in os.listdir(product_path) 
                if f.lower().endswith(('png', 'jpg', 'jpeg'))][:3]
    return []

def predict_skin_type(image):
    """Predict skin type from uploaded image"""
    # Convert to RGB if necessary
    image = image.resize((224, 224)).convert('RGB')
    
    # Correct preprocessing (matches training)
    image_array = img_to_array(image)
    image_array = image_array / 255.0  # Simple normalization
    
    image_array = np.expand_dims(image_array, axis=0)
    
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    return CATEGORIES[predicted_class].title()  # Capitalize for display

# Streamlit UI Configuration
st.set_page_config(page_title="Skin Type Detector", layout="wide", page_icon="✨")

# Custom CSS styling
st.markdown("""
    <style>
    .main {background: #fafafa;}
    .reportview-container .markdown { color: #2e4053; }
    .st-bb {background-color: #ffffff;}
    .st-at {background-color: #f0f2f6;}
    .footer {position: fixed; bottom: 0; width: 100%;}
    </style>
    """, unsafe_allow_html=True)

# Header Section
st.markdown("<h1 style='text-align: center; color: #2e4053;'>🧖 Skin Type Detection & Care</h1>", 
            unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #666;'>Upload a clear facial photo for analysis</h4><br>", 
            unsafe_allow_html=True)

# File Upload Section
uploaded_file = st.file_uploader(" ", type=["jpg", "png", "jpeg"], 
                               help="Upload a clear facial image for analysis")

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
        with col2:
            if st.button("🔍 Analyze Skin Type", use_container_width=True):
                with st.spinner("Analyzing your skin..."):
                    skin_type = predict_skin_type(image)
                    
                # Prediction Results
                st.markdown(f"""
                    <div style='padding: 20px; border-radius: 10px; 
                        background: #f0f2f6; margin-top: 20px;'>
                        <h3 style='color: #2e4053;'>🧪 Prediction Result</h3>
                        <h2 style='color: #2980b9;'>📜 {skin_type} Skin</h2>
                    </div>
                """, unsafe_allow_html=True)

                # Skin Care Tips
                tips = {
                    "Dry": "💧 Hydration is key! Use rich moisturizers and avoid hot water",
                    "Oily": "🌿 Use oil-free products and blotting papers throughout the day",
                    "Acne": "🩹 Maintain a consistent cleansing routine and avoid picking"
                }
                st.markdown(f"""
                    <div style='padding: 20px; border-radius: 10px; 
                        background: #fff3e6; margin: 20px 0;'>
                        <h4 style='color: #2e4053;'>💡 Care Tips</h4>
                        <p>{tips.get(skin_type, 'General care: Use SPF daily!')}</p>
                    </div>
                """, unsafe_allow_html=True)

                # Recommended Products
                product_images = get_product_images(skin_type.lower())
                if product_images:
                    st.markdown("### 🛍 Recommended Products")
                    cols = st.columns(len(product_images))
                    for col, img_path in zip(cols, product_images):
                        product_name = os.path.splitext(os.path.basename(img_path))[0]
                        with col:
                            st.image(img_path, use_column_width=True, 
                                   caption=product_name.replace('_', ' ').title())
                else:
                    st.warning("⚠️ No product recommendations available for this skin type")

    except Exception as e:
        st.error(f"Error processing image: {str(e)}")

# Footer
st.markdown("---")
st.markdown("<div class='footer' style='text-align: center; color: #666;'>"
            " </div>", 
            unsafe_allow_html=True)