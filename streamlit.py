import streamlit as st
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# ── Model ────────────────────────────────────────────────────────────────────
model = load_model('neural_mobilenet.h5')
CATEGORIES = ['acne', 'dry', 'oily']

# Product images live in root-level folder "skincare products"
# Sub-folders: acne / dry / oily  (files named acneskinproducts.jpg etc.)
PRODUCT_DIR = 'skincare products'

# ── Helpers ──────────────────────────────────────────────────────────────────
def get_product_images(skin_type: str):
    """Return up to 3 product image paths for the predicted skin type."""
    folder = os.path.join(PRODUCT_DIR, skin_type.lower())
    if not os.path.exists(folder):
        return []
    valid_ext = ('.png', '.jpg', '.jpeg')
    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(valid_ext)
    ]
    return sorted(files)[:3]


def predict_skin_type(image: Image.Image) -> str:
    """Return title-cased skin-type label for the uploaded image."""
    img = image.resize((224, 224)).convert('RGB')
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr)
    idx = int(np.argmax(preds, axis=1)[0])
    return CATEGORIES[idx].title()


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Luminae — Skin Analysis",
    layout="wide",
    page_icon=None,
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;600&family=DM+Sans:wght@300;400;500&display=swap');

/* Reset & base */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #F7F4F0 !important;
    color: #1C1C1C;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── Top bar ── */
.topbar {
    background: #1C1C1C;
    padding: 18px 48px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.topbar-brand {
    font-family: 'Cormorant Garamond', serif;
    font-size: 22px;
    font-weight: 600;
    color: #F0EBE3;
    letter-spacing: 4px;
    text-transform: uppercase;
}
.topbar-sub {
    font-size: 11px;
    color: #888;
    letter-spacing: 2px;
    text-transform: uppercase;
}

/* ── Hero section ── */
.hero {
    background: #EDE8E1;
    padding: 64px 48px 48px;
    border-bottom: 1px solid #D8D0C5;
}
.hero-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: clamp(36px, 5vw, 60px);
    font-weight: 300;
    color: #1C1C1C;
    line-height: 1.15;
    margin: 0 0 16px;
}
.hero-desc {
    font-size: 14px;
    color: #6B6560;
    max-width: 480px;
    line-height: 1.8;
    font-weight: 300;
}

/* ── Content wrapper ── */
.content {
    padding: 48px 48px 80px;
}

/* ── Upload zone ── */
.upload-label {
    font-size: 11px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #6B6560;
    margin-bottom: 12px;
    display: block;
}

/* Override Streamlit uploader */
[data-testid="stFileUploader"] {
    background: #FFFFFF !important;
    border: 1px solid #D8D0C5 !important;
    border-radius: 2px !important;
    padding: 24px !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #1C1C1C !important;
}

/* ── Button ── */
.stButton > button {
    background: #1C1C1C !important;
    color: #F0EBE3 !important;
    border: none !important;
    border-radius: 2px !important;
    padding: 14px 32px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 12px !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    font-weight: 500 !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: background 0.2s ease !important;
    margin-top: 16px !important;
}
.stButton > button:hover {
    background: #3A3A3A !important;
}

/* ── Result card ── */
.result-card {
    background: #FFFFFF;
    border: 1px solid #D8D0C5;
    border-radius: 2px;
    padding: 32px;
    margin-top: 24px;
}
.result-label {
    font-size: 10px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #9B8F86;
    margin-bottom: 8px;
}
.result-type {
    font-family: 'Cormorant Garamond', serif;
    font-size: 42px;
    font-weight: 400;
    color: #1C1C1C;
    margin: 0 0 24px;
    line-height: 1;
}
.divider {
    height: 1px;
    background: #EDE8E1;
    margin: 20px 0;
}
.tip-label {
    font-size: 10px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #9B8F86;
    margin-bottom: 10px;
}
.tip-text {
    font-size: 14px;
    color: #3A3532;
    line-height: 1.8;
    font-weight: 300;
}

/* ── Products section ── */
.section-label {
    font-size: 10px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #9B8F86;
    margin: 40px 0 20px;
    display: block;
}
.product-name {
    font-size: 12px;
    color: #6B6560;
    letter-spacing: 1px;
    text-align: center;
    margin-top: 8px;
    text-transform: capitalize;
}

/* ── Uploaded image ── */
[data-testid="stImage"] img {
    border-radius: 2px;
    border: 1px solid #D8D0C5;
}

/* ── Warning / error ── */
.stAlert {
    border-radius: 2px !important;
    font-size: 13px !important;
}

/* ── Spinner ── */
.stSpinner > div {
    border-top-color: #1C1C1C !important;
}
</style>
""", unsafe_allow_html=True)

# ── Top bar ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
    <span class="topbar-brand">Luminae</span>
    <span class="topbar-sub">Skin Analysis System</span>
</div>
""", unsafe_allow_html=True)

# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">Understand<br>Your Skin</div>
    <div class="hero-desc">
        Upload a clear facial photograph. Our model analyses your skin type
        and surfaces targeted care recommendations — in seconds.
    </div>
</div>
""", unsafe_allow_html=True)

# ── Main content ─────────────────────────────────────────────────────────────
st.markdown('<div class="content">', unsafe_allow_html=True)

col_upload, col_gap, col_result = st.columns([5, 1, 6])

with col_upload:
    st.markdown('<span class="upload-label">Upload Facial Image</span>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        label="",
        type=["jpg", "jpeg", "png"],
        help="Use a well-lit, front-facing photo for best accuracy.",
        label_visibility="collapsed",
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)
        analyze = st.button("Analyse Skin Type")

with col_result:
    if uploaded_file and analyze:
        with st.spinner("Analysing..."):
            skin_type = predict_skin_type(image)

        tips = {
            "Dry":  "Prioritise rich, occlusive moisturisers. Avoid hot water and harsh cleansers. "
                    "Look for ingredients such as hyaluronic acid, ceramides, and squalane.",
            "Oily": "Use lightweight, oil-free formulations. Incorporate a gentle salicylic acid "
                    "cleanser and a non-comedogenic SPF. Blotting papers help throughout the day.",
            "Acne": "Maintain a consistent, gentle cleansing routine twice daily. "
                    "Niacinamide and benzoyl peroxide can reduce inflammation. Avoid touching or picking.",
        }

        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Detected Skin Type</div>
            <div class="result-type">{skin_type}</div>
            <div class="divider"></div>
            <div class="tip-label">Care Recommendation</div>
            <div class="tip-text">{tips.get(skin_type, 'Apply broad-spectrum SPF 30 or higher daily.')}</div>
        </div>
        """, unsafe_allow_html=True)

        # Products
        product_images = get_product_images(skin_type)
        if product_images:
            st.markdown('<span class="section-label">Recommended Products</span>',
                        unsafe_allow_html=True)
            prod_cols = st.columns(len(product_images))
            for col, img_path in zip(prod_cols, product_images):
                raw_name = os.path.splitext(os.path.basename(img_path))[0]
                display_name = raw_name.replace('_', ' ').replace('-', ' ').title()
                with col:
                    st.image(img_path, use_column_width=True)
                    st.markdown(
                        f'<div class="product-name">{display_name}</div>',
                        unsafe_allow_html=True,
                    )
        else:
            st.info("Product images not found for this skin type. "
                    "Ensure the 'skincare products' folder is present in the repository.")

    elif not uploaded_file:
        st.markdown("""
        <div style="
            height: 320px;
            border: 1px dashed #C8C0B5;
            border-radius: 2px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #B0A89E;
            font-size: 13px;
            letter-spacing: 1px;
            text-transform: uppercase;
            margin-top: 36px;
        ">
            Results appear here
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
