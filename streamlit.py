import streamlit as st
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# ── Model ─────────────────────────────────────────────────────────────────────
model = load_model('neural_mobilenet.h5')
CATEGORIES = ['acne', 'dry', 'oily']
PRODUCT_DIR = 'skincare products'

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_product_images(skin_type: str):
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
    img = image.resize((224, 224)).convert('RGB')
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr)
    idx = int(np.argmax(preds, axis=1)[0])
    return CATEGORIES[idx].title()


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SkinSense — Skin Analysis",
    layout="wide",
    page_icon=None,
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;1,400&family=Nunito:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Nunito', sans-serif;
    background-color: #FDF6F9 !important;
    color: #3D2535;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }

.navbar {
    background: linear-gradient(135deg, #F2789F 0%, #E05C8A 100%);
    padding: 16px 52px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 2px 20px rgba(224,92,138,0.25);
}
.navbar-brand {
    font-family: 'Playfair Display', serif;
    font-size: 26px;
    font-weight: 600;
    color: #FFFFFF;
    letter-spacing: 2px;
}
.navbar-tagline {
    font-size: 12px;
    color: rgba(255,255,255,0.82);
    letter-spacing: 2px;
    text-transform: uppercase;
    font-weight: 300;
}

.hero {
    background: linear-gradient(160deg, #FEF0F5 0%, #FDE8F0 50%, #F9E0ED 100%);
    padding: 60px 52px 50px;
    border-bottom: 1px solid #F5C6D8;
}
.hero-badge {
    display: inline-block;
    background: rgba(242,120,159,0.15);
    color: #C4456E;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 3px;
    text-transform: uppercase;
    padding: 6px 16px;
    border-radius: 20px;
    border: 1px solid rgba(196,69,110,0.2);
    margin-bottom: 18px;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(38px, 5vw, 64px);
    font-weight: 600;
    color: #2E1420;
    line-height: 1.1;
    margin: 0 0 18px;
}
.hero-title em {
    font-style: italic;
    color: #E05C8A;
}
.hero-desc {
    font-size: 15px;
    color: #7A4B62;
    max-width: 500px;
    line-height: 1.9;
    font-weight: 300;
}
.hero-pills {
    display: flex;
    gap: 10px;
    margin-top: 28px;
    flex-wrap: wrap;
}
.pill {
    background: #FFFFFF;
    border: 1px solid #F0BDCF;
    border-radius: 20px;
    padding: 7px 18px;
    font-size: 12px;
    font-weight: 500;
    color: #C4456E;
}

.content { padding: 44px 52px 80px; }

.sec-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #C4456E;
    margin-bottom: 12px;
    display: block;
}

[data-testid="stFileUploader"] {
    background: #FFFFFF !important;
    border: 1.5px dashed #F0BDCF !important;
    border-radius: 16px !important;
    padding: 28px 20px !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #E05C8A !important;
}

.stButton > button {
    background: linear-gradient(135deg, #F2789F 0%, #E05C8A 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 30px !important;
    padding: 14px 36px !important;
    font-family: 'Nunito', sans-serif !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    width: 100% !important;
    cursor: pointer !important;
    margin-top: 18px !important;
    box-shadow: 0 4px 18px rgba(224,92,138,0.35) !important;
}
.stButton > button:hover {
    box-shadow: 0 6px 24px rgba(224,92,138,0.5) !important;
}

.result-card {
    background: #FFFFFF;
    border: 1px solid #F5C6D8;
    border-radius: 20px;
    padding: 32px 36px;
    margin-top: 8px;
    box-shadow: 0 4px 30px rgba(224,92,138,0.08);
}
.result-skin-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #C4456E;
    margin-bottom: 8px;
}
.result-skin-type {
    font-family: 'Playfair Display', serif;
    font-size: 48px;
    font-weight: 600;
    color: #2E1420;
    margin: 0 0 6px;
    line-height: 1;
}
.result-skin-sub {
    font-size: 13px;
    color: #A06080;
    margin-bottom: 24px;
    font-weight: 300;
}
.divider {
    height: 1px;
    background: linear-gradient(90deg, #F5C6D8, transparent);
    margin: 20px 0;
}
.tip-box {
    background: linear-gradient(135deg, #FEF0F5, #FDE4EE);
    border: 1px solid #F5C6D8;
    border-radius: 14px;
    padding: 20px 24px;
}
.tip-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #C4456E;
    margin-bottom: 10px;
}
.tip-text {
    font-size: 14px;
    color: #5A3048;
    line-height: 1.85;
    font-weight: 400;
}

.skin-badge {
    display: inline-block;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1px;
    padding: 4px 14px;
    border-radius: 20px;
    text-transform: uppercase;
    margin-bottom: 20px;
}
.skin-badge-acne { background:#FDE8EE; color:#B03060; border:1px solid #F5C6D8; }
.skin-badge-dry  { background:#EEF0FE; color:#3045B0; border:1px solid #C6CEF5; }
.skin-badge-oily { background:#E8FEEE; color:#207040; border:1px solid #C6F5D4; }

.product-name {
    font-size: 12px;
    font-weight: 500;
    color: #7A4B62;
    text-align: center;
    padding: 10px 12px;
    letter-spacing: 0.5px;
    text-transform: capitalize;
    background: #FEF6F9;
    border-top: 1px solid #F5E0E9;
}

[data-testid="stImage"] img {
    border-radius: 16px;
    border: 1px solid #F5C6D8;
    box-shadow: 0 4px 20px rgba(224,92,138,0.1);
}

.stAlert { border-radius: 12px !important; font-size: 13px !important; }
.stSpinner > div { border-top-color: #E05C8A !important; }

.footer {
    text-align: center;
    padding: 24px;
    font-size: 12px;
    color: #C4A0B4;
    letter-spacing: 1px;
    border-top: 1px solid #F5E0E9;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

# ── Navbar ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="navbar">
    <span class="navbar-brand">SkinSense</span>
    <span class="navbar-tagline">AI-Powered Skin Analysis</span>
</div>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">Smart Skincare Technology</div>
    <div class="hero-title">Discover Your<br><em>Perfect Skincare</em></div>
    <div class="hero-desc">
        Upload your photo and let our AI analyse your skin type in seconds.
        Get personalised care tips and product recommendations curated just for you.
    </div>
    <div class="hero-pills">
        <span class="pill">Acne Skin</span>
        <span class="pill">Dry Skin</span>
        <span class="pill">Oily Skin</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Main content ──────────────────────────────────────────────────────────────
st.markdown('<div class="content">', unsafe_allow_html=True)

col_left, col_gap, col_right = st.columns([5, 1, 6])

with col_left:
    st.markdown('<span class="sec-label">Upload Your Photo</span>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        label="",
        type=["jpg", "jpeg", "png"],
        help="Use a clear, well-lit front-facing photo for best results.",
        label_visibility="collapsed",
    )
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)
        analyze = st.button("Analyse My Skin")
    else:
        analyze = False

with col_right:
    if uploaded_file and analyze:
        with st.spinner("Analysing your skin..."):
            skin_type = predict_skin_type(image)

        subtitles = {
            "Dry":  "Your skin needs extra hydration and nourishment.",
            "Oily": "Your skin tends to produce excess sebum.",
            "Acne": "Your skin is prone to breakouts and blemishes.",
        }
        tips = {
            "Dry":  "Prioritise rich, occlusive moisturisers morning and night. Avoid hot water and "
                    "harsh cleansers. Look for hyaluronic acid, ceramides, and squalane in your products.",
            "Oily": "Use lightweight, oil-free formulations. Incorporate a gentle salicylic acid "
                    "cleanser and a non-comedogenic SPF. Blotting papers help manage shine throughout the day.",
            "Acne": "Maintain a consistent, gentle cleansing routine twice daily. Niacinamide and "
                    "benzoyl peroxide can reduce inflammation. Avoid touching or picking at blemishes.",
        }

        st.markdown(f"""
        <div class="result-card">
            <div class="result-skin-label">Detected Skin Type</div>
            <div class="result-skin-type">{skin_type}</div>
            <div class="result-skin-sub">{subtitles.get(skin_type, '')}</div>
            <span class="skin-badge skin-badge-{skin_type.lower()}">{skin_type} Skin</span>
            <div class="divider"></div>
            <div class="tip-box">
                <div class="tip-label">Personalised Care Advice</div>
                <div class="tip-text">{tips.get(skin_type, 'Apply broad-spectrum SPF 30 or higher every morning.')}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        product_images = get_product_images(skin_type)
        if product_images:
            st.markdown('<br><span class="sec-label">Recommended Products</span>', unsafe_allow_html=True)
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
            st.info("Product images not found. Ensure the 'skincare products' folder exists in the repository.")

    else:
        st.markdown("""
        <div style="height:340px; border:1.5px dashed #F0C8D8; border-radius:20px;
            display:flex; flex-direction:column; align-items:center; justify-content:center;
            color:#D4A0B8; gap:12px; margin-top:8px;
            background:linear-gradient(135deg,#FFF7FA,#FFF0F5);">
            <div style="font-size:36px; opacity:0.35;">&#10022;</div>
            <div style="font-size:12px; letter-spacing:2px; text-transform:uppercase; font-weight:500;">
                Your results will appear here
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    SkinSense &nbsp;&middot;&nbsp; AI Skin Analysis &nbsp;&middot;&nbsp; Built with Python &amp; Keras
</div>
""", unsafe_allow_html=True)
