import os
import json
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input   # ← MobileNetV2
from PIL import Image, ImageEnhance, ImageFilter
import time

st.set_page_config(page_title="LeafScan AI", page_icon="🌿", layout="wide")

# =====================================================
# LOAD CONFIG — written by train.py
# Everything comes from here: model path, input size,
# architecture, accuracy, class count, preprocessor.
# =====================================================
CONFIG_PATH = "model_config.json"

@st.cache_resource
def load_config():
    if not os.path.exists(CONFIG_PATH):
        return None
    with open(CONFIG_PATH) as f:
        return json.load(f)

cfg = load_config()

if cfg is None:
    st.error("❌ model_config.json not found. Run train.py first to generate it.")
    st.stop()

MODEL_PATH  = cfg["model_path"]
LABELS_PATH = cfg["labels_path"]
INPUT_SIZE  = cfg["input_size"]       # from train.py — not hardcoded
ARCHITECTURE= cfg["architecture"]     # "MobileNetV2"
VAL_ACC     = cfg["val_accuracy"]
TRAIN_ACC   = cfg["train_accuracy"]
NUM_CLASSES = cfg["num_classes"]
EPOCHS_DONE = cfg["epochs_trained"]

# =====================================================
# CSS
# =====================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html,body,[data-testid="stAppViewContainer"]{background-color:#0d1a0f!important;color:#e8f0e9!important;font-family:'DM Sans',sans-serif}
[data-testid="stAppViewContainer"]>.main{background-color:#0d1a0f!important}
#MainMenu,footer,header{visibility:hidden}
[data-testid="stSidebar"]{background-color:#0a1409!important;border-right:1px solid #1e3a20!important}
[data-testid="stSidebar"]*{color:#b8d4ba!important}
.block-container{padding:0!important;max-width:100%!important}

.hero{background:linear-gradient(135deg,#0a1a0c 0%,#0f2a12 40%,#081508 100%);border-bottom:1px solid #1e3a20;padding:60px 80px 50px;position:relative;overflow:hidden}
.hero::before{content:'';position:absolute;top:-60px;right:-60px;width:340px;height:340px;border-radius:50%;background:radial-gradient(circle,rgba(74,163,84,.12) 0%,transparent 70%);pointer-events:none}
.hero-badge{display:inline-flex;align-items:center;gap:6px;background:rgba(74,163,84,.15);border:1px solid rgba(74,163,84,.3);border-radius:100px;padding:5px 14px;font-size:12px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:#6dba75!important;margin-bottom:20px}
.hero-badge span{width:7px;height:7px;background:#4aa354;border-radius:50%;animation:pulse-dot 2s infinite}
@keyframes pulse-dot{0%,100%{opacity:1}50%{opacity:.3}}
.hero h1{font-family:'DM Serif Display',serif;font-size:clamp(38px,5vw,60px);font-weight:400;color:#e8f5e9!important;line-height:1.1;margin-bottom:16px}
.hero h1 em{font-style:italic;color:#6dba75!important}
.hero p{font-size:17px;color:#7a9e7d!important;max-width:560px;line-height:1.7;font-weight:300}
.hero-stats{display:flex;gap:40px;margin-top:40px}
.stat{border-left:2px solid #2a5c2e;padding-left:16px}
.stat-num{font-family:'DM Serif Display',serif;font-size:28px;color:#a8d4aa!important;line-height:1}
.stat-label{font-size:12px;color:#4d7a50!important;font-weight:500;margin-top:3px;text-transform:uppercase;letter-spacing:.06em}

.upload-section{padding:50px 80px;max-width:1200px;margin:0 auto}
.section-label{font-size:11px;font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:#4d7a50!important;margin-bottom:10px}
.section-title{font-family:'DM Serif Display',serif;font-size:30px;color:#d4edd5!important;margin-bottom:6px}
.section-sub{font-size:15px;color:#5a8560!important;font-weight:300;margin-bottom:30px}

[data-testid="stFileUploader"]{background:#0f2213!important;border:2px dashed #2a5c2e!important;border-radius:16px!important;padding:10px!important}
[data-testid="stFileUploader"]:hover{border-color:#4aa354!important}
[data-testid="stFileUploaderDropzone"]{background:transparent!important}
[data-testid="stFileUploaderDropzoneInstructions"] p,
[data-testid="stFileUploaderDropzoneInstructions"] small,
[data-testid="stFileUploaderDropzoneInstructions"] span{color:#5a8560!important}
[data-testid="stImage"] img{border-radius:12px!important}

.auto-enh{display:inline-flex;align-items:center;gap:8px;background:#061a0a;border:1px solid #1e4a22;border-radius:10px;padding:8px 14px;font-size:12px;color:#4dba60;margin-bottom:12px}
.enh-dot{width:6px;height:6px;border-radius:50%;background:#4aa354;animation:pulse-dot 1.5s infinite}

.img-metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:7px;margin-top:12px}
.img-metric{background:#0f2213;border:1px solid #1a3a1d;border-radius:10px;padding:10px 12px}
.im-label{font-size:9px;font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:#3d6b40;margin-bottom:4px}
.im-value{font-size:13px;font-weight:600;color:#a8d4aa;font-variant-numeric:tabular-nums}

.iss-warn{background:#1a0f04;border:1px solid rgba(251,146,60,.3);border-radius:10px;padding:9px 13px;font-size:12px;color:#fb923c;margin-bottom:7px;display:flex;gap:8px}
.iss-err{background:#1a0404;border:1px solid rgba(244,63,94,.3);border-radius:10px;padding:9px 13px;font-size:12px;color:#f43f5e;margin-bottom:7px;display:flex;gap:8px}

/* CONFIG INFO BANNER */
.cfg-banner{background:#061a0a;border:1px solid #1e4a22;border-radius:12px;padding:12px 16px;margin-bottom:16px;display:flex;gap:24px;flex-wrap:wrap}
.cfg-item{display:flex;flex-direction:column;gap:2px}
.cfg-lbl{font-size:9px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:#2a4d39}
.cfg-val{font-size:13px;font-weight:600;color:#6dba75;font-variant-numeric:tabular-nums}

.conf-panel{background:#0a1a0c;border:1px solid #1e3a20;border-radius:16px;padding:20px 22px;margin-bottom:16px}
.conf-top{display:flex;justify-content:space-between;align-items:flex-end;margin-bottom:12px}
.conf-number{font-family:'DM Serif Display',serif;font-size:52px;line-height:1}
.conf-pct{font-size:22px;opacity:.55}
.conf-right{text-align:right}
.conf-badge{display:inline-flex;align-items:center;gap:6px;padding:5px 14px;border-radius:99px;font-size:12px;font-weight:600;border:1px solid}
.conf-track{height:5px;background:#1a3a1d;border-radius:99px;overflow:hidden;margin-bottom:10px}
.conf-fill{height:100%;border-radius:99px}
.conf-note{font-size:12px;color:#4d7a50;line-height:1.5;margin-bottom:12px}
.conf-meta{display:flex;gap:20px;padding-top:12px;border-top:1px solid #1a3a1d}
.cm-lbl{font-size:9px;letter-spacing:.12em;text-transform:uppercase;color:#2a4d39;margin-bottom:3px}
.cm-val{font-size:13px;font-weight:600;color:#6dba75;font-variant-numeric:tabular-nums}

.evidence-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:14px}
.ev-box{background:#081508;border:1px solid #1a3a1d;border-radius:12px;padding:14px}
.ev-label{font-size:10px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:#3d6b40;margin-bottom:6px}
.ev-bar-track{height:4px;background:#1a3a1d;border-radius:99px;overflow:hidden;margin-bottom:5px}
.ev-bar-fill{height:100%;border-radius:99px}
.ev-value{font-size:14px;font-weight:600;color:#a8d4aa}
.ev-note{font-size:11px;color:#4d7a50;margin-top:3px}

[data-testid="stMetric"]{background:#0f2213!important;border:1px solid #1e3a20!important;border-radius:12px!important;padding:12px 14px!important}
[data-testid="stMetricValue"]{color:#a8d4aa!important;font-family:'DM Serif Display',serif!important}
[data-testid="stMetricLabel"]{color:#4d7a50!important;font-size:11px!important;text-transform:uppercase!important;letter-spacing:.06em!important}
[data-testid="stProgressBar"]>div{background:#1a3a1d!important;border-radius:100px!important}
[data-testid="stProgressBar"]>div>div{background:linear-gradient(90deg,#2a8c34,#6dba75)!important;border-radius:100px!important}
[data-testid="stAlert"]{border-radius:12px!important}
[data-testid="stAlert"][data-type="success"]{background:rgba(74,163,84,.12)!important;border:1px solid rgba(74,163,84,.3)!important;color:#6dba75!important}
[data-testid="stAlert"][data-type="error"]{background:rgba(220,80,60,.10)!important;border:1px solid rgba(220,80,60,.25)!important;color:#e87060!important}
[data-testid="stAlert"][data-type="info"]{background:#081508!important;border:1px solid #1e3a20!important;color:#a8c8aa!important}
hr{border-color:#1e3a20!important}
::-webkit-scrollbar{width:6px}
::-webkit-scrollbar-track{background:#0a1409}
::-webkit-scrollbar-thumb{background:#2a5c2e;border-radius:10px}
.stButton button{background:#1e3a20!important;border:1px solid #2a5c2e!important;color:#a8d4aa!important;border-radius:10px!important;font-family:'DM Sans',sans-serif!important;font-weight:500!important}
.stButton button:hover{background:#2a5c2e!important;color:#d4edd5!important}
[data-testid="stSidebar"] h2{font-family:'DM Serif Display',serif!important;font-size:22px!important;color:#a8d4aa!important}
[data-testid="stSidebar"] [data-testid="stMetric"]{background:#0f2213;border:1px solid #1e3a20;border-radius:10px;padding:10px 12px}
[data-testid="stSidebar"] [data-testid="stMetricValue"]{font-family:'DM Serif Display',serif!important;color:#6dba75!important;font-size:20px!important}
[data-testid="stSidebar"] [data-testid="stMetricLabel"]{color:#4d7a50!important;font-size:11px!important}
[data-testid="stSidebar"] p,[data-testid="stSidebar"] li{color:#7a9e7d!important;font-size:13px!important}
[data-testid="stSidebar"] strong{color:#a8d4aa!important}
[data-testid="stSidebar"] hr{border-color:#1e3a20!important}
</style>
""", unsafe_allow_html=True)

# =====================================================
# AUTO-ENHANCEMENT (always applied, silent)
# =====================================================
def auto_enhance(pil_img):
    arr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(arr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(l)
    arr = cv2.cvtColor(cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2RGB)
    enh = Image.fromarray(arr)
    blur = enh.filter(ImageFilter.GaussianBlur(radius=1))
    e  = np.array(enh,  dtype=np.float32)
    bl = np.array(blur, dtype=np.float32)
    enh = Image.fromarray(np.clip(e + 0.55*(e-bl), 0, 255).astype(np.uint8))
    enh = ImageEnhance.Color(enh).enhance(1.18)
    d = cv2.fastNlMeansDenoisingColored(
        cv2.cvtColor(np.array(enh), cv2.COLOR_RGB2BGR),
        None, h=4, hColor=4, templateWindowSize=7, searchWindowSize=21)
    return Image.fromarray(cv2.cvtColor(d, cv2.COLOR_BGR2RGB))

# =====================================================
# VISUAL EVIDENCE ANALYSER
# =====================================================
def analyse_visual_evidence(pil_img):
    arr  = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    hsv  = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    h, w = gray.shape

    green_mask = cv2.inRange(hsv, np.array([25,30,30]), np.array([95,255,255]))
    leaf_px    = max(green_mask.sum()//255, 1)

    # Spot / lesion detection
    blur2 = cv2.GaussianBlur(gray,(21,21),0)
    diff  = cv2.absdiff(gray, blur2)
    _, spot_mask = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
    spot_pct = cv2.bitwise_and(spot_mask, green_mask).sum()//255 / leaf_px * 100

    # Discolouration (yellow/brown)
    ym = cv2.inRange(hsv, np.array([10,40,40]),  np.array([35,255,255]))
    bm = cv2.inRange(hsv, np.array([0, 30,30]),  np.array([15,200,200]))
    discolour_pct = cv2.bitwise_and(cv2.bitwise_or(ym,bm), green_mask).sum()//255 / leaf_px * 100

    # Healthy green
    healthy_green_pct = min(cv2.inRange(hsv, np.array([35,60,60]), np.array([90,255,255])).sum()//255 / leaf_px * 100, 100)

    # Texture roughness
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    leaf_bool = green_mask > 0
    texture_pct = min(float(np.std(lap[leaf_bool]) if leaf_bool.sum()>100 else np.std(lap)) / 25 * 100, 100)

    # Quality
    sharpness  = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = gray.mean()
    qp = 0
    if sharpness  <  80: qp += (80-sharpness)/80*25
    if brightness <  50: qp += (50-brightness)/50*20
    if brightness > 220: qp += (brightness-220)/35*10
    qp = min(qp, 40)

    return {
        "spot_pct":      round(spot_pct, 1),
        "discolour_pct": round(discolour_pct, 1),
        "healthy_green": round(healthy_green_pct, 1),
        "texture_pct":   round(texture_pct, 1),
        "sharpness":     round(sharpness, 1),
        "brightness":    round(brightness, 1),
        "quality_penalty": round(qp, 1),
    }

# =====================================================
# CALIBRATED CONFIDENCE
# =====================================================
def calibrated_confidence(raw_pct, ev, pred_class):
    is_healthy = "healthy" in pred_class.lower()
    boost = 0
    if is_healthy:
        if ev["spot_pct"]      <  3: boost += 8
        elif ev["spot_pct"]    <  8: boost += 2
        else:                        boost -= 15
        if ev["discolour_pct"] <  2: boost += 6
        elif ev["discolour_pct"]<  6:boost += 1
        else:                        boost -= 12
        if ev["healthy_green"] > 50: boost += 6
        elif ev["healthy_green"]>25: boost += 2
        else:                        boost -= 8
    else:
        if ev["spot_pct"]      > 10: boost += 10
        elif ev["spot_pct"]    >  4: boost += 4
        else:                        boost -= 12
        if ev["discolour_pct"] >  8: boost += 8
        elif ev["discolour_pct"]>  3:boost += 3
        else:                        boost -= 8
        if ev["healthy_green"] < 30: boost += 5
        elif ev["healthy_green"]> 60:boost -= 10

    adjusted = max(22.0, min(raw_pct + boost - ev["quality_penalty"], 97.5))
    return round(adjusted, 1), boost

# =====================================================
# PREPROCESS — uses INPUT_SIZE from config
# =====================================================
def preprocess_img(pil_img):
    w, h  = pil_img.size
    side  = max(w, h)
    sq    = Image.new("RGB", (side,side), (0,0,0))
    sq.paste(pil_img, ((side-w)//2, (side-h)//2))
    r  = sq.resize((INPUT_SIZE, INPUT_SIZE), Image.LANCZOS)
    a  = image.img_to_array(r)
    a  = preprocess_input(a)             # MobileNetV2 preprocessor — matches train.py
    return np.expand_dims(a.astype(np.float32), 0)

# =====================================================
# QUALITY CHECK
# =====================================================
def check_quality(pil_img):
    arr  = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    hsv  = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    h, w = gray.shape
    blur   = cv2.Laplacian(gray, cv2.CV_64F).var()
    bright = gray.mean()
    cont   = gray.std()
    gm     = cv2.inRange(hsv, np.array([35,30,30]), np.array([90,255,255]))
    green  = gm.sum()/255/(h*w)*100
    issues = []
    if blur   <  30: issues.append(("error",f"Very blurry (sharpness {blur:.0f}). Retake with a steady hand."))
    elif blur <  80: issues.append(("warn", f"Slight blur (sharpness {blur:.0f})."))
    if bright < 35:  issues.append(("error",f"Too dark ({bright:.0f}/255). Use better lighting."))
    elif bright < 60:issues.append(("warn", f"Low brightness ({bright:.0f}/255)."))
    elif bright>225: issues.append(("warn", f"Overexposed ({bright:.0f}/255)."))
    if w<100 or h<100:issues.append(("error",f"Resolution too low ({w}×{h}px)."))
    if green  <  5:  issues.append(("warn", "Very little green detected."))
    metrics = {
        "res":   f"{w}×{h}",
        "sharp": f"{blur:.0f}",
        "bright":f"{bright:.0f}/255",
        "cont":  f"{cont:.0f}",
        "green": f"{green:.1f}%",
        "edges": f"{cv2.Canny(gray,50,150).sum()/255/(h*w)*100:.2f}%",
        "rgb":   f"{arr[:,:,0].mean():.0f}/{arr[:,:,1].mean():.0f}/{arr[:,:,2].mean():.0f}",
        "aspect":f"{w/h:.2f}",
    }
    return issues, metrics

# =====================================================
# LOAD MODEL + LABELS
# =====================================================
@st.cache_resource
def load_model_and_labels():
    if not os.path.exists(MODEL_PATH):
        return None, None
    m  = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH) as f:
        cls = [l.strip() for l in f.readlines()]
    return m, cls

model, class_names = load_model_and_labels()
if model is None:
    st.error(f"❌ Model not found at '{MODEL_PATH}'. Run train.py first.")
    st.stop()

# =====================================================
# TREATMENTS
# =====================================================
treatments = {
    "Apple___Apple_scab":       {"text":"Apply fungicide during early spring. Remove infected leaves. Ensure good air circulation through pruning.","severity":"Moderate","season":"Spring / Early Summer"},
    "Apple___Black_rot":        {"text":"Prune and destroy infected wood and mummified fruit. Apply copper-based fungicides at bud break.","severity":"High","season":"Year-round"},
    "Apple___Cedar_apple_rust": {"text":"Remove nearby cedar/juniper trees if possible. Apply preventive fungicides from pink bud to petal fall.","severity":"Moderate","season":"Spring"},
    "Apple___healthy":          {"text":"Plant is in excellent health! Maintain regular watering, adequate sunlight, and seasonal pruning.","severity":"None","season":"All seasons"},
    "Tomato___Late_blight":     {"text":"Apply copper-based fungicides immediately. Remove and bag infected parts. Improve drainage.","severity":"Severe","season":"Cool, Wet Periods"},
    "Tomato___healthy":         {"text":"Plant is thriving! Continue balanced fertilization and consistent watering at the base.","severity":"None","season":"All seasons"},
}
DEFAULT_TREATMENT = {"text":"Consult a local agricultural extension office. Isolate affected plants and improve air circulation.","severity":"Unknown","season":"Varies"}

# =====================================================
# SIDEBAR — values from config, not hardcoded
# =====================================================
with st.sidebar:
    st.markdown("## 🌿 LeafScan AI")
    st.caption("Deep Learning Plant Diagnostics")
    st.divider()

    st.markdown("<p style='font-size:10px;font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:#4d7a50;margin-bottom:8px;'>MODEL — FROM TRAINING</p>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Val Accuracy",  f"{VAL_ACC}%")
        st.metric("Architecture",  ARCHITECTURE)
    with c2:
        st.metric("Classes",       str(NUM_CLASSES))
        st.metric("Epochs Trained",str(EPOCHS_DONE))

    st.divider()
    st.markdown("<p style='font-size:10px;font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:#4d7a50;margin-bottom:8px;'>AUTO-ENHANCEMENT</p>", unsafe_allow_html=True)
    st.markdown("🔬 **CLAHE** — local contrast boost")
    st.markdown("✨ **Unsharp Mask** — sharpens texture")
    st.markdown("🎨 **Saturation** — highlights lesions")
    st.markdown("🧹 **NL-Denoise** — removes sensor noise")

    st.divider()
    st.markdown("<p style='font-size:10px;font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:#4d7a50;margin-bottom:8px;'>CONFIDENCE BASED ON</p>", unsafe_allow_html=True)
    st.markdown("🔍 Spot & lesion coverage")
    st.markdown("🌿 Colour health (green vs yellow/brown)")
    st.markdown("📐 Texture roughness pattern")
    st.markdown("📷 Image quality (sharpness, brightness)")
    st.markdown("🧠 Model softmax + top-2 gap")

    st.divider()
    st.markdown("<p style='font-size:10px;font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:#4d7a50;margin-bottom:8px;'>TIPS</p>", unsafe_allow_html=True)
    st.markdown("📷 Use natural light")
    st.markdown("🍃 Focus on a single leaf")
    st.markdown("🔍 Fill the frame with the leaf")
    st.markdown("📐 Keep image under 5MB")

# =====================================================
# HERO — uses config values
# =====================================================
st.markdown(f"""
<div class="hero">
    <div class="hero-badge"><span></span>AI-Powered · {ARCHITECTURE}</div>
    <h1>Detect plant diseases<br>with <em>precision.</em></h1>
    <p>Upload a leaf image. Auto-enhanced, visually analysed for spots and discolouration, then given a confidence level based on what the model actually sees.</p>
    <div class="hero-stats">
        <div class="stat"><div class="stat-num">{NUM_CLASSES}</div><div class="stat-label">Classes</div></div>
        <div class="stat"><div class="stat-num">{VAL_ACC}%</div><div class="stat-label">Val Accuracy</div></div>
        <div class="stat"><div class="stat-num">{EPOCHS_DONE}</div><div class="stat-label">Epochs Trained</div></div>
        <div class="stat"><div class="stat-num">4-step</div><div class="stat-label">Enhancement</div></div>
    </div>
</div>
""", unsafe_allow_html=True)

# =====================================================
# MAIN
# =====================================================
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
col_upload, col_result = st.columns([1,1], gap="large")

# ── LEFT ─────────────────────────────────────────────
with col_upload:
    st.markdown("""
    <div class="section-label">Step 1</div>
    <div class="section-title">Upload Leaf Image</div>
    <div class="section-sub">JPG, JPEG or PNG · Max 5MB · Auto-enhanced before analysis</div>
    """, unsafe_allow_html=True)

    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    uploaded_file = st.file_uploader(
        "Drop your leaf photo here or browse",
        type=["jpg","jpeg","png"],
        label_visibility="collapsed",
        key=f"leaf_{st.session_state.uploader_key}"
    )

    if uploaded_file:
        raw_img = Image.open(uploaded_file).convert("RGB")
        issues, metrics = check_quality(raw_img)

        for lvl, msg in issues:
            cls = "iss-err" if lvl=="error" else "iss-warn"
            ico = "✕" if lvl=="error" else "⚠"
            st.markdown(f'<div class="{cls}"><span>{ico}</span><span>{msg}</span></div>', unsafe_allow_html=True)

        enh_img = auto_enhance(raw_img)

        st.markdown("""
        <div class="auto-enh">
            <div class="enh-dot"></div>
            Auto-enhanced · CLAHE · Unsharp Mask · Saturation · Denoise
        </div>
        """, unsafe_allow_html=True)
        st.image(enh_img, use_container_width=True)

        m = metrics
        st.markdown(f"""
        <div class="img-metrics">
            <div class="img-metric"><div class="im-label">Resolution</div><div class="im-value">{m['res']}</div></div>
            <div class="img-metric"><div class="im-label">Sharpness</div><div class="im-value">{m['sharp']}</div></div>
            <div class="img-metric"><div class="im-label">Brightness</div><div class="im-value">{m['bright']}</div></div>
            <div class="img-metric"><div class="im-label">Contrast</div><div class="im-value">{m['cont']}</div></div>
            <div class="img-metric"><div class="im-label">Green Cover</div><div class="im-value">{m['green']}</div></div>
            <div class="img-metric"><div class="im-label">Edge Density</div><div class="im-value">{m['edges']}</div></div>
            <div class="img-metric"><div class="im-label">R/G/B Avg</div><div class="im-value" style="font-size:11px">{m['rgb']}</div></div>
            <div class="img-metric"><div class="im-label">Aspect Ratio</div><div class="im-value">{m['aspect']}</div></div>
        </div>
        """, unsafe_allow_html=True)

        fn_col, btn_col = st.columns([3,1])
        with fn_col:
            st.markdown(f"<div style='margin-top:10px;padding:10px 14px;background:#0f2213;border:1px solid #1e3a20;border-radius:10px;font-size:13px;color:#a8c8aa;'>📄 {uploaded_file.name}</div>", unsafe_allow_html=True)
        with btn_col:
            st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.uploader_key += 1
                st.rerun()

# ── RIGHT ─────────────────────────────────────────────
with col_result:
    st.markdown("<p style='font-size:11px;font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:#4d7a50;margin-bottom:2px;'>STEP 2</p>", unsafe_allow_html=True)
    st.markdown("<p style='font-family:DM Serif Display,serif;font-size:30px;color:#d4edd5;margin-bottom:2px;'>Diagnosis Result</p>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:14px;color:#5a8560;font-weight:300;margin-bottom:18px;'>Results appear automatically after upload</p>", unsafe_allow_html=True)

    if uploaded_file is not None:
        hard_errors = [msg for lv,msg in issues if lv=="error"]
        if hard_errors:
            st.error("⚠️ Image quality too low to analyse reliably. Please retake the photo.")
        else:
            with st.spinner("Analysing leaf tissue, spots, and colour patterns…"):
                time.sleep(0.3)
                arr        = preprocess_img(enh_img)
                prediction = model.predict(arr)
                pred_idx   = int(np.argmax(prediction))
                pred_class = class_names[pred_idx]
                raw_pct    = float(np.max(prediction)*100)
                ev         = analyse_visual_evidence(enh_img)
                cal_conf, ev_boost = calibrated_confidence(raw_pct, ev, pred_class)
                sorted_sc  = np.sort(prediction[0])[::-1]
                top2_gap   = float((sorted_sc[0]-sorted_sc[1])*100)

            clean    = pred_class.replace("___"," · ").replace("_"," ")
            parts    = clean.split(" · ")
            plant    = parts[0] if len(parts)>0 else ""
            disease  = parts[1] if len(parts)>1 else clean
            is_h     = "healthy" in pred_class.lower()
            t_info   = treatments.get(pred_class, DEFAULT_TREATMENT)

            # Config banner — shows model details used for this prediction
            st.markdown(f"""
            <div class="cfg-banner">
                <div class="cfg-item"><div class="cfg-lbl">Model</div><div class="cfg-val">{ARCHITECTURE}</div></div>
                <div class="cfg-item"><div class="cfg-lbl">Input Size</div><div class="cfg-val">{INPUT_SIZE}×{INPUT_SIZE}</div></div>
                <div class="cfg-item"><div class="cfg-lbl">Val Accuracy</div><div class="cfg-val">{VAL_ACC}%</div></div>
                <div class="cfg-item"><div class="cfg-lbl">Preprocessor</div><div class="cfg-val">mobilenet_v2</div></div>
                <div class="cfg-item"><div class="cfg-lbl">Classes</div><div class="cfg-val">{NUM_CLASSES}</div></div>
            </div>
            """, unsafe_allow_html=True)

            if is_h:
                st.success("✅  Plant Status: HEALTHY")
            else:
                st.error("⚠️  Plant Status: DISEASE DETECTED")

            st.markdown(f"<p style='font-size:11px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:#4d7a50;margin-top:8px;margin-bottom:2px;'>{plant}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-family:DM Serif Display,serif;font-size:32px;color:#e8f5e9;line-height:1.15;margin-bottom:16px;'>{disease}</p>", unsafe_allow_html=True)

            # Confidence colours
            if cal_conf >= 85:
                c_col,c_lbl = "#4ade80","✦ Very High"
                c_note = "Strong visual evidence supports this diagnosis."
            elif cal_conf >= 72:
                c_col,c_lbl = "#86efac","◈ High"
                c_note = "Good evidence visible. Result is reliable for most practical purposes."
            elif cal_conf >= 58:
                c_col,c_lbl = "#facc15","◇ Moderate"
                c_note = "Some evidence detected but patterns are not fully clear."
            elif cal_conf >= 42:
                c_col,c_lbl = "#fb923c","◇ Low"
                c_note = "Weak visual signals. Better lighting or focus may help."
            else:
                c_col,c_lbl = "#f43f5e","✕ Very Low"
                c_note = "Insufficient visual evidence. Please retake with better lighting."

            bar_col = f"linear-gradient(90deg,{c_col}99,{c_col})"

            st.markdown(f"""
            <div class="conf-panel">
                <div class="conf-top">
                    <div>
                        <div style="font-size:10px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:#3d6b40;margin-bottom:4px;">Confidence Level</div>
                        <div class="conf-number" style="color:{c_col};">{cal_conf:.1f}<span class="conf-pct">%</span></div>
                    </div>
                    <div class="conf-right">
                        <div style="font-size:10px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;color:#3d6b40;margin-bottom:6px;">Reliability</div>
                        <div class="conf-badge" style="color:{c_col};border-color:{c_col}44;background:{c_col}11;">{c_lbl}</div>
                    </div>
                </div>
                <div class="conf-track"><div class="conf-fill" style="width:{cal_conf:.1f}%;background:{bar_col};"></div></div>
                <div class="conf-note">{c_note}</div>
                <div class="conf-meta">
                    <div><div class="cm-lbl">Model Score</div><div class="cm-val">{raw_pct:.1f}%</div></div>
                    <div><div class="cm-lbl">Visual Adjust</div><div class="cm-val">{ev_boost:+.0f}%</div></div>
                    <div><div class="cm-lbl">Quality Penalty</div><div class="cm-val">-{ev['quality_penalty']:.0f}%</div></div>
                    <div><div class="cm-lbl">Top-2 Gap</div><div class="cm-val">{top2_gap:.1f}%</div></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Visual evidence
            s_col = "#f43f5e" if ev["spot_pct"]>8      else "#facc15" if ev["spot_pct"]>3      else "#4ade80"
            d_col = "#f43f5e" if ev["discolour_pct"]>8  else "#facc15" if ev["discolour_pct"]>3  else "#4ade80"
            g_col = "#4ade80" if ev["healthy_green"]>40  else "#facc15" if ev["healthy_green"]>20 else "#f43f5e"
            s_note= "Lesions present" if ev["spot_pct"]>8    else "Minor spots"     if ev["spot_pct"]>3    else "Clean surface"
            d_note= "Discolouration found" if ev["discolour_pct"]>8 else "Slight change" if ev["discolour_pct"]>3 else "Normal colour"
            g_note= "Rich healthy green" if ev["healthy_green"]>40  else "Reduced green"  if ev["healthy_green"]>20 else "Low green signal"

            st.markdown(f"""
            <div style="font-size:11px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:#4d7a50;margin-bottom:10px;">What the model sees</div>
            <div class="evidence-grid">
                <div class="ev-box">
                    <div class="ev-label">Spot / Lesion</div>
                    <div class="ev-bar-track"><div class="ev-bar-fill" style="width:{min(ev['spot_pct']*5,100):.0f}%;background:{s_col};"></div></div>
                    <div class="ev-value" style="color:{s_col};">{ev['spot_pct']:.1f}%</div>
                    <div class="ev-note">{s_note}</div>
                </div>
                <div class="ev-box">
                    <div class="ev-label">Discolouration</div>
                    <div class="ev-bar-track"><div class="ev-bar-fill" style="width:{min(ev['discolour_pct']*5,100):.0f}%;background:{d_col};"></div></div>
                    <div class="ev-value" style="color:{d_col};">{ev['discolour_pct']:.1f}%</div>
                    <div class="ev-note">{d_note}</div>
                </div>
                <div class="ev-box">
                    <div class="ev-label">Healthy Green</div>
                    <div class="ev-bar-track"><div class="ev-bar-fill" style="width:{min(ev['healthy_green'],100):.0f}%;background:{g_col};"></div></div>
                    <div class="ev-value" style="color:{g_col};">{ev['healthy_green']:.1f}%</div>
                    <div class="ev-note">{g_note}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            m1, m2 = st.columns(2)
            with m1: st.metric("Severity",    t_info["severity"])
            with m2: st.metric("Peak Season", t_info["season"])

            st.divider()

            # Top-3
            st.markdown("<p style='font-size:11px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:#4d7a50;margin-bottom:10px;'>Top 3 Predictions</p>", unsafe_allow_html=True)
            top3   = prediction[0].argsort()[-3:][::-1]
            top_sc = [float(prediction[0][i])*100 for i in top3]
            max_s  = max(top_sc) if top_sc else 1
            for i, score in zip(top3, top_sc):
                lbl = class_names[i].replace("___"," · ").replace("_"," ")
                wb  = score/max_s*100
                col = "#4ade80" if score>=80 else "#facc15" if score>=55 else "#f43f5e"
                st.markdown(f"""
                <div style="margin-bottom:10px">
                    <div style="display:flex;justify-content:space-between;margin-bottom:4px">
                        <span style="font-size:12px;color:#a8d4aa">{lbl}</span>
                        <span style="font-size:12px;color:#4d7a50;font-variant-numeric:tabular-nums">{score:.1f}%</span>
                    </div>
                    <div style="height:3px;background:#1a3a1d;border-radius:99px;overflow:hidden">
                        <div style="width:{wb:.1f}%;height:100%;background:{col};border-radius:99px"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.divider()
            st.markdown("<p style='font-size:11px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:#4d7a50;margin-bottom:6px;'>💊 Recommended Treatment</p>", unsafe_allow_html=True)
            st.info(t_info["text"])

    else:
        st.markdown("""
        <div style="background:#0f2213;border:2px dashed #1e3a20;border-radius:16px;padding:50px 20px;text-align:center;margin-top:8px;">
            <div style="font-size:44px;opacity:.45;margin-bottom:12px;">🌿</div>
            <p style="font-family:DM Serif Display,serif;font-size:20px;color:#4d7a50;margin-bottom:6px;">Awaiting your leaf</p>
            <p style="font-size:13px;color:#2a5c2e;font-weight:300;">Upload an image on the left to receive an instant diagnosis</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div style="border-top:1px solid #1e3a20;padding:24px 80px;display:flex;justify-content:space-between;align-items:center;margin-top:20px;">
    <div style="font-family:'DM Serif Display',serif;font-size:16px;color:#4d7a50;">🌿 LeafScan AI</div>
    <div style="font-size:12px;color:#2a5c2e;">TensorFlow · MobileNetV2 · OpenCV CLAHE · PlantVillage</div>
    <div style="font-size:12px;color:#2a5c2e;">For educational use only</div>
</div>
""", unsafe_allow_html=True)