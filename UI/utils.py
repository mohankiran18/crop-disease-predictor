import streamlit as st
import datetime
import io
import re
import os
import numpy as np
import google.generativeai as genai
import firebase_admin
from firebase_admin import credentials, firestore
from PIL import Image

# Machine Learning Imports
import tensorflow as tf

# PDF Generation
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ============ CONFIGURATION ============

GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# IMPORTANT: Change this to match your model's training input!
# Common sizes: (128, 128), (224, 224), (256, 256)
IMAGE_SIZE = (128, 128) 

# ============ MODEL LOADING ============

@st.cache_resource
def load_local_model():
    """Loads the model from the Backend_Model folder."""
    try:
        # We need to go 'up' one level from UI folder to find Backend_Model
        base_path = os.path.dirname(os.path.dirname(__file__))
        
        # Try finding the .keras file first
        model_path = os.path.join(base_path, "Backend_Model", "trained_model.keras")
        
        # Fallback to .h5 if .keras doesn't exist
        if not os.path.exists(model_path):
             model_path = os.path.join(base_path, "plant_disease", "trained_model.h5")

        if os.path.exists(model_path):
            return tf.keras.models.load_model(model_path)
        else:
            st.error(f"Model file not found at: {model_path}")
            return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load model once at startup
model = load_local_model()

# ============ PREDICTION FUNCTION ============

def predict_via_api(image_file):
    """
    Runs prediction directly in the app (No external backend needed).
    """
    if model is None:
        return "Model Failed to Load", 0.0

    try:
        # 1. Preprocess the image
        image = Image.open(image_file)
        image = image.resize(IMAGE_SIZE) # Resize to match training
        img_array = np.array(image)
        
        # Normalize if your training did (usually /255.0)
        # Assuming standard Rescaling(1./255)
        img_array = img_array.astype("float32") / 255.0
        
        # Add batch dimension (1, 128, 128, 3)
        img_array = np.expand_dims(img_array, axis=0)

        # 2. Predict
        predictions = model.predict(img_array)
        confidence = float(np.max(predictions))
        
        # 3. Get Class Name (You need your class names list here!)
        # Replace this list with your ACTUAL class names from training
        class_names = [
            "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
            "Blueberry___healthy", "Cherry___Powdery_mildew", "Cherry___healthy", 
            "Corn___Cercospora_leaf_spot", "Corn___Common_rust", "Corn___Northern_Leaf_Blight", "Corn___healthy",
            "Grape___Black_rot", "Grape___Esca", "Grape___Leaf_blight", "Grape___healthy",
            "Orange___Haunglongbing", "Peach___Bacterial_spot", "Peach___healthy",
            "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", 
            "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
            "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
            "Strawberry___Leaf_scorch", "Strawberry___healthy", "Tomato___Bacterial_spot",
            "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold",
            "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites", "Tomato___Target_Spot",
            "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
        ]
        
        predicted_class = class_names[np.argmax(predictions)]
        
        return predicted_class, confidence

    except Exception as e:
        return f"Prediction Error: {e}", 0.0

# ============ DATABASE ============

@st.cache_resource
def get_db():
    try:
        if not firebase_admin._apps:
            if "firebase" in st.secrets:
                key_dict = dict(st.secrets["firebase"])
                cred = credentials.Certificate(key_dict)
                firebase_admin.initialize_app(cred)
            elif os.path.exists("firebase-key.json"):
                cred = credentials.Certificate("firebase-key.json")
                firebase_admin.initialize_app(cred)
        return firestore.client()
    except Exception as e:
        return None

db = get_db()

# ============ STYLING ============

def get_custom_css():
    return """
    <style>
        /* Keeping your existing CSS */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        .stApp { background: url('https://images.unsplash.com/photo-1625246333195-78d9c38ad449?q=80&w=3870&auto=format&fit=crop'); background-size: cover; background-attachment: fixed; }
        section[data-testid="stSidebar"] { background-color: rgba(255, 255, 255, 0.95) !important; border-right: 1px solid rgba(0,0,0,0.1); }
        .header-pill { background-color: rgba(255, 255, 255, 0.9); padding: 10px 20px; border-radius: 10px; display: inline-block; color: #064E3B; font-weight: 700; margin-bottom: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
        div[data-testid="stFileUploader"], div[data-testid="stDataFrame"], div[data-testid="stChatInput"], div[data-testid="stPlotlyChart"], div[data-testid="stSpinner"], div[data-testid="stExpander"] { background-color: rgba(255, 255, 255, 0.95); border-radius: 12px; padding: 20px; box-shadow: 0 4px 10px rgba(0,0,0,0.05); margin-bottom: 20px; border: 1px solid #E5E7EB; }
        .stCard { background-color: rgba(255, 255, 255, 0.95) !important; padding: 25px; border-radius: 12px; box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05) !important; border: 1px solid #E5E7EB; }
        .hero-container { background-color: rgba(255, 255, 255, 0.95); border-radius: 15px; padding: 30px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); margin-bottom: 30px; display: flex; justify-content: space-between; align-items: center; }
        .hero-text h1 { color: #064E3B; font-size: 2.5rem; margin:0;} .hero-text p { color: #374151; font-size: 1.1rem; margin-top:5px;} .hero-date { color: #059669; font-weight: 600; font-size: 1.2rem; text-align: right;}
        .stButton>button { background: #059669; color: white; border-radius: 8px; border: none; padding: 0.7rem 1.2rem; font-weight: 600; width: 100%; } .stButton>button:hover { background: #047857; }
        div[data-testid="stChatMessage"] { background-color: rgba(255, 255, 255, 0.95); border-radius: 12px; padding: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); margin-bottom: 10px; border: 1px solid #E5E7EB; } div[data-testid="stChatMessage"] p { color: #374151; }
        .stTabs [data-baseweb="tab-list"] { gap: 8px; background-color: rgba(255, 255, 255, 0.95); padding: 10px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border: 1px solid #E5E7EB; margin-bottom: 20px; } .stTabs [data-baseweb="tab"] { height: 45px; white-space: pre-wrap; background-color: transparent; border-radius: 8px; color: #374151; font-weight: 600; border: none; } .stTabs [aria-selected="true"] { background-color: #059669 !important; color: white !important; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        h1, h2, h3 { color: #064E3B !important; font-family: 'Inter', sans-serif;}
    </style>
    """

# ============ LOGIC FUNCTIONS ============

def clean_markdown_for_pdf(text):
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'##\s*(.*)', r'<br/><font size=14 color="#064E3B"><b>\1</b></font><br/>', text)
    text = text.replace('\n', '<br/>')
    return text

def create_pdf_report(disease: str, details_text: str) -> io.BytesIO:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=26, textColor='#064E3B', spaceAfter=10, alignment=TA_CENTER)
    body_style = ParagraphStyle('Body', parent=styles['Normal'], fontSize=11, leading=16, spaceAfter=12)
    story = [
        Paragraph("Crop Disease Report", title_style),
        Paragraph(f"Diagnosis: {disease} | Date: {datetime.datetime.now().strftime('%Y-%m-%d')}", body_style),
        Spacer(1, 20),
        Paragraph(clean_markdown_for_pdf(details_text), body_style)
    ]
    doc.build(story)
    buffer.seek(0)
    return buffer

def get_disease_details_gemini(disease_name, lang_code="en"):
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        prompt = (f"Explain {disease_name} in language: {lang_code}.\n"
                  "Return sections with headers: ## Causes, ## Symptoms, ## Precautions, ## Treatments.\n"
                  "Use bullet points for each section.")
        return model.generate_content(prompt).text
    except Exception:
        return "AI Service Unavailable."

def parse_sections(text):
    sections = {"Causes": "", "Symptoms": "", "Precautions": "", "Treatments": ""}
    curr = None
    for line in text.split('\n'):
        if "Causes" in line: curr = "Causes"; continue
        if "Symptoms" in line: curr = "Symptoms"; continue
        if "Precautions" in line: curr = "Precautions"; continue
        if "Treatments" in line: curr = "Treatments"; continue
        if curr: sections[curr] += line + "\n"
    return sections

def check_backend_status():
    # Now that we run locally, the 'backend' is the model itself
    if model is not None:
        return True
    return False