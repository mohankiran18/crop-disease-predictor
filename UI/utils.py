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

# PDF Generation Imports (For creating the download report)
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================

# Get the Gemini API key from Streamlit Secrets (Cloud)
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ⚠️ CRITICAL SETTING: 
# This size MUST match exactly what you used when training your model.
# If you trained with (224, 224), change this number!
IMAGE_SIZE = (128, 128) 

# ==========================================
# 2. LOAD THE AI MODEL
# ==========================================

@st.cache_resource # This tells Streamlit: "Load this once and keep it in memory"
def load_local_model():
    try:
        # Find the folder where the app is running
        base_path = os.path.dirname(os.path.dirname(__file__))
        
        # Look for the model in the 'Backend_Model' folder first
        model_path = os.path.join(base_path, "Backend_Model", "trained_model.keras")
        
        # If not found, look in the 'plant_disease' folder (Backup)
        if not os.path.exists(model_path):
             model_path = os.path.join(base_path, "plant_disease", "trained_model.h5")

        # Load the model using TensorFlow
        if os.path.exists(model_path):
            return tf.keras.models.load_model(model_path)
        else:
            st.error(f"Model file not found at: {model_path}")
            return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the model immediately when the app starts
model = load_local_model()

# ==========================================
# 3. PREDICTION LOGIC (The "Brain")
# ==========================================

def predict_via_api(image_file):
    if model is None:
        return "Model Failed to Load", 0.0

    try:
        # Step A: Open the image file
        image = Image.open(image_file)
        
        # Step B: Fix for PNG images (Transparency issue)
        # If the image has a transparent background (Alpha channel), convert to standard RGB.
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        # Step C: Resize the image to match the AI model's expected size
        image = image.resize(IMAGE_SIZE)
        
        # Step D: Convert image to numbers (numpy array)
        img_array = np.array(image)
        
        # Step E: Normalize the numbers (Make them floats)
        # NOTE: If your model expects numbers 0-1, uncomment the "/ 255.0" part.
        # If your model expects 0-255, keep it as is.
        img_array = img_array.astype("float32") # / 255.0
        
        # Step F: Add a "Batch" dimension
        # The model expects a list of images, even if it's just one.
        # Shape becomes: (1, 128, 128, 3)
        img_array = np.expand_dims(img_array, axis=0)

        # Step G: Make the prediction
        predictions = model.predict(img_array)
        confidence = float(np.max(predictions))
        
        # Step H: Match the result to a Disease Name
        # IMPORTANT: This list must be in the exact alphabetical order of your training folders.
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
        
        # Get the name with the highest score
        predicted_class = class_names[np.argmax(predictions)]
        
        return predicted_class, confidence

    except Exception as e:
        return f"Prediction Error: {e}", 0.0

# ==========================================
# 4. DATABASE CONNECTION
# ==========================================

@st.cache_resource
def get_db():
    try:
        # Check if Firebase is already running to avoid errors
        if not firebase_admin._apps:
            # Plan A: Try loading keys from Streamlit Cloud Secrets
            if "firebase" in st.secrets:
                key_dict = dict(st.secrets["firebase"])
                cred = credentials.Certificate(key_dict)
                firebase_admin.initialize_app(cred)
            # Plan B: Try loading from a local file (for laptop testing)
            elif os.path.exists("firebase-key.json"):
                cred = credentials.Certificate("firebase-key.json")
                firebase_admin.initialize_app(cred)
        return firestore.client()
    except Exception as e:
        return None

db = get_db()

# ==========================================
# 5. STYLING & HELPERS
# ==========================================

def get_custom_css():
    """Returns the CSS code to make the app look beautiful."""
    return """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        /* Main Background Image */
        .stApp { 
            background: url('https://images.unsplash.com/photo-1625246333195-78d9c38ad449?q=80&w=3870&auto=format&fit=crop'); 
            background-size: cover; 
            background-attachment: fixed; 
        }
        
        /* Make Sidebar White and Glassy */
        section[data-testid="stSidebar"] { 
            background-color: rgba(255, 255, 255, 0.95) !important; 
            border-right: 1px solid rgba(0,0,0,0.1); 
        }
        
        /* Styling for Cards and Containers */
        .stCard, div[data-testid="stFileUploader"], div[data-testid="stDataFrame"], 
        div[data-testid="stChatInput"], div[data-testid="stPlotlyChart"], 
        div[data-testid="stSpinner"], div[data-testid="stExpander"] { 
            background-color: rgba(255, 255, 255, 0.95); 
            border-radius: 12px; 
            padding: 20px; 
            box-shadow: 0 4px 10px rgba(0,0,0,0.05); 
            border: 1px solid #E5E7EB; 
            margin-bottom: 20px;
        }
        
        /* Green Buttons */
        .stButton>button { 
            background: #059669; 
            color: white; 
            border-radius: 8px; 
            border: none; 
            padding: 0.7rem 1.2rem; 
            font-weight: 600; 
            width: 100%; 
        } 
        .stButton>button:hover { background: #047857; }
        
        /* Headers and Text */
        h1, h2, h3 { color: #064E3B !important; font-family: 'Inter', sans-serif;}
    </style>
    """

def clean_markdown_for_pdf(text):
    """Cleans up the AI text so it looks good in a PDF."""
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text) # Bold text
    text = re.sub(r'##\s*(.*)', r'<br/><font size=14 color="#064E3B"><b>\1</b></font><br/>', text) # Headers
    text = text.replace('\n', '<br/>') # New lines
    return text

def create_pdf_report(disease: str, details_text: str) -> io.BytesIO:
    """Generates the downloadable PDF report."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
    styles = getSampleStyleSheet()
    
    # Define styles for Title and Body
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=26, textColor='#064E3B', spaceAfter=10, alignment=TA_CENTER)
    body_style = ParagraphStyle('Body', parent=styles['Normal'], fontSize=11, leading=16, spaceAfter=12)
    
    # Build the PDF content
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
    """Asks Google Gemini AI for details about the disease."""
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        prompt = (f"Explain {disease_name} in language: {lang_code}.\n"
                  "Return sections with headers: ## Causes, ## Symptoms, ## Precautions, ## Treatments.\n"
                  "Use bullet points for each section.")
        return model.generate_content(prompt).text
    except Exception:
        return "AI Service Unavailable."

def parse_sections(text):
    """Splits the AI response into separate tabs (Causes, Symptoms, etc.)."""
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
    """Simple check to see if the model loaded correctly."""
    if model is not None:
        return True
    return False