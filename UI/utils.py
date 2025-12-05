import streamlit as st
import requests
import datetime
import io
import re
import os
import google.generativeai as genai
import firebase_admin
from firebase_admin import credentials, firestore

# PDF Generation imports
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ============ CONFIGURATION ============

# 1. API KEYS & BACKEND
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
BACKEND_API_URL = "http://127.0.0.1:8000/predict"

# Configure Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ============ DATABASE ============

@st.cache_resource
def get_db():
    """Initializes and returns the Firestore client."""
    try:
        if not firebase_admin._apps:
            # Ensure you have firebase-key.json in your root directory
            cred = credentials.Certificate("firebase-key.json")
            firebase_admin.initialize_app(cred)
        return firestore.client()
    except Exception as e:
        print(f"Firebase Error: {e}")
        return None

# ============ STYLING ============

def get_custom_css():
    """Returns the CSS string for the application."""
    return """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        /* Background */
        .stApp {
            background: url('https://images.unsplash.com/photo-1625246333195-78d9c38ad449?q=80&w=3870&auto=format&fit=crop');
            background-size: cover;
            background-attachment: fixed;
        }
        
        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: rgba(255, 255, 255, 0.95) !important;
            border-right: 1px solid rgba(0,0,0,0.1);
        }
        
        /* Headers & Pills */
        .header-pill {
            background-color: rgba(255, 255, 255, 0.9);
            padding: 10px 20px;
            border-radius: 10px;
            display: inline-block;
            color: #064E3B;
            font-weight: 700;
            margin-bottom: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }
        
        /* Glass Widgets */
        div[data-testid="stFileUploader"], 
        div[data-testid="stDataFrame"], 
        div[data-testid="stChatInput"], 
        div[data-testid="stPlotlyChart"],
        div[data-testid="stSpinner"],
        div[data-testid="stExpander"] {
            background-color: rgba(255, 255, 255, 0.95);
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.05);
            margin-bottom: 20px;
            border: 1px solid #E5E7EB;
        }
        
        /* Spinner Text */
        div[data-testid="stSpinner"] p {
            font-size: 1.1rem !important; color: #064E3B !important; font-weight: 600;
        }

        /* Feature Cards */
        .stCard {
            background-color: rgba(255, 255, 255, 0.95) !important;
            padding: 25px;
            border-radius: 12px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05) !important;
            border: 1px solid #E5E7EB;
        }
        
        /* Dashboard Header */
        .hero-container {
            background-color: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            margin-bottom: 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .hero-text h1 { color: #064E3B; font-size: 2.5rem; margin:0;}
        .hero-text p { color: #374151; font-size: 1.1rem; margin-top:5px;}
        .hero-date { color: #059669; font-weight: 600; font-size: 1.2rem; text-align: right;}
        
        /* Buttons */
        .stButton>button {
            background: #059669; color: white; border-radius: 8px; border: none;
            padding: 0.7rem 1.2rem; font-weight: 600; width: 100%;
        }
        .stButton>button:hover { background: #047857; }
        
        /* Chat Bubbles */
        div[data-testid="stChatMessage"] {
            background-color: rgba(255, 255, 255, 0.95);
            border-radius: 12px;
            padding: 15px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            margin-bottom: 10px;
            border: 1px solid #E5E7EB;
        }
        div[data-testid="stChatMessage"] p { color: #374151; }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px; background-color: rgba(255, 255, 255, 0.95); padding: 10px;
            border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border: 1px solid #E5E7EB; margin-bottom: 20px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 45px; white-space: pre-wrap; background-color: transparent;
            border-radius: 8px; color: #374151; font-weight: 600; border: none;
        }
        .stTabs [aria-selected="true"] {
            background-color: #059669 !important; color: white !important; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
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
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    styles = getSampleStyleSheet()
    font_name = 'Helvetica'
    try:
        if os.path.exists("C:/Windows/Fonts/arial.ttf"):
            pdfmetrics.registerFont(TTFont('Arial', "C:/Windows/Fonts/arial.ttf"))
            pdfmetrics.registerFont(TTFont('Arial-Bold', "C:/Windows/Fonts/arialbd.ttf"))
            font_name = 'Arial'
    except Exception:
        pass

    title_style = ParagraphStyle(
        'Title',
        parent=styles['Heading1'],
        fontName=font_name if font_name == 'Helvetica' else 'Arial-Bold',
        fontSize=26,
        textColor='#064E3B',
        spaceAfter=10,
        alignment=TA_CENTER
    )
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName=font_name,
        fontSize=11,
        leading=16,
        spaceAfter=12
    )

    story = [
        Paragraph("Crop Disease Report", title_style),
        Paragraph(
            f"Diagnosis: {disease} | Date: {datetime.datetime.now().strftime('%Y-%m-%d')}",
            body_style
        ),
        Spacer(1, 20),
        Paragraph(clean_markdown_for_pdf(details_text), body_style)
    ]
    doc.build(story)
    buffer.seek(0)
    return buffer

def get_disease_details_gemini(disease_name, lang_code="en"):
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        prompt = (
            f"Explain {disease_name} in language: {lang_code}.\n"
            "Return sections with headers: ## Causes, ## Symptoms, ## Precautions, ## Treatments.\n"
            "Use bullet points for each section."
        )
        return model.generate_content(prompt).text
    except Exception:
        return "AI Service Unavailable."

def parse_sections(text):
    sections = {"Causes": "", "Symptoms": "", "Precautions": "", "Treatments": ""}
    curr = None
    for line in text.split('\n'):
        if "Causes" in line:
            curr = "Causes"
            continue
        if "Symptoms" in line:
            curr = "Symptoms"
            continue
        if "Precautions" in line:
            curr = "Precautions"
            continue
        if "Treatments" in line:
            curr = "Treatments"
            continue
        if curr:
            sections[curr] += line + "\n"
    return sections

def predict_via_api(image_file):
    # Reset pointer in case Streamlit preview consumed it
    image_file.seek(0)
    files = {"file": (image_file.name, image_file.getvalue(), image_file.type)}
    try:
        response = requests.post(BACKEND_API_URL, files=files, timeout=60)
        if response.status_code == 200:
            data = response.json()
            return data.get("disease", "Unknown"), float(data.get("confidence", 0.0))
        else:
            return f"Backend Error: {response.status_code}", 0.0
    except requests.exceptions.ConnectionError:
        return "Connection Error", 0.0
    except Exception as e:
        return f"Error: {e}", 0.0

def check_backend_status():
    try:
        requests.get(BACKEND_API_URL.replace("/predict", "/"), timeout=1)
        return True
    except Exception:
        return False