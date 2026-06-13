import streamlit as st
import pandas as pd
import numpy as np
import spacy
import plotly.express as px
import plotly.graph_objects as go
from PyPDF2 import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io
import base64
from datetime import datetime

# -----------------------------------------------------------------------------
# 1. CONFIGURATION & SETUP
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ResuMetric AI | Intelligent Resume Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Try loading spacy model, handle error if not installed
try:
    nlp = spacy.load("en_core_web_sm")
except:
    st.warning("SpaCy model not found. Using fallback logic. Run: python -m spacy download en_core_web_sm")
    nlp = None

# -----------------------------------------------------------------------------
# 2. CUSTOM CSS (PREMIUM & GLASSMORPHISM)
# -----------------------------------------------------------------------------
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        /* Global Font */
        html, body, [class="stApp"] {
            font-family: 'Inter', sans-serif;
            background-color: #f8f9fa;
        }
        
        /* Sidebar Customization */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1a1f35 0%, #2d344f 100%);
            border-right: 1px solid rgba(255,255,255,0.1);
        }
        
        /* Glassmorphism Card */
        .glass-card {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.5);
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.07);
            padding: 20px;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .glass-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.15);
        }

        /* Metrics Styling */
        .metric-card {
            background: #fff;
            border-radius: 12px;
            padding: 15px;
            text-align: center;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        }
        
        /* Animations */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .animate-fade {
            animation: fadeIn 0.8s ease-out forwards;
        }

        /* Headings */
        h1, h2, h3 {
            color: #1e293b;
            font-weight: 700;
        }
        
        /* Custom Buttons */
        .stButton>button {
            background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1.5rem;
            font-weight: 600;
            transition: all 0.3s;
        }
        .stButton>button:hover {
            transform: scale(1.02);
            box-shadow: 0 4px 15px rgba(79, 70, 229, 0.4);
        }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. LOGIC & DATA PROCESSING
# -----------------------------------------------------------------------------

# Job Role Definitions (Skills Database)
JOB_ROLES = {
    "Data Scientist": {
        "required": ["Python", "R", "SQL", "Machine Learning", "Deep Learning", "TensorFlow", "Pandas", "Scikit-learn", "Data Visualization", "Statistics", "NLP", "Big Data", "Tableau"],
        "soft": ["Analytical Thinking", "Communication", "Problem Solving"]
    },
    "AI Engineer": {
        "required": ["Python", "TensorFlow", "PyTorch", "Deep Learning", "Computer Vision", "NLP", "Linux", "Git", "Docker", "Kubernetes", "C++", "Algorithms"],
        "soft": ["Teamwork", "Agile", "Research"]
    },
    "Python Developer": {
        "required": ["Python", "Django", "Flask", "FastAPI", "SQL", "REST API", "Git", "Docker", "Linux", "JavaScript", "HTML", "CSS"],
        "soft": ["Code Optimization", "Documentation", "Debugging"]
    },
    "Data Analyst": {
        "required": ["Excel", "SQL", "Power BI", "Tableau", "Python", "Pandas", "Statistics", "Data Visualization", "Reporting"],
        "soft": ["Attention to Detail", "Presentation Skills"]
    },
    "Full Stack Developer": {
        "required": ["JavaScript", "React", "Angular", "Node.js", "HTML", "CSS", "SQL", "MongoDB", "Git", "REST API", "TypeScript"],
        "soft": ["Creativity", "Time Management"]
    }
}

# Common Skill Vocabulary for matching (Expanding the detection scope)
COMMON_SKILLS_DB = [
    "python", "java", "c++", "c#", "ruby", "go", "rust", "javascript", "typescript",
    "html", "css", "sql", "mysql", "postgresql", "mongodb", "oracle", "aws", "azure",
    "gcp", "docker", "kubernetes", "linux", "unix", "git", "jenkins", "jira",
    "machine learning", "deep learning", "neural networks", "nlp", "computer vision",
    "artificial intelligence", "data science", "analytics", "statistics",
    "tableau", "power bi", "excel", "word", "powerpoint", "communication",
    "project management", "agile", "scrum", "java spring", "django", "flask", "fastapi",
    "react", "angular", "vue", "node.js", "flutter", "swift", "ui/ux", "figma"
]

def extract_text_from_file(uploaded_file):
    """Extracts text from PDF or DOCX files."""
    text = ""
    try:
        if uploaded_file.name.endswith('.pdf'):
            pdf_reader = PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        elif uploaded_file.name.endswith('.docx'):
            doc = Document(uploaded_file)
            for para in doc.paragraphs:
                text += para.text + "\n"
    except Exception as e:
        st.error(f"Error reading file: {e}")
    return text

def analyze_resume(text, selected_role):
    """Analyzes resume text against job requirements."""
    text_lower = text.lower()
    
    # 1. Extract Skills (Simple Keyword Matching + NLP)
    found_skills = []
    for skill in COMMON_SKILLS_DB:
        if skill in text_lower:
            found_skills.append(skill.title() if skill.lower() != "nlp" else "NLP") # Capitalize properly
    
    # Deduplicate
    found_skills = list(set(found_skills))
    
    # 2. Determine Missing Skills
    req_skills = JOB_ROLES[selected_role]["required"]
    missing_skills = [skill for skill in req_skills if skill.lower() not in text_lower]
    
    # 3. ATS Scoring Logic
    # Base score
    ats_score = 50
    
    # Length check (Penalty if too short)
    if len(text.split()) < 200: ats_score -= 10
    elif len(text.split()) > 1000: ats_score += 10
    
    # Keyword match score
    match_count = sum(1 for s in req_skills if s.lower() in text_lower)
    match_score = (match_count / len(req_skills)) * 40
    
    ats_score += match_score
    
    # Clamp score
    ats_score = min(max(int(ats_score), 0), 100)
    
    return found_skills, missing_skills, ats_score

# -----------------------------------------------------------------------------
# 4. PAGE SECTIONS
# -----------------------------------------------------------------------------

def sidebar_nav():
    with st.sidebar:
        st.title("ResuMetric AI 🧠")
        st.markdown("---")
        
        st.header("Configuration")
        selected_job = st.selectbox(
            "Target Job Role",
            list(JOB_ROLES.keys()),
            index=0
        )
        
        st.markdown("---")
        st.markdown("### 🛠️ Tools Used")
        st.markdown("""
        - **Streamlit**
        - **SpaCy NLP**
        - **Pandas/Plotly**
        - **Scikit-Learn**
        """)
        
    return selected_job

def main_dashboard(role):
    # Hero Section
    st.markdown("""
    <div class="glass-card animate-fade" style="text-align:center; padding: 40px; margin-bottom: 30px;">
        <h1 style="margin-bottom: 10px;">AI-Powered Resume Analyzer</h1>
        <p style="font-size: 18px; color: #475569;">
            Upload your resume to parse skills, calculate ATS score, and detect skill gaps 
            for the <b>{}</b> role.
        </p>
    </div>
    """.format(role), unsafe_allow_html=True)
    
    # Upload Section
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 📄 Upload Resume")
        uploaded_file = st.file_uploader("Upload PDF or DOCX", type=["pdf", "docx"])
        
    if uploaded_file:
        with st.spinner('Analyzing Resume with AI 🤖...'):
            # Extract and Analyze
            raw_text = extract_text_from_file(uploaded_file)
            found_skills, missing_skills, ats_score = analyze_resume(raw_text, role)
            
            # Save to session state to persist in other columns
            st.session_state['analyzed'] = True
            st.session_state['ats_score'] = ats_score
            st.session_state['found_skills'] = found_skills
            st.session_state['missing_skills'] = missing_skills
            st.session_state['role'] = role

    # Results Section (Only if analyzed)
    if st.session_state.get('analyzed'):
        display_analysis_results()

def display_analysis_results():
    st.markdown("---")
    st.markdown("<h2 style='color: #1e293b;'>Analysis Dashboard</h2>", unsafe_allow_html=True)
    
    # 1. Top Metrics
    m1, m2, m3, m4 = st.columns(4)
    
    score = st.session_state['ats_score']
    found = len(st.session_state['found_skills'])
    missing = len(st.session_state['missing_skills'])