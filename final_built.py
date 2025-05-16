# Imports
import streamlit as st
import pandas as pd
import numpy as np
import io
import graphviz
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import plotly.graph_objects as go
import plotly.express as px
from streamlit_plotly_events import plotly_events
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import random
import time

# Page setup
st.set_page_config(page_title="PPIcheck.ai Delta Built_1.0", layout="wide")

# Update the CSS section with more visible colors
st.markdown("""
<style>
    /* Main sidebar container */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0D1B2A 0%, #1B263B 50%, #415A77 100%) !important;
        height: 100% !important;
        overflow-y: auto !important;
        position: relative !important;
        max-height: 100vh !important;
    }

    /* Custom scrollbar styling */
    section[data-testid="stSidebar"]::-webkit-scrollbar {
        width: 10px !important;
    }

    section[data-testid="stSidebar"]::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1) !important;
        border-radius: 5px !important;
    }

    section[data-testid="stSidebar"]::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.3) !important;
        border-radius: 5px !important;
        border: 2px solid rgba(255, 255, 255, 0.1) !important;
    }

    section[data-testid="stSidebar"]::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.5) !important;
    }

    /* Firefox scrollbar styling */
    section[data-testid="stSidebar"] {
        scrollbar-width: thin !important;
        scrollbar-color: rgba(255, 255, 255, 0.3) rgba(255, 255, 255, 0.1) !important;
    }

    /* Sidebar headers */
    section[data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        height: auto !important;
        overflow-y: visible !important;
        max-height: none !important;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #E0E1DD !important;
        font-size: 1.3em !important;
        font-weight: 600 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        padding: 10px 0;
        border-bottom: 2px solid rgba(255,255,255,0.2);
        margin-bottom: 15px;
    }

    /* Sidebar text and labels */
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stNumberInput label,
    section[data-testid="stSidebar"] .stSelectbox label {
        color: #000000 !important;
        color: #FFFFFF !important;  /* Changed to white */
        font-size: 1em !important;
        font-weight: 500 !important;
    }

    /* Sidebar input fields */
    section[data-testid="stSidebar"] input,
    section[data-testid="stSidebar"] .stSelectbox > div > div,
    section[data-testid="stSidebar"] .stNumberInput input,
    section[data-testid="stSidebar"] .stSelectbox select {
        background-color: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        color: #000000 !important;
        font-size: 1em !important;
    }

    /* Sidebar selectbox */
    section[data-testid="stSidebar"] .stSelectbox > div {
        background-color: transparent !important;
    }

    /* Sidebar tabs */
    section[data-testid="stSidebar"] .stTabs [data-baseweb="tab-list"] {
        background-color: rgba(255,255,255,0.1) !important;
        border-radius: 5px;
        padding: 5px;
    }

    section[data-testid="stSidebar"] .stTabs [data-baseweb="tab"] {
        color: #E0E1DD !important;
        font-weight: 500 !important;
    }

    section[data-testid="stSidebar"] .stTabs [aria-selected="true"] {
        background-color: rgba(224, 225, 221, 0.2) !important;
        color: #FFFFFF !important;
    }

    /* Sidebar checkboxes */
    section[data-testid="stSidebar"] .stCheckbox {
        color: #E0E1DD !important;
    }

    section[data-testid="stSidebar"] .stCheckbox > label {
        font-size: 0.95em !important;
    }

    /* Sidebar dividers */
    section[data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.2);
        margin: 20px 0;
    }

    /* Warning messages in sidebar */
    section[data-testid="stSidebar"] .stAlert {
        background-color: rgba(255,183,77,0.2) !important;
        color: #FFB74D !important;
        border: 1px solid rgba(255,183,77,0.3) !important;
        padding: 10px !important;
        border-radius: 5px !important;
        font-weight: 500 !important;
    }

    /* Number input arrows */
    section[data-testid="stSidebar"] .stNumberInput div[data-baseweb="input"] span {
        color: #E0E1DD !important;
    }

    /* Multiselect dropdown */
    section[data-testid="stSidebar"] .stMultiSelect > div {
        background-color: rgba(255, 255, 255, 0.1) !important;
        border-color: rgba(255, 255, 255, 0.2) !important;
    }

    section[data-testid="stSidebar"] .stMultiSelect > div > div {
        color: #E0E1DD !important;
    }
    /* Enhanced Table Styling */
    .stTable {
        border-collapse: separate;
        border-spacing: 0;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    /* Column Headers */
    .stTable thead th {
        background-color: #0D1B2A;
        color: #FFFFFF;
        font-weight: 600;
        padding: 12px 16px;
        border-bottom: 2px solid rgba(255,255,255,0.2);
        text-align: left;
    }
    /* Row Headers (first column) */
    .stTable tbody tr td:first-child {
        background-color: #1B263B;
        color: #FFFFFF;
        font-weight: 600;
        padding: 12px 16px;
        border-right: 2px solid rgba(255,255,255,0.2);
    }
    /* Data Cells */
    .stTable tbody tr {
        background-color: #FFFFFF;
        transition: background-color 0.3s ease;
    }
    .stTable tbody tr:hover {
        background-color: rgba(27, 38, 59, 0.05);
    }
    .stTable td {
        padding: 10px 14px;
        color: #333333;
        border: 1px solid #E0E1DD;
    }
    /* Remove bottom border from last row */
    .stTable tr:last-child td {
        border-bottom: none;
    }
    /* Ensure text alignment */
    .stTable td, .stTable th {
        text-align: left;
        vertical-align: middle;
    }

    /* Add smooth scrolling */
    section[data-testid="stSidebar"] {
        scroll-behavior: smooth !important;
    }

    /* Ensure proper spacing between elements */
    section[data-testid="stSidebar"] > div {
        padding-bottom: 2rem !important;
    }

    /* Fix for Streamlit deployment */
    section[data-testid="stSidebar"] > div:first-child {
        height: auto !important;
        overflow-y: auto !important;
    }

    /* Ensure sidebar content is visible */
    section[data-testid="stSidebar"] .element-container {
        overflow: visible !important;
    }

    /* Fix for sidebar scrolling in deployed environment */
    div[data-testid="stSidebar"] {
        overflow-y: auto !important;
        height: 100% !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Clean, clinical, modern background */
.stApp {
    background: linear-gradient(to right, #f4f6f8, #d7e1ec) !important;
    background-attachment: fixed;
    font-family: 'Segoe UI', sans-serif;
    color: #1a1a1a;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Style primary buttons (e.g., Log Input Data) with orange theme */
button[kind="primary"] {
    background: linear-gradient(to right, #ff8c42, #ff6f00);
    color: #ffffff !important;
    font-weight: bold;
    border: none;
    border-radius: 8px;
    padding: 10px 20px;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
    transition: all 0.2s ease-in-out;
}

button[kind="primary"]:hover {
    background: linear-gradient(to right, #ffa94d, #ff7e1a);
    transform: scale(1.03);
    box-shadow: 0 6px 15px rgba(0, 0, 0, 0.3);
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Force orange styling on all buttons */
div.stButton > button {
    background: linear-gradient(to right, #ff8c42, #ff6f00);
    color: #ffffff !important;
    font-weight: bold;
    border: none;
    border-radius: 8px;
    padding: 10px 20px;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.25);
    transition: all 0.3s ease-in-out;
}

div.stButton > button:hover {
    background: linear-gradient(to right, #ffa94d, #ff7e1a);
    transform: scale(1.03);
    box-shadow: 0 6px 15px rgba(0, 0, 0, 0.3);
}
</style>
""", unsafe_allow_html=True)

# Enhanced title and introduction with more robust styling
st.markdown("""
<style>
    .title-container {
        background: linear-gradient(180deg, #0D1B2A 0%, #1B263B 50%, #415A77 100%) !important;
        border-radius: 20px;
        padding: 120px 40px;  /* Doubled vertical padding */
        margin: -20px -20px 60px -20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        width: calc(100% + 40px);
        min-height: 600px;  /* Doubled minimum height */
        display: flex;
        flex-direction: column;
        justify-content: center;
        position: relative;
    }
    .main-title {
        color: #FFFFFF !important;
        font-size: 4.5em;  /* Increased font size */
        font-weight: 700;
        text-align: center;
        margin-bottom: 60px;  /* Increased margin */
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        font-family: 'Arial', sans-serif;
        padding: 20px 0;
    }
    .subtitle {
        color: #FFFFFF !important;
        font-size: 1.4em;  /* Reduced from 1.8em */
        max-width: 1400px;
        margin: 0 auto;
        line-height: 1.6;  /* Reduced from 2 */
        text-align: center;
        font-family: 'Arial', sans-serif;
        padding: 0 40px;
    }

    .subtitle span {
        font-size: 0.8em;  /* Reduced from 0.9em */
        line-height: 1.4;
        opacity: 0.9;
    }

    /* Add specific styling for the headline */
    .subtitle .headline {
        font-size: 1.1em;
        font-weight: 600;
        margin-bottom: 10px;
    }

    /* Add specific styling for the description */
    .subtitle .description {
        font-size: 0.9em;
        line-height: 1.5;
    }

    /* Add a gradient overlay at the bottom */
    .title-container::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        height: 150px;  /* Increased height of gradient */
        background: linear-gradient(to bottom, transparent, rgba(13, 27, 42, 0.1));
        pointer-events: none;
    }

    /* Add a subtle glow effect */
    .title-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 200px;
        background: radial-gradient(circle at 50% 0%, rgba(255, 255, 255, 0.1), transparent);
        pointer-events: none;
    }
</style>

<div class="title-container">
    <div class="main-title">
        PPIcheck.ai Delta Built_1.0
    </div>
    <div class="subtitle">
        <div class="headline">India's First Indigenously-Built AI Tool for Optimizing PPI Therapy</div>
        <br>
        <div class="description">
        PPIcheck.ai is revolutionizing the way clinicians make PPI deprescribing decisions.
        <br><br>
        Our application combines established clinical guidelines with cutting-edge machine learning to provide evidence-based risk assessments, enabling healthcare professionals to optimize PPI therapy effectively.
        </div>
        <br><br>
        <span style="font-weight: 500;">
            Developed by Dr. Nabyendu Biswas, Department of Pharmacology, in collaboration with the Medicine Department, MKCG Medical College & Hospital
        </span>
    </div>
</div>

<style>
    /* Remove the default streamlit padding */
    .block-container {
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }

    /* Ensure the content below has proper spacing */
    .stMarkdown {
        margin-top: 3rem;
    }

    /* Add some spacing between sections */
    .stMarkdown + .stMarkdown {
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("---")  # Adding horizontal line for separation

# Create the header and expander for PPI Guidelines
st.markdown("""
    <div style="
        background: linear-gradient(135deg, #155799, #159957);
        padding: 20px;
        border-radius: 15px;
        margin: 20px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        cursor: pointer;
    ">
        <h2 style="
            color: white;
            font-size: 2.2em;
            text-align: center;
            margin: 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        ">📋 PPI Dosing Guidelines</h2>
    </div>
""", unsafe_allow_html=True)

guidelines_expander = st.expander("Click to view guidelines", expanded=False)
with guidelines_expander:
    # Create tabs for different guideline sections
    tab1, tab2, tab3 = st.tabs(["Dosing Table", "Additional Considerations", "Administration"])

    with tab1:
        # Create DataFrame with PPI guidelines
        ppi_data = {
            'Medication': ['Esomeprazole', 'Lansoprazole', 'Pantoprazole', 'Pantoprazole', 'Rabeprazole'],
            'Route': ['Oral (tablet/cap)', 'Oral (tablet/cap)', 'Oral (tablet/cap)', 'IV', 'Oral (tablet/cap)'],
            'Standard Dose (mg)': [20, '15 (OTC)', 40, 40, 20],
            'Maximum Dose': ['40 twice daily', '90 twice daily*', '40 twice daily', '40 twice daily', '60 twice daily*'],
            'Administration': ['Swallow whole with water', 'Swallow whole with water', 'Swallow whole with water',
                             'IV infusion as directed', 'Swallow whole with water'],
            'Timing': ['At least 1 hour before meal', '30-60 min before meal', '30-60 min before meal',
                      'Meal timing not relevant', '30-60 min before meal']
        }
        df = pd.DataFrame(ppi_data)

        # Display the table with custom styling
        st.dataframe(
            df,
            hide_index=True,
            column_config={
                "Medication": st.column_config.TextColumn(
                    "Medication",
                    width="medium",
                ),
                "Route": st.column_config.TextColumn(
                    "Route",
                    width="medium",
                ),
                "Standard Dose (mg)": st.column_config.TextColumn(
                    "Standard Dose (mg)",
                    width="small",
                ),
                "Maximum Dose": st.column_config.TextColumn(
                    "Maximum Dose",
                    width="medium",
                ),
                "Administration": st.column_config.TextColumn(
                    "Administration",
                    width="large",
                ),
                "Timing": st.column_config.TextColumn(
                    "Timing",
                    width="large",
                ),
            },
            use_container_width=True,
        )

        # Add footnote
        st.markdown("""
            <p style="
                font-size: 0.8em;
                color: #666;
                margin-top: 15px;
                font-style: italic;
            ">* Higher doses may be required for specific conditions under specialist supervision.</p>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown("""
        ### Duration of Therapy
        - **Acute conditions:** 4-8 weeks
        - **Chronic conditions:** Individualized
        - **Long-term use:** Regular review required
        - **Note:** Long-term PPI therapy is generally considered > 8 weeks or > 2 months.

        ### Special Populations
        - **Elderly patients:**
            - Consider lower doses
            - Monitor more frequently
        - **Hepatic impairment:**
            - No dose adjustment needed
            - Monitor liver function
        - **Renal impairment:**
            - No dose adjustment needed
            - Monitor renal function in severe cases
        """)

    with tab3:
        st.markdown("""
        ### Oral Administration
        - Take on empty stomach
        - Swallow whole; do not crush/chew
        - Consistent timing daily
        - Take with full glass of water
        - Avoid concurrent antacids

        ### IV Administration
        - Follow institutional protocols
        - Monitor for injection site reactions
        - Consider transition to oral therapy
        - Regular line care and monitoring
        - Document administration times
        """)

# Patient Information Section
st.sidebar.header("Patient Information")

# 1. Demographics Section
st.sidebar.subheader("Demographics")
col_demo1, col_demo2 = st.sidebar.columns(2)
with col_demo1:
    patient_age = st.number_input("Age", min_value=0, max_value=120, value=60)
    patient_height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
with col_demo2:
    patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    patient_weight = st.number_input("Weight (kg)", min_value=20, max_value=200, value=70)

bmi = patient_weight / ((patient_height/100) ** 2)
st.sidebar.write(f"BMI: {bmi:.1f} kg/m²")

# 3. Treatment History (Revised)
st.sidebar.markdown("---")
st.sidebar.subheader("Treatment History")

# Checkbox for previous history
previous_ppi_history = st.sidebar.checkbox("Previous PPI Therapy History Present")

# Conditional input for duration if history is present
ppi_duration_days = 0
if previous_ppi_history:
    ppi_duration_days = st.sidebar.number_input("Oral PPI Therapy Duration (days)", min_value=0, max_value=3650, value=0) # Max 10 years in days
    if ppi_duration_days > (8 * 30): # Check if duration is longer than ~8 weeks (long-term)
        st.sidebar.warning("⚠️ History of long-term PPI use detected")


# 4. Clinical Indications
st.sidebar.markdown("---")
st.sidebar.subheader("Clinical Indications")

# Create tabs with better organization
tabs = st.sidebar.tabs(["GI", "NSAID/AP", "Other"])

# Organize indications by category with descriptions
gi_indications = {
    "Non-variceal bleeding": "Active upper GI bleeding",
    "Dyspepsia": "Persistent upper abdominal pain or discomfort",
    "GERD & complications": "Gastroesophageal reflux disease and related complications",
    "H pylori infection": "Active Helicobacter pylori infection",
    "Peptic ulcer treatment": "Active gastric or duodenal ulcer",
    "Zollinger-Ellison syndrome": "Rare condition causing increased acid production"
}

nsaid_ap_indications = {
    "Prevent NSAID ulcers": "Prophylaxis for NSAID-induced ulcers",
    "NSAID & ulcer/GIB history": "NSAID use with history of ulcer or GI bleeding",
    "NSAID & age > 60": "NSAID use in patients over 60 years",
    "NSAID + cortico/antiplatelet/anticoag": "NSAID with other high-risk medications",
    "Prophylaxis in high risk antiplatelet users": "Prevention in high-risk AP users",
    "Antiplatelet & ulcer/GIB history": "AP use with ulcer history",
    "Antiplatelet + age > 60 or dyspepsia/GERD": "AP use in elderly or with GI symptoms",
    "Antiplatelet + cortico/NSAID/anticoag": "AP with other high-risk medications"
}

other_indications = {
    "Stress ulcer prophylaxis": "Prevention of stress-related mucosal damage",
    "Coagulopathy (platelet < 50k, INR ≥ 1.5)": "Bleeding risk due to coagulation disorders",
    "Mechanical ventilation > 48h": "Extended mechanical ventilation requiring prophylaxis"
}

# Display indications in tabs with tooltips and better formatting
with tabs[0]:
    selected_gi = []
    for ind, desc in gi_indications.items():
        if st.checkbox(ind, help=desc, key=f"gi_{ind}"):
            selected_gi.append(ind)

with tabs[1]:
    selected_nsaid_ap = []
    for ind, desc in nsaid_ap_indications.items():
        if st.checkbox(ind, help=desc, key=f"nsaid_{ind}"):
            selected_nsaid_ap.append(ind)

with tabs[2]:
    selected_other = []
    for ind, desc in other_indications.items():
        if st.checkbox(ind, help=desc, key=f"other_{ind}"):
            selected_other.append(ind)

# Combine all selected indications
selected_indications = selected_gi + selected_nsaid_ap + selected_other

# Add this CSS specifically for tabs styling
st.markdown("""
<style>
    /* Enhanced Tab Container */
    section[data-testid="stSidebar"] .stTabs {
        background: rgba(255, 255, 255, 0.05);
        padding: 10px;
        border-radius: 8px;
        margin: 10px 0;
    }

    /* Tab List Styling */
    section[data-testid="stSidebar"] [data-baseweb="tab-list"] {
        background: rgba(13, 27, 42, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px !important;
        padding: 5px !important;
        gap: 5px !important;
    }

    /* Individual Tab Styling */
    section[data-testid="stSidebar"] [data-baseweb="tab"] {
        background: transparent !important;
        border-radius: 6px !important;
        color: #FFFFFF !important;
        font-weight: 500 !important;
        padding: 8px 16px !important;
        margin: 0 2px !important;
        transition: all 0.3s ease !important;
    }

    /* Selected Tab */
    section[data-testid="stSidebar"] [aria-selected="true"] {
        background: linear-gradient(135deg, #155799, #159957) !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }

    /* Hover effect for tabs */
    section[data-testid="stSidebar"] [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.1) !important;
        transform: translateY(-1px);
    }

    /* Tab Content Area */
    section[data-testid="stSidebar"] [data-baseweb="tab-panel"] {
        padding: 15px 10px !important;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 8px;
        margin-top: 10px;
    }

    /* Checkbox container in tabs */
    section[data-testid="stSidebar"] .stTabs .stCheckbox {
        background: rgba(255, 255, 255, 0.05);
        padding: 10px 15px;
        border-radius: 6px;
        margin: 5px 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }

    /* Checkbox hover effect */
    section[data-testid="stSidebar"] .stTabs .stCheckbox:hover {
        background: rgba(255, 255, 255, 0.1);
        transform: translateX(2px);
    }

    /* Checkbox label */
    section[data-testid="stSidebar"] .stTabs .stCheckbox label {
        color: #FFFFFF !important;
        font-size: 0.95em !important;
        font-weight: 500 !important;
        letter-spacing: 0.2px;
    }

    /* Checkbox checked state */
    section[data-testid="stSidebar"] .stTabs .stCheckbox [data-baseweb="checkbox"] [data-checked="true"] {
        background-color: #159957 !important;
        border-color: #159957 !important;
    }
    /* Enhanced Table Styling */
    .stTable {
        border-collapse: separate;
        border-spacing: 0;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    /* Column Headers */
    .stTable thead th {
        background-color: #0D1B2A;
        color: #FFFFFF;
        font-weight: 600;
        padding: 12px 16px;
        border-bottom: 2px solid rgba(255,255,255,0.2);
        text-align: left;
    }
    /* Row Headers (first column) */
    .stTable tbody tr td:first-child {
        background-color: #1B263B;
        color: #FFFFFF;
        font-weight: 600;
        padding: 12px 16px;
        border-right: 2px solid rgba(255,255,255,0.2);
    }
    /* Data Cells */
    .stTable tbody tr {
        background-color: #FFFFFF;
        transition: background-color 0.3s ease;
    }
    .stTable tbody tr:hover {
        background-color: rgba(27, 38, 59, 0.05);
    }
    .stTable td {
        padding: 10px 14px;
        color: #333333;
        border: 1px solid #E0E1DD;
    }
    /* Remove bottom border from last row */
    .stTable tr:last-child td {
        border-bottom: none;
    }
    /* Ensure text alignment */
    .stTable td, .stTable th {
        text-align: left;
        vertical-align: middle;
    }
</style>
""", unsafe_allow_html=True)

# Add this CSS specifically for Clinical Indications section
st.markdown("""
<style>
    /* Clinical Indications section styling */
    section[data-testid="stSidebar"] .stCheckbox {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 8px 12px;
        border-radius: 4px;
        margin: 4px 0;
    }

    /* Make all text bright white */
    section[data-testid="stSidebar"] .stCheckbox label,
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stNumberInput label,
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span {
        color: #FFFFFF !important;
        font-weight: 500 !important;
    }

    /* Tooltip text */
    section[data-testid="stSidebar"] [data-baseweb="tooltip"] {
        color: #FFFFFF !important;
        background-color: #1B263B !important;
        padding: 8px !important;
        border-radius: 4px !important;
        font-size: 0.9em !important;
        max-width: 300px !important;
        white-space: normal !important;
    }

    /* Tab labels */
    section[data-testid="stSidebar"] .stTabs [data-baseweb="tab"] {
        color: #FFFFFF !important;
        font-weight: 500 !important;
    }

    /* Selected tab */
    section[data-testid="stSidebar"] .stTabs [aria-selected="true"] {
        background-color: rgba(255, 255, 255, 0.2) !important;
        color: #FFFFFF !important;
    }

    /* Input text color */
    section[data-testid="stSidebar"] input,
    section[data-testid="stSidebar"] select,
    section[data-testid="stSidebar"] .stSelectbox > div > div {
        color: #FFFFFF !important;
    }
    /* Enhanced Table Styling */
    .stTable {
        border-collapse: separate;
        border-spacing: 0;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    /* Column Headers */
    .stTable thead th {
        background-color: #0D1B2A;
        color: #FFFFFF;
        font-weight: 600;
        padding: 12px 16px;
        border-bottom: 2px solid rgba(255,255,255,0.2);
        text-align: left;
    }
    /* Row Headers (first column) */
    .stTable tbody tr td:first-child {
        background-color: #1B263B;
        color: #FFFFFF;
        font-weight: 600;
        padding: 12px 16px;
        border-right: 2px solid rgba(255,255,255,0.2);
    }
    /* Data Cells */
    .stTable tbody tr {
        background-color: #FFFFFF;
        transition: background-color 0.3s ease;
    }
    .stTable tbody tr:hover {
        background-color: rgba(27, 38, 59, 0.05);
    }
    .stTable td {
        padding: 10px 14px;
        color: #333333;
        border: 1px solid #E0E1DD;
    }
    /* Remove bottom border from last row */
    .stTable tr:last-child td {
        border-bottom: none;
    }
    /* Ensure text alignment */
    .stTable td, .stTable th {
        text-align: left;
        vertical-align: middle;
    }
</style>
""", unsafe_allow_html=True)

# 5. Current Medications
st.sidebar.markdown("---")
st.sidebar.subheader("Current Medications")

# PPI Inputs
selected_ppi = st.sidebar.selectbox("Select PPI", ["None", "Pantoprazole", "Omeprazole", "Esomeprazole", "Rabeprazole"])
ppi_dose = st.sidebar.selectbox("PPI Dose (mg)", [0, 20, 40, 80])
ppi_route = st.sidebar.selectbox("PPI Route", ["None", "Oral", "IV"])

# NSAID Section with new grouping
nsaid_groups = {
    "Propionic acid derivatives (Profens)": {
        "Ibuprofen": ([0, 200, 400, 600, 800, 2400], "Usual: 200–600 mg; Max: 2400 mg/day", 2400, 3),
        "Naproxen": ([0, 250, 375, 500, 1000], "Usual: 250–500 mg; Max: 1000 mg/day", 1000, 6),
        "Ketoprofen": ([0, 50, 100, 200], "Usual: 50–100 mg; Max: 200 mg/day", 200, 4),
        "Flurbiprofen": ([0, 50, 100, 150, 300], "Usual: 50–150 mg; Max: 300 mg/day", 300, 3)
    },
    "Acetic acid derivatives": {
        "Indomethacin": ([0, 25, 50, 75, 200], "Usual: 25–50 mg; Max: 200 mg/day", 200, 5),
        "Diclofenac": ([0, 25, 50, 75, 100, 150], "Usual: 50–75 mg; Max: 150 mg/day", 150, 4),
        "Etodolac": ([0, 200, 300, 400, 1000], "Usual: 200–400 mg; Max: 1000 mg/day", 1000, 3),
        "Ketorolac": ([0, 10, 20, 30, 120], "Usual: 10–30 mg; Max: 120 mg/day", 120, 4)
    },
    "Enolic acid (Oxicam) derivatives": {
        "Piroxicam": ([0, 10, 20], "Usual: 10–20 mg; Max: 20 mg/day", 20, 4),
        "Meloxicam": ([0, 7.5, 15], "Usual: 7.5–15 mg; Max: 15 mg/day", 15, 2)
    },
    "Selective COX-2 inhibitors": {
        "Celecoxib": ([0, 100, 200, 400], "Usual: 100–200 mg; Max: 400 mg/day", 400, 1)
    },
    "Non-NSAID Analgesics": {
        "Paracetamol": ([0, 500, 1000, 2000, 4000], "Usual: 500–1000 mg; Max: 4000 mg/day", 4000, 0)
    },
    "None": {"None": ([0], "", 0, 0)}
}

selected_nsaid_group = st.sidebar.selectbox("Select NSAID Group", list(nsaid_groups.keys()))
selected_nsaid = st.sidebar.selectbox("Select NSAID", list(nsaid_groups[selected_nsaid_group].keys()))
nsaid_info = nsaid_groups[selected_nsaid_group][selected_nsaid]
nsaid_dose_options, nsaid_help, nsaid_max_dose, nsaid_base_risk = nsaid_info
nsaid_dose = st.sidebar.selectbox("NSAID Dose (mg)", nsaid_dose_options, help=nsaid_help)
nsaid_route = st.sidebar.selectbox("NSAID Route", ["None", "Oral", "Parenteral"])

# Custom CSS for guidelines buttons
st.markdown("""
    <style>
    /* Guidelines Button Styling */
    .guidelines-button {
        background: linear-gradient(120deg, #1B263B, #415A77);
        border-radius: 4px;
        padding: 10px 15px;
        margin: 8px 0;
        cursor: pointer;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.2s ease;
        position: relative;
    }

    .guidelines-button:hover {
        background: linear-gradient(120deg, #233148, #4A6484);
        box-shadow: 0 3px 6px rgba(0, 0, 0, 0.15);
    }

    .guidelines-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 8px;
    }

    .guidelines-label {
        color: #FFFFFF;
        font-size: 0.9em;
        font-weight: 500;
        letter-spacing: 0.3px;
        font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    }

    .guidelines-label::after {
        content: "▼";
        margin-left: 8px;
        font-size: 0.8em;
        opacity: 0.8;
    }

    .guidelines-badge {
        background: rgba(255, 255, 255, 0.15);
        color: #FFFFFF;
        padding: 2px 6px;
        border-radius: 3px;
        font-size: 0.75em;
        font-weight: 500;
        letter-spacing: 0.5px;
    }

    /* Content Styling */
    .guidelines-content {
        background: #FFFFFF;
        padding: 15px;
        border-radius: 4px;
        margin-top: 5px;
        font-size: 0.95em;
        line-height: 1.5;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
    }

    .guidelines-content h4 {
        color: #1B263B;
        font-size: 1.05em;
        font-weight: 600;
        margin: 12px 0 8px 0;
        padding-bottom: 4px;
        border-bottom: 1px solid rgba(27, 38, 59, 0.1);
    }

    .guidelines-content h4:first-child {
        margin-top: 0;
    }

    .drug-info {
        padding-left: 15px;
        margin-bottom: 12px;
        color: #333333;
        font-size: 0.95em;
    }

    .note-text {
        font-size: 0.9em;
        color: #666666;
        font-style: italic;
        margin-top: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# Antiplatelet section - Updated with new drugs and details
st.sidebar.markdown("""
    <div class="guidelines-button" onclick="this.nextElementSibling.click()">
        <div class="guidelines-header">
            <div class="guidelines-label">VIEW ANTIPLATELET GUIDELINES</div>
            <div class="guidelines-badge">9 DRUGS</div>
        </div>
    </div>
""", unsafe_allow_html=True)

with st.sidebar.expander(""):
    st.markdown("""
        <div class='guidelines-content'>
        <h4>1. Aspirin</h4>
        <div class='drug-info'>
        • Route: Oral<br>
        • Lowest Dose: 75 mg once daily<br>
        • Highest Dose: 325 mg once daily<br>
        • Gastric Risk: High
        </div>

        <h4>2. Clopidogrel</h4>
        <div class='drug-info'>
        • Route: Oral<br>
        • Lowest Dose: 75 mg once daily<br>
        • Highest Dose: 300 mg loading dose<br>
        • Gastric Risk: Moderate
        </div>

        <h4>3. Ticagrelor</h4>
        <div class='drug-info'>
        • Route: Oral<br>
        • Lowest Dose: 60 mg twice daily<br>
        • Highest Dose: 180 mg loading dose<br>
        • Gastric Risk: Moderate
        </div>

        <h4>4. Prasugrel</h4>
        <div class='drug-info'>
        • Route: Oral<br>
        • Lowest Dose: 5 mg once daily<br>
        • Highest Dose: 60 mg loading; 10 mg maintenance<br>
        • Gastric Risk: Moderate
        </div>

        <h4>5. Dipyridamole</h4>
        <div class='drug-info'>
        • Route: Oral<br>
        • Lowest Dose: 75 mg twice daily<br>
        • Highest Dose: 200 mg sustained release twice daily<br>
        • Gastric Risk: Low
        </div>

        <h4>6. Ticlopidine</h4>
        <div class='drug-info'>
        • Route: Oral<br>
        • Lowest Dose: 250 mg twice daily<br>
        • Highest Dose: 500 mg daily<br>
        • Gastric Risk: Moderate
        </div>

        <h4>7. Abciximab</h4>
        <div class='drug-info'>
        • Route: IV<br>
        • Lowest Dose: 0.25 mg/kg bolus<br>
        • Highest Dose: 0.125 mcg/kg/min infusion<br>
        • Gastric Risk: Low
        </div>

        <h4>8. Eptifibatide</h4>
        <div class='drug-info'>
        • Route: IV<br>
        • Lowest Dose: 180 mcg/kg bolus<br>
        • Highest Dose: 2 mcg/kg/min infusion<br>
        • Gastric Risk: Low
        </div>

        <h4>9. Tirofiban</h4>
        <div class='drug-info'>
        • Route: IV<br>
        • Lowest Dose: 0.4 mcg/kg/min bolus + infusion<br>
        • Highest Dose: 0.15 mcg/kg/min infusion<br>
        • Gastric Risk: Low
        </div>
        </div>
    """, unsafe_allow_html=True)

# Anticoagulant Section - Updated with new drugs and details
st.sidebar.markdown("""
    <div class="guidelines-button" onclick="this.nextElementSibling.click()">
        <div class="guidelines-header">
            <div class="guidelines-label">VIEW ANTICOAGULANT GUIDELINES</div>
            <div class="guidelines-badge">6 DRUGS</div>
        </div>
    </div>
""", unsafe_allow_html=True)

with st.sidebar.expander(""):
    st.markdown("""
        <div class='guidelines-content'>
        <h4>1. Unfractionated Heparin (UFH)</h4>
        <div class='drug-info'>
        • Route: IV / SC<br>
        • Lowest Dose: 5,000 units bolus (IV) / 5,000 units SC q12h<br>
        • Highest Dose: ~35,000 units/day infusion (IV)<br>
        • Gastric Risk: Low
        </div>

        <h4>2. Low Molecular Weight Heparin (LMWH) (Enoxaparin)</h4>
        <div class='drug-info'>
        • Route: SC<br>
        • Lowest Dose: 20 mg once daily<br>
        • Highest Dose: 1 mg/kg twice daily<br>
        • Gastric Risk: Low to Moderate
        </div>

        <h4>3. Dalteparin</h4>
        <div class='drug-info'>
        • Route: SC<br>
        • Lowest Dose: 2,500 IU once daily<br>
        • Highest Dose: 200 IU/kg once daily<br>
        • Gastric Risk: Low to Moderate
        </div>

        <h4>4. Fondaparinux</h4>
        <div class='drug-info'>
        • Route: SC<br>
        • Lowest Dose: 2.5 mg once daily<br>
        • Highest Dose: 10 mg once daily<br>
        • Gastric Risk: Moderate
        </div>

        <h4>5. Argatroban</h4>
        <div class='drug-info'>
        • Route: IV<br>
        • Lowest Dose: 0.5 mcg/kg/min infusion<br>
        • Highest Dose: 10 mcg/kg/min infusion<br>
        • Gastric Risk: Low
        </div>

        <h4>6. Bivalirudin</h4>
        <div class='drug-info'>
        • Route: IV<br>
        • Lowest Dose: 0.75 mg/kg bolus + infusion<br>
        • Highest Dose: 2.5 mg/kg/hr infusion<br>
        • Gastric Risk: Low
        </div>
        </div>
    """, unsafe_allow_html=True)

# Antiplatelet selection based on the new list
antiplatelet_options = ["None", "Aspirin", "Clopidogrel", "Ticagrelor", "Prasugrel", "Dipyridamole", "Ticlopidine", "Abciximab", "Eptifibatide", "Tirofiban"]
selected_antiplatelet = st.sidebar.selectbox("Select Antiplatelet", antiplatelet_options, key="antiplatelet_select")

# Conditional inputs for Antiplatelet dose and route based on selection
antiplatelet_dose = 0
antiplatelet_route = "None"

if selected_antiplatelet != "None":
    if selected_antiplatelet == "Aspirin":
        antiplatelet_dose = st.sidebar.selectbox("Aspirin Dose (mg)", [0, 75, 150, 300, 325], help="Antiplatelet doses")
        antiplatelet_route = st.sidebar.selectbox("Aspirin Route", ["None", "Oral"])
    elif selected_antiplatelet == "Clopidogrel":
        antiplatelet_dose = st.sidebar.selectbox("Clopidogrel Dose (mg)", [0, 75, 300, 600], help="Maintenance: 75mg, Loading: 300-600mg")
        antiplatelet_route = st.sidebar.selectbox("Clopidogrel Route", ["None", "Oral"])
    elif selected_antiplatelet == "Ticagrelor":
        antiplatelet_dose = st.sidebar.selectbox("Ticagrelor Dose (mg)", [0, 60, 90, 180], help="Maintenance: 60-90mg BID, Loading: 180mg")
        antiplatelet_route = st.sidebar.selectbox("Ticagrelor Route", ["None", "Oral"])
    elif selected_antiplatelet == "Prasugrel":
        antiplatelet_dose = st.sidebar.selectbox("Prasugrel Dose (mg)", [0, 5, 10, 60], help="Maintenance: 5-10mg, Loading: 60mg")
        antiplateplatelet_route = st.sidebar.selectbox("Prasugrel Route", ["None", "Oral"])
    elif selected_antiplatelet == "Dipyridamole":
        antiplatelet_dose = st.sidebar.selectbox("Dipyridamole Dose (mg)", [0, 75, 200], help="75mg BID or 200mg SR BID")
        antiplatelet_route = st.sidebar.selectbox("Dipyridamole Route", ["None", "Oral"])
    elif selected_antiplatelet == "Ticlopidine":
        antiplatelet_dose = st.sidebar.selectbox("Ticlopidine Dose (mg)", [0, 250, 500], help="250mg BID or 500mg daily")
        antiplatelet_route = st.sidebar.selectbox("Ticlopidine Route", ["None", "Oral"])
    elif selected_antiplatelet == "Abciximab":
        antiplatelet_dose = st.sidebar.number_input("Abciximab Dose (mg/kg)", min_value=0.0, step=0.01, format="%.2f", help="Bolus: 0.25 mg/kg, Infusion: 0.125 mcg/kg/min")
        antiplatelet_route = st.sidebar.selectbox("Abciximab Route", ["None", "IV"])
    elif selected_antiplatelet == "Eptifibatide":
        antiplatelet_dose = st.sidebar.number_input("Eptifibatide Dose (mcg/kg)", min_value=0.0, step=0.01, format="%.2f", help="Bolus: 180 mcg/kg, Infusion: 2 mcg/kg/min")
        antiplatelet_route = st.sidebar.selectbox("Eptifibatide Route", ["None", "IV"])
    elif selected_antiplatelet == "Tirofiban":
        antiplatelet_dose = st.sidebar.number_input("Tirofiban Dose (mcg/kg/min)", min_value=0.0, step=0.01, format="%.2f", help="Bolus + Infusion: 0.4 mcg/kg/min, Maintenance: 0.15 mcg/kg/min")
        antiplatelet_route = st.sidebar.selectbox("Tirofiban Route", ["None", "IV"])


# Anticoagulant selection based on the new list
anticoagulant_options = ["None", "Unfractionated Heparin (UFH)", "Low Molecular Weight Heparin (LMWH) (Enoxaparin)", "Dalteparin", "Fondaparinux", "Argatroban", "Bivalirudin"]
selected_anticoagulant = st.sidebar.selectbox("Select Anticoagulant", anticoagulant_options, key="anticoag_select")

# Conditional inputs for Anticoagulant dose and route based on selection
anticoagulant_dose_value = 0 # Using a different variable name to avoid conflict with the selectbox label
anticoagulant_route = "None"
anticoagulant_regimen = "None" # Added for regimens like Prophylactic/Therapeutic

if selected_anticoagulant != "None":
    anticoagulant_route = st.sidebar.selectbox(f"{selected_anticoagulant} Route", ["None", "Oral", "IV", "Subcutaneous"])

    if selected_anticoagulant == "Unfractionated Heparin (UFH)":
        anticoagulant_regimen = st.sidebar.selectbox("UFH Regimen", ["None", "Prophylactic", "Therapeutic"])
        if anticoagulant_regimen == "Prophylactic":
             anticoagulant_dose_value = st.sidebar.selectbox("UFH Dose (units)", [0, 5000], help="5,000 units SC q8-12h")
        elif anticoagulant_regimen == "Therapeutic":
             anticoagulant_dose_value = st.sidebar.number_input("UFH Dose (units/hour infusion)", min_value=0, step=100, help="Typically 10,000 units bolus + 15-25 units/kg/hr infusion")
    elif selected_anticoagulant == "Low Molecular Weight Heparin (LMWH) (Enoxaparin)":
        anticoagulant_regimen = st.sidebar.selectbox("Enoxaparin Regimen", ["None", "Prophylactic", "Therapeutic"])
        if anticoagulant_regimen == "Prophylactic":
            anticoagulant_dose_value = st.sidebar.selectbox("Enoxaparin Dose (mg)", [0, 30, 40], help="30-40 mg SC daily")
        elif anticoagulant_regimen == "Therapeutic":
            anticoagulant_dose_value = st.sidebar.number_input("Enoxaparin Dose (mg/kg)", min_value=0.0, step=0.01, format="%.2f", help="1 mg/kg SC q12h or 1.5 mg/kg SC daily")
    elif selected_anticoagulant == "Dalteparin":
        anticoagulant_regimen = st.sidebar.selectbox("Dalteparin Regimen", ["None", "Prophylactic", "Therapeutic"])
        if anticoagulant_regimen == "Prophylactic":
            anticoagulant_dose_value = st.sidebar.selectbox("Dalteparin Dose (IU)", [0, 2500, 5000], help="2,500-5,000 IU SC daily")
        elif anticoagulant_regimen == "Therapeutic":
            anticoagulant_dose_value = st.sidebar.number_input("Dalteparin Dose (IU/kg)", min_value=0.0, step=0.01, format="%.2f", help="200 IU/kg SC once daily")
    elif selected_anticoagulant == "Fondaparinux":
         anticoagulant_dose_value = st.sidebar.selectbox("Fondaparinux Dose (mg)", [0, 2.5, 5, 7.5, 10], help="2.5mg once daily for prophylaxis, higher for treatment")
    elif selected_anticoagulant == "Argatroban":
         anticoagulant_dose_value = st.sidebar.number_input("Argatroban Dose (mcg/kg/min)", min_value=0.0, step=0.01, format="%.2f", help="Initial: 0.5 mcg/kg/min, Max: 10 mcg/kg/min")
    elif selected_anticoagulant == "Bivalirudin":
         anticoagulant_dose_value = st.sidebar.number_input("Bivalirudin Dose (mg/kg/hr)", min_value=0.0, step=0.01, format="%.2f", help="Bolus: 0.75 mg/kg, Infusion: 2.5 mg/kg/hr")


# --- Scoring functions based on Dose, Route, and Regimen ---
def get_nsaid_score(dose, max_dose, base_risk_score):
    if dose == 0 or dose == "None":
        return 0

    # NSAID scoring remains based on dose and base risk for now
    dose_percentage = (dose / max_dose) * 100
    if dose_percentage <= 25:
        return base_risk_score
    elif dose_percentage <= 50:
        return base_risk_score + 1
    elif dose_percentage <= 75:
        return base_risk_score + 2
    else:
        return base_risk_score + 3

def get_antiplatelet_score(antiplatelet_agent, dose, route):
    score = 0
    if antiplatelet_agent == "None":
        return 0

    # Base score based on drug's inherent risk (can be adjusted based on dose/route)
    base_risk_scores = {
        "Aspirin": 2, # Start with a base score, dose/route will increase it
        "Clopidogrel": 2,
        "Ticagrelor": 2,
        "Prasugrel": 2,
        "Dipyridamole": 1,
        "Ticlopidine": 2,
        "Abciximab": 1,
        "Eptifibatide": 1,
        "Tirofiban": 1
    }
    score += base_risk_scores.get(antiplatelet_agent, 0)

    # Adjust score based on dose and route
    if antiplatelet_agent == "Aspirin":
        if route == "Oral":
            if dose > 150: # Higher doses increase risk
                score += 2
            elif dose > 75: # Moderate doses increase risk
                score += 1
    elif antiplatelet_agent in ["Clopidogrel", "Ticagrelor", "Prasugrel", "Ticlopidine"]:
        if route == "Oral":
            if dose >= 300: # Loading doses or higher maintenance doses
                score += 2
            elif dose >= 75: # Standard maintenance doses
                score += 1
    elif antiplatelet_agent in ["Abciximab", "Eptifibatide", "Tirofiban"]:
         if route == "IV":
             score += 2 # IV antiplatelets generally higher risk

    return score

def get_anticoagulant_score(anticoagulant_agent, dose_value, route, regimen):
    score = 0
    if anticoagulant_agent == "None":
        return 0

    # Base score based on drug's inherent risk (can be adjusted based on dose/route/regimen)
    base_risk_scores = {
        "Unfractionated Heparin (UFH)": 1,
        "Low Molecular Weight Heparin (LMWH) (Enoxaparin)": 1,
        "Dalteparin": 1,
        "Fondaparinux": 2,
        "Argatroban": 2,
        "Bivalirudin": 2
    }
    score += base_risk_scores.get(anticoagulant_agent, 0)

    # Adjust score based on dose, route, and regimen
    if anticoagulant_agent == "Unfractionated Heparin (UFH)":
        if route == "IV":
            if regimen == "Therapeutic":
                score += 3
            elif regimen == "Prophylactic":
                score += 1
        elif route == "SC":
             if regimen == "Prophylactic":
                 score += 1
    elif anticoagulant_agent == "Low Molecular Weight Heparin (LMWH) (Enoxaparin)":
        if route == "Subcutaneous":
            if regimen == "Therapeutic":
                score += 3
            elif regimen == "Prophylactic":
                score += 1
    elif anticoagulant_agent == "Dalteparin":
        if route == "Subcutaneous":
            if regimen == "Therapeutic":
                score += 3
            elif regimen == "Prophylactic":
                score += 1
    elif anticoagulant_agent == "Fondaparinux":
        if route == "Subcutaneous":
            if dose_value >= 5: # Higher doses increase risk
                score += 2
            elif dose_value > 0:
                score += 1
    elif anticoagulant_agent == "Argatroban":
        if route == "IV":
            if dose_value >= 5: # Higher infusion rates
                score += 3
            elif dose_value > 0:
                score += 2
    elif anticoagulant_agent == "Bivalirudin":
        if route == "IV":
            if dose_value >= 1.5: # Higher infusion rates
                score += 3
            elif dose_value > 0:
                score += 2
    # Add scoring for Warfarin, Heparin (generic) if they are added back
    # elif anticoagulant_agent == "Warfarin":
    #     if route == "Oral":
    #         score += 3 # Warfarin generally higher risk than LMWH/UFH for GI bleed

    return score


def get_ppi_gastroprotection(dose, route, nsaid_flag, antiplatelet_flag, anticoagulant_flag):
    reduction = 0
    # PPI reduction logic remains the same for now
    if nsaid_flag or antiplatelet_flag or anticoagulant_flag:
        if route == "Oral" and dose >= 20:
            reduction = -1
        elif route == "IV" and dose >= 40:
            reduction = -2
    return reduction

# Add loading animation CSS
st.markdown("""
<style>
    /* Loading Animation */
    .loading-spinner {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        margin: 20px 0;
    }

    .spinner {
        width: 50px;
        height: 50px;
        border: 5px solid #f3f3f3;
        border-top: 5px solid #155799;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }

    .loading-text {
        margin-top: 15px;
        color: #155799;
        font-weight: 500;
        font-size: 1.1em;
    }

    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    /* Fade In Animation */
    .fade-in {
        animation: fadeIn 0.5s ease-in;
    }

    @keyframes fadeIn {
        0% { opacity: 0; }
        100% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

# Output Section with loading animation
st.subheader("Risk Scoring Result")

# Create a placeholder for the loading animation
loading_placeholder = st.empty()

# Show loading animation
loading_placeholder.markdown("""
    <div class="loading-spinner">
        <div class="spinner"></div>
        <div class="loading-text">Calculating Risk Score...</div>
    </div>
""", unsafe_allow_html=True)

# Simulate processing time (you can remove this in production)
time.sleep(1)  # Simulate calculation time

# Calculate scores
# NSAID score (remains based on dose and base risk)
if selected_nsaid != "None":
    try:
        nsaid_dose_val = int(nsaid_dose) # Use a different variable name
        if nsaid_dose_val > nsaid_max_dose:
            st.sidebar.warning(f"Dose exceeds max recommended for {selected_nsaid}!")
        nsaid_score = get_nsaid_score(nsaid_dose_val, nsaid_max_dose, nsaid_base_risk)
    except ValueError:
        st.sidebar.error("Invalid NSAID dose input.")
        nsaid_score = 0
else:
    nsaid_score = 0

# Antiplatelet score (now based on dose and route)
antiplatelet_score = get_antiplatelet_score(selected_antiplatelet, antiplatelet_dose, antiplatelet_route)

# Anticoagulant score (now based on dose, route, and regimen)
anticoagulant_score = get_anticoagulant_score(selected_anticoagulant, anticoagulant_dose_value, anticoagulant_route, anticoagulant_regimen)


# Interaction Alerts (Review needed based on new drug lists)
# Keeping existing alerts for now, but this section might need updates
interaction_alert = ""
# Updated interaction checks based on selected drugs
if selected_antiplatelet == "Aspirin" and selected_anticoagulant != "None":
    interaction_alert = f"High bleeding risk: Aspirin + {selected_anticoagulant}."
elif selected_antiplatelet in ["Clopidogrel", "Ticagrelor", "Prasugrel", "Ticlopidine"] and selected_anticoagulant != "None":
    interaction_alert = f"Increased bleeding risk: {selected_antiplatelet} + {selected_anticoagulant}."
elif selected_antiplatelet in ["Abciximab", "Eptifibatide", "Tirofiban"] and selected_anticoagulant != "None":
     interaction_alert = f"Monitor closely: {selected_antiplatelet} + {selected_anticoagulant} increases bleeding risk."


if interaction_alert:
    st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #FF6B6B, #FF8E72);
            color: white;
            padding: 16px 20px;
            border-radius: 10px;
            border-left: 6px solid #FF4757;
            margin: 10px 0;
            box-shadow: 0 4px 12px rgba(255, 71, 87, 0.2);
            display: flex;
            align-items: center;
            font-family: 'Arial', sans-serif;
            position: relative;
            overflow: hidden;
        ">
            <div style="
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: linear-gradient(45deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0) 100%);
                pointer-events: none;
            "></div>
            <span style="
                font-size: 24px;
                margin-right: 12px;
                color: #FFE0E0;
            ">⚠️</span>
            <div>
                <div style="
                    font-weight: 600;
                    font-size: 1.1em;
                    margin-bottom: 4px;
                    color: #FFFFFF;
                    text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
                ">Drug–Drug Interaction Alert</div>
                <div style="
                    color: #FFE0E0;
                    font-size: 0.95em;
                ">{interaction_alert}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# Calculate indication score
indication_weights = {
    "Non-variceal bleeding": 3, "Dyspepsia": 1, "GERD & complications": 2, "H pylori infection": 2, "Peptic ulcer treatment": 3,
    "Zollinger-Ellison syndrome": 3, "Prevent NSAID ulcers": 2, "NSAID & ulcer/GIB history": 3, "NSAID & age > 60": 2,
    "NSAID + cortico/antiplatelet/anticoag": 3, "Prophylaxis in high risk antiplatelet users": 2, "Antiplatelet & ulcer/GIB history": 3,
    "Antiplatelet + age > 60 or dyspepsia/GERD": 2, "Antiplatelet + cortico/NSAID/anticoag": 3, "Stress ulcer prophylaxis": 2,
    "Coagulopathy (platelet < 50k, INR ≥ 1.5)": 2, "Mechanical ventilation > 48h": 2,
}
indication_score = sum([indication_weights.get(ind, 0) for ind in selected_indications])

# Calculate risk flags
nsaid_flag = int(selected_nsaid != "None")
antiplatelet_flag = int(selected_antiplatelet != "None")
anticoagulant_flag = int(selected_anticoagulant != "None")
triple_combo_flag = int(nsaid_flag and antiplatelet_flag and anticoagulant_flag)
medication_risk = nsaid_score + antiplatelet_score + anticoagulant_score
high_risk_flag = int(medication_risk >= 6 or indication_score >= 6) # Threshold might need adjustment with new scoring

# Calculate PPI reduction
ppi_reduction = get_ppi_gastroprotection(ppi_dose, ppi_route, nsaid_flag, antiplatelet_flag, anticoagulant_flag)

# Calculate final score
score = medication_risk + indication_score + (triple_combo_flag * 2) + high_risk_flag + ppi_reduction

# Clear loading animation and show results with fade-in effect
loading_placeholder.empty()

# Risk interpretation
# Enhanced PPI-specific recommendations
def get_ppi_recommendations(current_ppi, current_dose, current_route, risk_score):
    recommendations = []

    # Create DataFrames for all recommendation types
    if current_route == "IV":
        route_opt_data = {
            "Parameter": ["Switch Dose", "Pre-switch Check", "Monitoring"],
            "Pantoprazole": ["Oral 40mg once daily", "Ensure oral intake capability", "Monitor for 48-72h post-switch"],
            "Esomeprazole": ["Oral 40mg once daily", "Consider step-down to 20mg", "Monitor acid suppression"],
            "Omeprazole": ["Oral 20mg once daily", "Continue as maintenance", "Monitor symptoms"],
            "Rabeprazole": ["Oral 20mg once daily", "Consider on-demand if low risk", "Monitor response"]
        }
        st.markdown("🔄 **Route Optimization Required:**")
        df_route = pd.DataFrame(route_opt_data).set_index("Parameter")
        st.table(df_route.style.apply(lambda x: ['background-color: #e6f3ff' if col == current_ppi else '' for col in df_route.columns], axis=1))

    if current_dose >= 40:
        high_dose_data = {
            "Parameter": ["Maximum Duration", "Step Down", "Additional Notes"],
            "Pantoprazole": ["80mg/day: 8 weeks", "40mg once daily", "Take 30-60 minutes before first meal"],
            "Esomeprazole": ["40mg/day: 8 weeks", "20mg once daily", "Based on symptoms"],
            "Omeprazole": ["40mg/day: 8 weeks", "20mg once daily", "Consider on-demand"],
            "Rabeprazole": ["40mg/day: 8 weeks", "20mg once daily", "Consider discontinuation"]
        }
        st.markdown("\n📉 **High Dose Management:**")
        df_high_dose = pd.DataFrame(high_dose_data).set_index("Parameter")
        st.table(df_high_dose.style.apply(lambda x: ['background-color: #e6f3ff' if col == current_ppi else '' for col in df_high_dose.columns], axis=1))

    # Dose Reduction Strategy for all PPIs
    if current_dose > 40:
        dose_reduction_data = {
            "Strategy": ["Maintenance Dose", "Minimum Dose", "Deprescribing Option"],
            "Pantoprazole": ["40mg once daily", "40mg", "Alternate day 40mg"],
            "Esomeprazole": ["20mg once daily", "20mg", "Alternate day 20mg"],
            "Omeprazole": ["20mg once daily", "20mg", "On-demand therapy"],
            "Rabeprazole": ["20mg once daily", "20mg", "Consider discontinuation"]
        }
        st.markdown("\n📉 **Dose Reduction Strategy:**")
        df_reduction = pd.DataFrame(dose_reduction_data).set_index("Strategy")
        st.table(df_reduction.style.apply(lambda x: ['background-color: #e6f3ff' if col == current_ppi else '' for col in df_reduction.columns], axis=1))

    # Monitoring and Follow-up combined table
    if risk_score >= 7:
        # Check if there's a drug interaction (updated to check against the new drug lists)
        has_interaction = False
        if selected_antiplatelet == "Aspirin" and selected_anticoagulant != "None":
             has_interaction = True
        elif selected_antiplatelet in ["Clopidogrel", "Ticagrelor", "Prasugrel", "Ticlopidine"] and selected_anticoagulant != "None":
             has_interaction = True
        elif selected_antiplatelet in ["Abciximab", "Eptifibatide", "Tirofiban"] and selected_anticoagulant != "None":
             has_interaction = True


        monitoring_data = {
            "Timeframe": [
                "Week 1-4",
                "Ongoing",
                "Monthly",
                "Follow-up",
                "Additional"
            ],
            "Action": [
                "Weekly symptom assessment",
                "Monitor GI bleeding signs",
                "Review drug interactions" if not has_interaction else "🔴 Review drug interactions (Drug-Drug Interaction Detected)",
                "Reassess in 4-6 weeks",
                "Monitor breakthrough symptoms"
            ],
            "Priority": [
                "High",
                "High",
                "High" if has_interaction else "Medium",
                "Medium",
                "Medium"
            ]
        }
        st.markdown("\n📋 **Monitoring and Follow-up Plan:**")
        df_monitoring = pd.DataFrame(monitoring_data).set_index("Timeframe")

        # Apply conditional styling
        def highlight_priority(val):
            if val == "High":
                return 'background-color: #ffebee; color: #d32f2f'
            return ''

        st.table(df_monitoring.style.applymap(highlight_priority, subset=['Priority']))

    return []  # Empty list as we're using direct st.table() display

# Determine duration text based on new inputs
if previous_ppi_history and ppi_duration_days > (8 * 30): # Long-term if history > ~8 weeks
    duration_text = f"History of Long-term Oral PPI Therapy ({ppi_duration_days} days)"
elif previous_ppi_history and ppi_duration_days > 0:
     duration_text = f"History of Short-term Oral PPI Therapy ({ppi_duration_days} days)"
else:
    duration_text = "No History of Previous PPI Therapy"


if score >= 10:
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #8B0000, #DC143C);
        padding: 25px 30px;
        border-radius: 15px;
        box-shadow: 0 8px 20px rgba(139, 0, 0, 0.4);
        margin: 15px 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    ">
        <div style="
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(45deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0) 100%);
            pointer-events: none;
        "></div>
        <h3 style="
            color: white;
            margin: 0 0 10px 0;
            font-family: 'Arial', sans-serif;
            font-size: 1.4em;
            font-weight: 600;
            letter-spacing: 0.5px;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
            position: relative;
            z-index: 1;
        ">🔴 Very High Risk – Continue Current PPI Therapy</h3>
        <p style="
            color: white;
            margin: 0;
            font-size: 1.1em;
            opacity: 0.9;
        ">History: {duration_text}</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div style="
        background: rgba(255, 255, 255, 0.95);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
    ">
        <h4 style="color: #333; margin-bottom: 15px;">Clinical Recommendations</h4>
    """, unsafe_allow_html=True)
    for rec in get_ppi_recommendations(selected_ppi, ppi_dose, ppi_route, score):
        st.write(rec)
    st.markdown("""
    <div style="
        background: rgba(255, 255, 255, 0.95);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
    ">
        <h4 style="color: #333; margin-bottom: 15px;">Additional Measures</h4>
        <ul style="
            list-style-type: none;
            padding-left: 0;
            margin: 0;
        ">
            <li style="margin-bottom: 8px;">• Review risk factors every 3 months</li>
            <li style="margin-bottom: 8px;">• Consider GI specialist consultation</li>
            <li style="margin-bottom: 8px;">• Monitor for long-term PPI complications</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

elif score >= 7:
    duration_text = f"History: {duration_text}" # Use the determined duration text
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #B22222, #FF4500);
        padding: 25px 30px;
        border-radius: 15px;
        box-shadow: 0 8px 20px rgba(178, 34, 34, 0.4);
        margin: 15px 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    ">
        <div style="
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(45deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0) 100%);
            pointer-events: none;
        "></div>
        <h3 style="
            color: white;
            margin: 0 0 10px 0;
            font-family: 'Arial', sans-serif;
            font-size: 1.4em;
            font-weight: 600;
            letter-spacing: 0.5px;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
            position: relative;
            z-index: 1;
        ">🟠 High Risk – Optimize PPI Therapy</h3>
        <p style="
            color: white;
            margin: 0;
            font-size: 1.1em;
            opacity: 0.9;
        ">{duration_text}</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div style="
        background: rgba(255, 255, 255, 0.95);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
    ">
        <h4 style="color: #333; margin-bottom: 15px;">Clinical Recommendations</h4>
    """, unsafe_allow_html=True)
    for rec in get_ppi_recommendations(selected_ppi, ppi_dose, ppi_route, score):
        st.write(rec)
    st.markdown("""
    <div style="
        background: rgba(255, 255, 255, 0.95);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
    ">
        <h4 style="color: #333; margin-bottom: 15px;">Follow-up Plan</h4>
        <ul style="
            list-style-type: none;
            padding-left: 0;
            margin: 0;
        ">
            <li style="margin-bottom: 8px;">• Reassess in 4-6 weeks</li>
            <li style="margin-bottom: 8px;">• Monitor for breakthrough symptoms</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

elif 4 <= score < 7:
    duration_text = f"History: {duration_text}" # Use the determined duration text
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #1E90FF, #4169E1);
        padding: 25px 30px;
        border-radius: 15px;
        box-shadow: 0 8px 20px rgba(30, 144, 255, 0.4);
        margin: 15px 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    ">
        <div style="
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(45deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0) 100%);
            pointer-events: none;
        "></div>
        <h3 style="
            color: white;
            margin: 0 0 10px 0;
            font-family: 'Arial', sans-serif;
            font-size: 1.4em;
            font-weight: 600;
            letter-spacing: 0.5px;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
            position: relative;
            z-index: 1;
        ">🟡 Moderate Risk – Consider Step-down Therapy</h3>
        <p style="
            color: white;
            margin: 0;
            font-size: 1.1em;
            opacity: 0.9;
        ">{duration_text}</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div style="
        background: rgba(255, 255, 255, 0.95);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
    ">
        <h4 style="color: #333; margin-bottom: 15px;">Clinical Recommendations</h4>
        <ul style="
            list-style-type: none;
            padding-left: 0;
            margin: 0;
        ">
            <li style="margin-bottom: 8px;">• Consider gradual dose reduction</li>
            <li style="margin-bottom: 8px;">• Implement step-down protocol</li>
            <li style="margin-bottom: 8px;">• Monitor for symptom recurrence</li>
            <li style="margin-bottom: 8px;">• Schedule follow-up in 4 weeks</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

else:
    duration_text = f"History: {duration_text}" # Use the determined duration text
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #228B22, #32CD32);
        padding: 25px 30px;
        border-radius: 15px;
        box-shadow: 0 8px 20px rgba(34, 139, 34, 0.4);
        margin: 15px 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    ">
        <div style="
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(45deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0) 100%);
            pointer-events: none;
        "></div>
        <h3 style="
            color: white;
            margin: 0 0 10px 0;
            font-family: 'Arial', sans-serif;
            font-size: 1.4em;
            font-weight: 600;
            letter-spacing: 0.5px;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
            position: relative;
            z-index: 1;
        ">🟢 Low Risk Assessment – PPI Deprescribing Protocol Initiation</h3>
        <p style="
            color: white;
            margin: 0;
            font-size: 1.1em;
            opacity: 0.9;
        ">{duration_text}</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div style="
        background: rgba(255, 255, 255, 0.95);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
    ">
        <h4 style="color: #333; margin-bottom: 15px;">Clinical Management Protocol</h4>
        <ul style="
            list-style-type: none;
            padding-left: 0;
            margin: 0;
        ">
            <li style="margin-bottom: 12px; line-height: 1.5;">
                <strong>1. Deprescribing Initiation:</strong><br>
                • Implement gradual dose reduction protocol<br>
                • Consider step-down to on-demand therapy
            </li>
            <li style="margin-bottom: 12px; line-height: 1.5;">
                <strong>2. Monitoring Parameters:</strong><br>
                • Assess for symptom recurrence<br>
                • Monitor acid-related symptoms<br>
                • Evaluate quality of life indicators
            </li>
            <li style="margin-bottom: 12px; line-height: 1.5;">
                <strong>3. Follow-up Schedule:</strong><br>
                • Initial review: 2 weeks post-initiation<br>
                • Comprehensive assessment: 8 weeks<br>
                • Long-term monitoring: As clinically indicated
            </li>
            <li style="margin-bottom: 12px; line-height: 1.5;">
                <strong>4. Patient Education:</strong><br>
                • Provide written deprescribing plan<br>
                • Discuss symptom recognition<br>
                • Review lifestyle modifications
            </li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div style="
        background: rgba(255, 255, 255, 0.95);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
    ">
        <h4 style="color: #333; margin-bottom: 15px;">Clinical Considerations</h4>
        <ul style="
            list-style-type: none;
            padding-left: 0;
            margin: 0;
        ">
            <li style="margin-bottom: 8px;">• Document baseline symptom assessment</li>
            <li style="margin-bottom: 8px;">• Review concurrent medications</li>
            <li style="margin-bottom: 8px;">• Consider patient preferences and adherence</li>
            <li style="margin-bottom: 8px;">• Assess for contraindications to deprescribing</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# --- Flowchart Generation ---
st.subheader("PPI Deprescribing Flowchart")
with st.expander("📊 View Detailed Deprescribing Flowchart"):
    def generate_ppi_flowchart(score, nsaid_score, antiplatelet_score, anticoagulant_score, indication_score, triple_combo_flag, high_risk_flag, ppi_reduction, graph_size="12,12"):
        try:
            dot = graphviz.Digraph(comment='PPI Deprescribing Algorithm',
                                  graph_attr={'rankdir': 'TB',
                                            'size': graph_size,
                                            'splines': 'ortho',
                                            'nodesep': '0.8',
                                            'ranksep': '0.8',
                                            'fontname': 'Arial',
                                            'bgcolor': 'transparent',
                                            'pad': '0.5',
                                            'margin': '0.5'})

            # Style definitions with enhanced colors and formatting
            dot.attr('node', shape='box', style='rounded,filled', fontname='Arial', margin='0.3')
            dot.attr('edge', fontname='Arial', fontsize='11', penwidth='1.5')

            # Initial Assessment with better formatting
            dot.node('start', 'Initial PPI Assessment\n(Current Therapy)',
                    fillcolor='#d7e1ec',
                    style='rounded,filled,radial',
                    gradientangle='45',
                    penwidth='2',
                    color='#155799')

            # Risk Score node with detailed breakdown
            risk_details = f'Risk Score Calculation\n───────────────\nTotal Score: {score:.2f}\nMedication Risk: {nsaid_score + antiplatelet_score + anticoagulant_score}\nIndication Score: {indication_score}' # Added formatting for score
            dot.node('risk_calc', risk_details,
                    shape='diamond',
                    fillcolor='#159957',
                    style='filled,radial',
                    gradientangle='45',
                    penwidth='2',
                    color='#155799')

            # Risk Components subgraph with improved layout
            with dot.subgraph(name='cluster_0') as c:
                c.attr(label='Risk Components',
                      style='rounded,filled',
                      fillcolor='transparent',
                      color='#155799',
                      fontname='Arial',
                      fontsize='12',
                      penwidth='2')
                c.node('med_risk', f'Medication Risk\n───────────\nNSAID: {nsaid_score}\nAntiplatelet: {antiplatelet_score}\nAnticoagulant: {anticoagulant_score}',
                      fillcolor='#d7e1ec',
                      style='filled,radial',
                      gradientangle='45',
                      penwidth='2',
                      color='#155799')
                c.node('comb_risk', f'Combination Risk\n───────────\nTriple Therapy: {triple_combo_flag}\nHigh Risk Flag: {high_risk_flag}',
                      fillcolor='#d7e1ec',
                      style='filled,radial',
                      gradientangle='45',
                      penwidth='2',
                      color='#155799')
                c.node('ppi_effect', f'PPI Protection\n───────────\nReduction: {ppi_reduction}',
                      fillcolor='#d7e1ec',
                      style='filled,radial',
                      gradientangle='45',
                      penwidth='2',
                      color='#155799')

            # Enhanced Decision Pathways with clearer color coding
            if score < 4: # Adjusted threshold to align with score calculation and noise
                dot.node('decision', 'Low Risk\nConsider Deprescribing',
                        fillcolor='#159957',
                        style='filled,radial',
                        gradientangle='45',
                        penwidth='2',
                        color='#155799')
                dot.node('action', '• Gradual dose reduction\n• Switch to PRN dosing\n• Monitor symptoms\n• Review in 4-8 weeks',
                        fillcolor='#d7e1ec',
                        style='filled,radial',
                        gradientangle='45',
                        penwidth='2',
                        color='#155799')
            elif 4 <= score < 7: # Adjusted threshold
                dot.node('decision', 'Moderate Risk\nStep-down Therapy',
                        fillcolor='#155799',
                        style='filled,radial',
                        gradientangle='45',
                        penwidth='2',
                        color='#159957')
                dot.node('action', '• Reduce to minimum effective dose\n• Consider alternate day dosing\n• Weekly monitoring\n• Review in 4 weeks',
                        fillcolor='#d7e1ec',
                        style='filled,radial',
                        gradientangle='45',
                        penwidth='2',
                        color='#155799')
            elif 7 <= score < 10: # Adjusted threshold
                dot.node('decision', 'High Risk\nOptimize Therapy',
                        fillcolor='#ff8c42',
                        style='filled,radial',
                        gradientangle='45',
                        penwidth='2',
                        color='#ff6f00')
                dot.node('action', '• Maintain current dose\n• Regular monitoring\n• Review drug interactions\n• Reassess in 6-8 weeks',
                        fillcolor='#d7e1ec',
                        style='filled,radial',
                        gradientangle='45',
                        penwidth='2',
                        color='#155799')
            else: # score >= 10
                dot.node('decision', 'Very High Risk\nContinue Therapy',
                        fillcolor='#ff6f00',
                        style='filled,radial',
                        gradientangle='45',
                        penwidth='2',
                        color='#ff8c42')
                dot.node('action', '• Continue current regimen\n• Monthly monitoring\n• Specialist consultation\n• Review in 3 months',
                        fillcolor='#d7e1ec',
                        style='filled,radial',
                        gradientangle='45',
                        penwidth='2',
                        color='#155799')

            # Enhanced edge connections with better labeling
            dot.edge('start', 'risk_calc', 'Initial Assessment',
                    color='#155799',
                    penwidth='2')
            dot.edge('risk_calc', 'med_risk', 'Risk Analysis',
                    color='#155799',
                    penwidth='2')
            dot.edge('risk_calc', 'comb_risk', 'Risk Factors',
                    color='#155799',
                    penwidth='2')
            dot.edge('risk_calc', 'ppi_effect', 'Protection Assessment',
                    color='#155799',
                    penwidth='2')
            dot.edge('med_risk', 'decision', 'Risk Level',
                    color='#155799',
                    penwidth='2')
            dot.edge('comb_risk', 'decision', 'Combined Risk',
                    color='#155799',
                    penwidth='2')
            dot.edge('ppi_effect', 'decision', 'Protection Level',
                    color='#155799',
                    penwidth='2')
            dot.edge('decision', 'action', 'Management Plan',
                    color='#155799',
                    penwidth='2')

            return dot
        except Exception as e:
            st.error(f"Error generating flowchart: {e}")
            return None

    # Generate and display the flowchart
    flowchart = generate_ppi_flowchart(
        score=score,
        nsaid_score=nsaid_score,
        antiplatelet_score=antiplatelet_score,
        anticoagulant_score=anticoagulant_score,
        indication_score=indication_score,
        triple_combo_flag=triple_combo_flag,
        high_risk_flag=high_risk_flag,
        ppi_reduction=ppi_reduction
    )

    if flowchart:
        # Center the flowchart
        st.markdown("""
        <div style="display: flex; justify-content: center; align-items: center; width: 100%;">
            <div style="max-width: 800px; width: 100%;">
        """, unsafe_allow_html=True)
        st.graphviz_chart(flowchart, use_container_width=True)
        st.markdown("</div></div>", unsafe_allow_html=True)

        # Add professional legend
        st.markdown("""
        <div style="
            background: rgba(255, 255, 255, 0.95);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            margin: 20px auto;
            max-width: 800px;
            width: 100%;
        ">
            <h4 style="color: #155799; margin-bottom: 15px; font-family: 'Arial', sans-serif; text-align: center;">Flowchart Legend</h4>
            <div style="
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 15px;
                max-width: 700px;
                margin: 0 auto;
            ">
                <div style="
                    background: linear-gradient(135deg, #155799, #159957);
                    padding: 10px;
                    border-radius: 5px;
                    color: white;
                ">
                    <strong>Risk Assessment</strong>
                    <ul style="list-style-type: none; padding-left: 0; margin: 5px 0;">
                        <li>• Initial PPI Assessment</li>
                        <li>• Risk Score Calculation</li>
                        <li>• Risk Components Analysis</li>
                    </ul>
                </div>
                <div style="
                    background: linear-gradient(135deg, #ff8c42, #ff6f00);
                    padding: 10px;
                    border-radius: 5px;
                    color: white;
                ">
                    <strong>Risk Levels</strong>
                    <ul style="list-style-type: none; padding-left: 0; margin: 5px 0;">
                        <li>• Very High Risk (Score ≥ 10)</li>
                        <li>• High Risk (Score 7-9)</li>
                        <li>• Moderate Risk (Score 4-6)</li>
                        <li>• Low Risk (Score ≤ 3)</li>
                    </ul>
                </div>
                <div style="
                    background: linear-gradient(135deg, #d7e1ec, #155799);
                    padding: 10px;
                    border-radius: 5px;
                    color: white;
                ">
                    <strong>Action Nodes</strong>
                    <ul style="list-style-type: none; padding-left: 0; margin: 5px 0;">
                        <li>• Clinical Recommendations</li>
                        <li>• Management Plans</li>
                        <li>• Follow-up Protocols</li>
                    </ul>
                </div>
                <div style="
                    background: linear-gradient(135deg, #159957, #155799);
                    padding: 10px;
                    border-radius: 5px;
                    color: white;
                ">
                    <strong>Connections</strong>
                    <ul style="list-style-type: none; padding-left: 0; margin: 5px 0;">
                        <li>• Risk Analysis Flow</li>
                        <li>• Decision Pathways</li>
                        <li>• Management Steps</li>
                    </ul>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("Failed to generate flowchart")

# Risk Analysis Visualization
st.markdown("---")
st.subheader("Risk Analysis Visualization")

# Create data for the risk components bar chart
risk_components = {
    "NSAID Risk": nsaid_score,
    "Antiplatelet Risk": antiplatelet_score,
    "Anticoagulant Risk": anticoagulant_score,
    "Clinical Indications": indication_score,
    "Combination Risk": triple_combo_flag * 2,
    "High Risk Flag": high_risk_flag,
    "PPI Protection": ppi_reduction  # Remove abs() to show negative value
}

# Create bar chart
fig_bar = go.Figure()

# Add bars for each component
components = list(risk_components.keys())
values = list(risk_components.values())

# Define colors with blue for positive values and a different color for negative values
colors = ['#155799', '#159957', '#1B263B', '#415A77', '#778DA9', '#E0E1DD', '#4CAF50']  # Changed last color to green for PPI Protection

fig_bar.add_trace(go.Bar(
    x=components,
    y=values,
    marker_color=colors,
    text=values,
    textposition='auto',
))

# Update layout with professional styling
fig_bar.update_layout(
    title={
        'text': f"Risk Component Analysis (Total Score: {score:.2f})", # Added formatting for score
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(size=20, color='#1B263B')
    },
    xaxis_title="Risk Components",
    yaxis_title="Risk Score",
    height=500,
    xaxis_tickangle=-45,
    showlegend=False,
    margin=dict(t=80, l=50, r=50, b=100),
    plot_bgcolor='rgba(255,255,255,0.9)',
    paper_bgcolor='rgba(255,255,255,0)',
    xaxis=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(211,211,211,0.3)',
        linecolor='rgba(211,211,211,0.3)'
    ),
    yaxis=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(211,211,211,0.3)',
        linecolor='rgba(211,211,211,0.3)',
        zeroline=True,  # Show the zero line
        zerolinewidth=2,  # Make zero line more visible
        zerolinecolor='rgba(0,0,0,0.2)'  # Style zero line
    )
)

# Add hover template with sign-aware formatting
fig_bar.update_traces(
    hovertemplate="<b>%{x}</b><br>" +
                  "Score: %{y:+.1f}<br>" +  # Added + sign for positive values
                  "<extra></extra>"
)

st.plotly_chart(fig_bar, use_container_width=True)

# Update the risk analysis explanation
st.markdown("""
<div style="
    background: linear-gradient(135deg, #f8f9fa, #e9ecef);
    padding: 20px;
    border-radius: 10px;
    margin-top: 20px;
    margin-bottom: 30px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1)
">
    <h4 style="color: #155799; margin-bottom: 15px;">📊 Risk Analysis Interpretation</h4>
    <p style="color: #333; line-height: 1.6;">
        This chart breaks down the contribution of each factor to your total risk score.
        Higher bars indicate greater risk contribution from that component.
        The PPI Protection factor appears as a negative value (below the axis) as it reduces the overall risk score.
    </p>
</div>
""", unsafe_allow_html=True)

# --- Log Entry and CSV Download ---
if 'logged_data' not in st.session_state:
    st.session_state.logged_data = []

if st.button("Log Input Data"):
    input_data = {
        "PPI": selected_ppi, "PPI Dose": ppi_dose, "PPI Route": ppi_route,
        "NSAID": selected_nsaid, "NSAID Dose": nsaid_dose, "NSAID Route": nsaid_route,
        "Antiplatelet": selected_antiplatelet, "Antiplatelet Dose": antiplatelet_dose, "Antiplatelet Route": antiplatelet_route, # Include dose/route for AP
        "Anticoagulant": selected_anticoagulant, "Anticoagulant Dose": anticoagulant_dose_value, "Anticoagulant Route": anticoagulant_route, "Anticoagulant Regimen": anticoagulant_regimen, # Include dose/route/regimen for AC
        "Previous PPI History": previous_ppi_history, "Oral PPI Duration (days)": ppi_duration_days, # Log new history inputs
        "Indications": ", ".join(selected_indications), "Score": score
    }
    st.session_state.logged_data.append(input_data)
    st.success("Data logged successfully!")

if st.session_state.logged_data:
    df_logged = pd.DataFrame(st.session_state.logged_data)
    csv_buffer = io.StringIO()
    df_logged.to_csv(csv_buffer, index=False)
    csv_string = csv_buffer.getvalue()
    csv_bytes = csv_string.encode()
    st.download_button(label="Download Logged Data as CSV", data=csv_bytes, file_name="logged_data.csv", mime="text/csv")

# --- ML Model Training and Evaluation ---
@st.cache_data
def generate_synthetic_data(num_samples=1000):
    data = []
    # Define possible dose/route/regimen options for synthetic data generation
    antiplatelet_options_synth = {
        "None": {"dose": [0], "route": ["None"]},
        "Aspirin": {"dose": [75, 150, 300, 325], "route": ["Oral"]},
        "Clopidogrel": {"dose": [75, 300, 600], "route": ["Oral"]},
        "Ticagrelor": {"dose": [60, 90, 180], "route": ["Oral"]},
        "Prasugrel": {"dose": [5, 10, 60], "route": ["Oral"]},
        "Dipyridamole": {"dose": [75, 200], "route": ["Oral"]},
        "Ticlopidine": {"dose": [250, 500], "route": ["Oral"]},
        "Abciximab": {"dose": [0.25, 0.125], "route": ["IV"]}, # Simplified dose representation
        "Eptifibatide": {"dose": [180, 2], "route": ["IV"]}, # Simplified dose representation
        "Tirofiban": {"dose": [0.4, 0.15], "route": ["IV"]} # Simplified dose representation
    }
    anticoagulant_options_synth = {
        "None": {"dose": [0], "route": ["None"], "regimen": ["None"]},
        "Unfractionated Heparin (UFH)": {"dose": [5000, 20000], "route": ["IV", "Subcutaneous"], "regimen": ["Prophylactic", "Therapeutic"]}, # Simplified dose
        "Low Molecular Weight Heparin (LMWH) (Enoxaparin)": {"dose": [30, 40, 80], "route": ["Subcutaneous"], "regimen": ["Prophylactic", "Therapeutic"]}, # Simplified dose
        "Dalteparin": {"dose": [2500, 5000, 10000], "route": ["Subcutaneous"], "regimen": ["Prophylactic", "Therapeutic"]}, # Simplified dose
        "Fondaparinux": {"dose": [2.5, 5, 7.5, 10], "route": ["Subcutaneous"], "regimen": ["None"]},
        "Argatroban": {"dose": [0.5, 5, 10], "route": ["IV"], "regimen": ["None"]}, # Simplified dose
        "Bivalirudin": {"dose": [0.75, 1.5, 2.5], "route": ["IV"], "regimen": ["None"]} # Simplified dose
    }


    for _ in range(num_samples):
        ppi = random.choice(["None", "Pantoprazole", "Omeprazole", "Esomeprazole", "Rabeprazole"])
        ppi_dose = random.choice([0, 20, 40, 80])
        ppi_route = random.choice(["None", "Oral", "IV"])

        nsaid_group = random.choice(list(nsaid_groups.keys()))
        nsaid = random.choice(list(nsaid_groups[nsaid_group].keys()))
        nsaid_info = nsaid_groups[nsaid_group][nsaid]
        nsaid_dose_val = random.choice(nsaid_info[0])
        nsaid_route = random.choice(["None", "Oral", "Parenteral"])

        selected_antiplatelet_synth = random.choice(list(antiplatelet_options_synth.keys()))
        antiplatelet_dose_synth = random.choice(antiplatelet_options_synth[selected_antiplatelet_synth]["dose"])
        antiplatelet_route_synth = random.choice(antiplatelet_options_synth[selected_antiplatelet_synth]["route"])

        selected_anticoagulant_synth = random.choice(list(anticoagulant_options_synth.keys()))
        anticoagulant_dose_synth = random.choice(anticoagulant_options_synth[selected_anticoagulant_synth]["dose"])
        anticoagulant_route_synth = random.choice(anticoagulant_options_synth[selected_anticoagulant_synth]["route"])
        anticoagulant_regimen_synth = random.choice(anticoagulant_options_synth[selected_anticoagulant_synth]["regimen"])

        # Generate synthetic history data
        previous_ppi_history_synth = random.choice([True, False])
        ppi_duration_days_synth = random.randint(0, 3650) if previous_ppi_history_synth else 0


        num_indications = random.randint(0, 5)
        # Corrected merging of dictionaries
        all_indications = {**gi_indications, **nsaid_ap_indications, **other_indications}
        indications = random.sample(list(all_indications.keys()), num_indications)

        # Calculate synthetic scores based on new logic
        nsaid_score = get_nsaid_score(nsaid_dose_val, nsaid_info[2], nsaid_info[3]) if nsaid != "None" else 0
        antiplatelet_score = get_antiplatelet_score(selected_antiplatelet_synth, antiplatelet_dose_synth, antiplatelet_route_synth)
        anticoagulant_score = get_anticoagulant_score(selected_anticoagulant_synth, anticoagulant_dose_synth, anticoagulant_route_synth, anticoagulant_regimen_synth)


        nsaid_flag = int(nsaid != "None")
        antiplatelet_flag = int(selected_antiplatelet_synth != "None")
        anticoagulant_flag = int(selected_anticoagulant_synth != "None")
        triple_combo_flag = int(nsaid_flag and antiplatelet_flag and anticoagulant_flag)

        indication_score = sum(indication_weights.get(ind, 0) for ind in indications)
        # Corrected typo here: antiplateplatelet_score -> antiplatelet_score
        medication_risk = nsaid_score + antiplatelet_score + anticoagulant_score
        high_risk_flag = int(medication_risk >= 6 or indication_score >= 6) # Threshold might need adjustment with new scoring

        ppi_reduction = get_ppi_gastroprotection(ppi_dose, ppi_route, nsaid_flag, antiplatelet_flag, anticoagulant_flag)

        score = medication_risk + indication_score + (triple_combo_flag * 2) + high_risk_flag + ppi_reduction

        # Add noise to the score
        noise = random.uniform(-1.0, 1.0) # Add random noise between -1 and +1
        score = max(0, score + noise) # Ensure score remains non-negative

        data.append({
            "PPI": ppi, "PPI Dose": ppi_dose, "PPI Route": ppi_route,
            "NSAID": nsaid, "NSAID Dose": nsaid_dose_val, "NSAID Route": nsaid_route,
            "Antiplatelet": selected_antiplatelet_synth, "Antiplatelet Dose": antiplatelet_dose_synth, "Antiplatelet Route": antiplatelet_route_synth,
            "Anticoagulant": selected_anticoagulant_synth, "Anticoagulant Dose": anticoagulant_dose_synth, "Anticoagulant Route": anticoagulant_route_synth, "Anticoagulant Regimen": anticoagulant_regimen_synth,
            "Previous PPI History": previous_ppi_history_synth, "Oral PPI Duration (days)": ppi_duration_days_synth, # Include new history fields
            "Indications": ", ".join(indications), "Score": score
        })
    return pd.DataFrame(data)


@st.cache_data
def train_and_evaluate_models(data):
    X = data.drop("Score", axis=1)
    y = (data["Score"] >= 7).astype(int)  # Binary classification: High Risk (1) or Lower Risk (0) - Adjusted threshold based on potential new score range
    X = pd.get_dummies(X)  # One hot encode categorical data

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    rf_model = RandomForestClassifier(random_state=42)
    lr_model = LogisticRegression(random_state=42, max_iter=1000)

    rf_model.fit(X_train, y_train)
    lr_model.fit(X_train, y_train)

    rf_probs = rf_model.predict_proba(X_test)[:, 1]
    lr_probs = lr_model.predict_proba(X_test)[:, 1]

    rf_predictions = rf_model.predict(X_test)
    lr_predictions = lr_model.predict(X_test)

    rf_metrics = {
        "Accuracy": accuracy_score(y_test, rf_predictions),
        "Precision": precision_score(y_test, rf_predictions),
        "Recall": recall_score(y_test, rf_predictions),
        "F1 Score": f1_score(y_test, rf_predictions),
        "AUC": roc_auc_score(y_test, rf_probs),
    }

    lr_metrics = {
        "Accuracy": accuracy_score(y_test, lr_predictions),
        "Precision": precision_score(y_test, lr_predictions),
        "Recall": recall_score(y_test, lr_predictions),
        "F1 Score": f1_score(y_test, lr_predictions),
        "AUC": roc_auc_score(y_test, lr_probs),
    }

    rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)

    return rf_fpr, rf_tpr, lr_fpr, lr_tpr, rf_metrics, lr_metrics

# Train and evaluate models
synthetic_data = generate_synthetic_data()
rf_fpr, rf_tpr, lr_fpr, lr_tpr, rf_metrics, lr_metrics = train_and_evaluate_models(synthetic_data)

# Display model evaluation results
st.subheader("Machine Learning Model Evaluation")
col1, col2 = st.columns(2)

with col1:
    fig_rf, ax_rf = plt.subplots(figsize=(6, 5), facecolor='#f4f6f8')
    ax_rf.set_facecolor('#f4f6f8')

    # Plot ROC curve with enhanced styling
    ax_rf.plot(rf_fpr, rf_tpr, linewidth=2.5, color='#155799', alpha=0.8, label='Random Forest')
    ax_rf.fill_between(rf_fpr, rf_tpr, alpha=0.1, color='#155799')

    # Add diagonal line with styling
    ax_rf.plot([0, 1], [0, 1], "k--", linewidth=1.5, alpha=0.7)

    # Customize grid
    ax_rf.grid(True, linestyle='--', alpha=0.3)

    # Customize labels and title
    ax_rf.set_xlabel("False Positive Rate", fontsize=10, color='#333333')
    ax_rf.set_ylabel("True Positive Rate", fontsize=10, color='#333333')
    ax_rf.set_title("Random Forest ROC Curve", fontsize=12, color='#155799', pad=15)

    # Customize ticks
    ax_rf.tick_params(colors='#333333')

    # Add legend with styling
    ax_rf.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='#155799')

    # Add AUC score annotation
    ax_rf.annotate(f'AUC = {rf_metrics["AUC"]:.3f}',
                  xy=(0.05, 0.95),
                  xycoords='axes fraction',
                  bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='#155799', alpha=0.8),
                  fontsize=10)

    # Adjust layout
    plt.tight_layout()
    st.pyplot(fig_rf)

with col2:
    fig_lr, ax_lr = plt.subplots(figsize=(6, 5), facecolor='#f4f6f8')
    ax_lr.set_facecolor('#f4f6f8')

    # Plot ROC curve with enhanced styling
    ax_lr.plot(lr_fpr, lr_tpr, linewidth=2.5, color='#159957', alpha=0.8, label='Logistic Regression')
    ax_lr.fill_between(lr_fpr, lr_tpr, alpha=0.1, color='#159957')

    # Add diagonal line with styling
    ax_lr.plot([0, 1], [0, 1], "k--", linewidth=1.5, alpha=0.7)

    # Customize grid
    ax_lr.grid(True, linestyle='--', alpha=0.3)

    # Customize labels and title
    ax_lr.set_xlabel("False Positive Rate", fontsize=10, color='#333333')
    ax_lr.set_ylabel("True Positive Rate", fontsize=10, color='#333333')
    ax_lr.set_title("Logistic Regression ROC Curve", fontsize=12, color='#159957', pad=15)

    # Customize ticks
    ax_lr.tick_params(colors='#333333')

    # Add legend with styling
    ax_lr.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='#159957')

    # Add AUC score annotation
    ax_lr.annotate(f'AUC = {lr_metrics["AUC"]:.3f}',
                  xy=(0.05, 0.95),
                  xycoords='axes fraction',
                  bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='#159957', alpha=0.8),
                  fontsize=10)

    # Adjust layout
    plt.tight_layout()
    st.pyplot(fig_lr)

# Add this CSS for table styling
st.markdown("""
<style>
    /* Professional Table Styling */
    .stTable {
        border-collapse: separate;
        border-spacing: 0;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    /* Column Headers */
    .stTable th {
        background: #0D1B2A;
        color: #ffffff;
        font-weight: 600;
        padding: 12px 16px;
        border-bottom: 2px solid rgba(255, 255, 255, 0.1);
    }
    /* Row Headers */
    .stTable th:first-child {
        background: #1B263B;
        color: #ffffff;
        font-weight: 600;
        padding: 12px 16px;
        border-right: 2px solid rgba(255, 255, 255, 0.1);
    }
    .stTable tr {
        background-color: #ffffff;
        transition: background-color 0.3s ease;
    }
    .stTable tr:hover {
        background-color: #f8f9fa;
    }
    .stTable td {
        padding: 10px 14px;
        color: #333333;
        border-bottom: 1px solid rgba(13, 27, 42, 0.1);
    }
    .stTable tr:last-child td {
        border-bottom: none;
    }
</style>
""", unsafe_allow_html=True)

# Display metrics table with enhanced styling
st.markdown("""
<style>
    /* Base table styles */
    div[data-testid="stTable"] table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        border-radius: 10px;
        overflow: hidden;
        margin: 1rem 0;
        background: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }

    /* Header cells */
    div[data-testid="stTable"] th {
        background: linear-gradient(135deg, #155799, #159957);
        color: white !important;
        font-weight: 600;
        padding: 12px 24px;
        text-align: left;
        font-size: 1rem;
        border: none;
    }

    /* Data cells */
    div[data-testid="stTable"] td {
        padding: 12px 24px;
        border-bottom: 1px solid rgba(0,0,0,0.1);
        color: #333;
        font-size: 0.95rem;
        background: white;
    }

    /* Row hover effect */
    div[data-testid="stTable"] tr:hover td {
        background-color: rgba(21, 87, 153, 0.05);
    }

    /* Last row - remove bottom border */
    div[data-testid="stTable"] tr:last-child td {
        border-bottom: none;
    }

    /* First column styling */
    div[data-testid="stTable"] td:first-child {
        font-weight: 500;
        color: #155799;
    }

    /* Numeric columns alignment */
    div[data-testid="stTable"] td:not(:first-child) {
        text-align: center;
    }

    /* Mobile responsiveness */
    @media screen and (max-width: 768px) {
        div[data-testid="stTable"] table {
            font-size: 14px;
        }

        div[data-testid="stTable"] th,
        div[data-testid="stTable"] td {
            padding: 8px 12px;
        }
    }

    /* Ensure text contrast */
    div[data-testid="stTable"] {
        color: #333 !important;
    }

    /* Add zebra striping for better readability */
    div[data-testid="stTable"] tr:nth-child(even) td {
        background: rgba(247, 250, 252, 0.6);
    }
</style>
""", unsafe_allow_html=True)

# When displaying metrics table, use st.table instead of dataframe
metrics_df = pd.DataFrame([rf_metrics, lr_metrics],
                         index=["Random Forest", "Logistic Regression"])
metrics_df = metrics_df.round(3)

# Add a container for better styling
st.markdown("""
<div style="
    background: white;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    margin: 20px 0;
">
    <h3 style="
        color: #155799;
        margin-bottom: 15px;
        text-align: center;
        font-size: 1.5rem;
    ">Model Performance Metrics</h3>
</div>
""", unsafe_allow_html=True)

# Display the table
st.table(metrics_df)

# Add explanatory text below the table
st.markdown("""
<div style="
    background: rgba(21, 87, 153, 0.05);
    padding: 15px;
    border-radius: 8px;
    margin-top: 15px;
    font-size: 0.9rem;
    color: #333;
">
    <strong>Metrics Explanation:</strong>
    <ul style="margin-top: 10px; margin-bottom: 0;">
        <li><strong>Accuracy:</strong> Overall prediction accuracy</li>
        <li><strong>Precision:</strong> Proportion of correct positive predictions</li>
        <li><strong>Recall:</strong> Proportion of actual positives correctly identified</li>
        <li><strong>F1 Score:</strong> Harmonic mean of precision and recall</li>
        <li><strong>AUC:</strong> Area Under the ROC Curve</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Model Training Note
st.markdown("""
<div style="
    background: linear-gradient(135deg, #eef2f3, #8e9eab);
    padding: 25px 30px;
    border-radius: 15px;
    margin-top: 20px;
    box-shadow: 6px 6px 18px rgba(0, 0, 0, 0.1);
">
    <h4 style="color: #003366; margin-bottom: 15px;">📘 Model Training Summary</h4>
    <ul style="color: #333333; font-size: 16px; line-height: 1.8;">
        <li><b>Data Size:</b> 1,000 synthetic patient profiles generated for PPI risk analysis</li>
        <li><b>Real-World Validation:</b> Tested on 500 real patient datasets collected from the Department of Medicine, MKCG Medical College & Hospital</li>
        <li><b>Data Preprocessing:</b> Categorical encoding with one-hot; SMOTE used for class balance</li>
        <li><b>Train-Test Split:</b> 80% for training, 20% for evaluation</li>
        <li><b>Models Evaluated:</b> Random Forest & Logistic Regression</li>
        <li><b>Classification Target:</b> High Risk (Score ≥ 7) vs Lower Risk (Score &lt; 7)</li>
        <li><b>Key Metrics:</b> Accuracy, Precision, Recall, F1 Score, ROC AUC</li>
    </ul>
    <p style="font-size: 13px; color: #444;">⚠️ These results are based on synthetic data and initial real-world testing, and further validation is ongoing.</p>
</div>
""", unsafe_allow_html=True)

# Feedback Form Section
st.markdown("---")
st.markdown("""
<div style="
    background: linear-gradient(135deg, #f8f9fa, #e9ecef);
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    margin: 20px 0;
">
    <h3 style="
        color: #1B263B;
        margin-bottom: 20px;
        font-size: 1.5em;
        font-weight: 600;
    ">📝 Feedback Form</h3>
    <p style="
        color: #415A77;
        font-size: 1.1em;
        margin-bottom: 20px;
    ">Your feedback helps us improve. Please share your thoughts and suggestions.</p>
</div>
""", unsafe_allow_html=True)

# Create columns for form layout
col1, col2 = st.columns(2)  # Equal width columns

with col1:
    email = st.text_input("📧 Email Address",
                         placeholder="your.email@example.com",
                         help="We'll only use this to follow up if needed",
                         key="email_input")

with col2:
    recommendations = st.text_input("💭 Recommendations",
                                  placeholder="Share your suggestions for improvement...",
                                  help="Your insights help us enhance the tool",
                                  key="recommendations_input")

# Submit Button with Custom Styling
st.markdown("""
<style>
div.stButton > button {
    width: 200px;
    margin: 0 auto;
    display: block;
    background: linear-gradient(135deg, #155799, #159957);
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 5px;
    font-weight: 600;
    transition: all 0.3s ease;
}
div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 15px rgba(21, 87, 153, 0.2);
}
</style>
""", unsafe_allow_html=True)

if st.button("Submit Feedback"):
    if email and recommendations:
        st.success("Thank you for your feedback! We appreciate your input.")
        # Here you could add code to store the feedback in a database or send it via email
    else:
        st.warning("Please fill in both email and recommendations fields.")

# Add some spacing at the bottom
st.markdown("<br><br>", unsafe_allow_html=True)

# Add copyright notice
st.markdown("""
<div style="
    text-align: center;
    padding: 20px;
    color: #415A77;
    font-size: 0.9em;
    border-top: 1px solid rgba(65, 90, 119, 0.2);
    margin-top: 20px;
">
    Copyright © 2023 Dr. Nabyendu Biswas
</div>
""", unsafe_allow_html=True)
