import pickle
import streamlit as st
import pandas as pd
import numpy as np
import os

# Page configuration
st.set_page_config(
    page_title="House Price Prediction System | Arun Pandey Laudari",
    page_icon="🏘️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Professional CSS styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    .header-container {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2.5rem 0;
        margin: -1rem -1rem 2rem -1rem;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .header-content {
        max-width: 1200px;
        margin: 0 auto;
        padding: 0 2rem;
    }
    
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .subtitle {
        font-size: 1.2rem;
        font-weight: 400;
        opacity: 0.9;
        margin-bottom: 1rem;
    }
    
    .developer-info {
        font-size: 0.95rem;
        opacity: 0.8;
        border-top: 1px solid rgba(255,255,255,0.2);
        padding-top: 1rem;
        margin-top: 1rem;
    }
    
    .form-section {
        background: white;
        border: 1px solid #e1e5e9;
        border-radius: 8px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    }
    
    .section-title {
        color: #2c3e50;
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #3498db;
    }
    
    .input-group {
        margin-bottom: 1.5rem;
    }
    
    .input-label {
        font-weight: 500;
        color: #34495e;
        margin-bottom: 0.5rem;
        display: block;
    }
    
    .results-panel {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 2rem;
        margin-bottom: 2rem;
    }
    
    .prediction-result {
        background: white;
        border: 2px solid #28a745;
        border-radius: 8px;
        padding: 2rem;
        text-align: center;
        margin: 1.5rem 0;
    }
    
    .price-value {
        font-size: 2.8rem;
        font-weight: 700;
        color: #28a745;
        margin: 0;
    }
    
    .price-label {
        font-size: 1.1rem;
        color: #6c757d;
        margin-top: 0.5rem;
    }
    
    .analysis-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-top: 1.5rem;
    }
    
    .analysis-card {
        background: white;
        border: 1px solid #e9ecef;
        border-radius: 6px;
        padding: 1.5rem;
        text-align: center;
    }
    
    .analysis-value {
        font-size: 1.5rem;
        font-weight: 600;
        color: #495057;
        margin-bottom: 0.5rem;
    }
    
    .analysis-label {
        font-size: 0.9rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .property-summary {
        background: white;
        border: 1px solid #e1e5e9;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    
    .summary-item {
        display: flex;
        justify-content: space-between;
        padding: 0.75rem 0;
        border-bottom: 1px solid #f1f3f4;
    }
    
    .summary-item:last-child {
        border-bottom: none;
    }
    
    .summary-label {
        font-weight: 500;
        color: #495057;
    }
    
    .summary-value {
        font-weight: 600;
        color: #2c3e50;
    }
    
    .model-info {
        background: #e3f2fd;
        border: 1px solid #bbdefb;
        border-radius: 6px;
        padding: 1.5rem;
        margin-top: 2rem;
    }
    
    .model-info h4 {
        color: #1565c0;
        margin-bottom: 1rem;
        font-size: 1.1rem;
    }
    
    .model-info p {
        color: #424242;
        font-size: 0.95rem;
        line-height: 1.5;
        margin-bottom: 0.5rem;
    }
    
    .predict-button {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border: none;
        padding: 1rem 2rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 1.1rem;
        width: 100%;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .predict-button:hover {
        background: linear-gradient(135deg, #1a3460 0%, #245086 100%);
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    .footer {
        background: #f8f9fa;
        border-top: 1px solid #e9ecef;
        padding: 2rem 0;
        margin: 3rem -1rem -1rem -1rem;
        text-align: center;
        color: #6c757d;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%) !important;
        color: white !important;
        border: none !important;
        padding: 1rem 2rem !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        width: 100% !important;
        transition: all 0.2s ease !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #1a3460 0%, #245086 100%) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15) !important;
    }
</style>
""", unsafe_allow_html=True)


# loading the model and scaler
# Load the model
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'notebook', 'linear_regression_model.pkl')
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

@st.cache_resource
def load_scaler():
    scaler_path = os.path.join(os.path.dirname(__file__), 'notebook', 'scaler.pkl')
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return scaler

scaler = load_scaler()

# Header Section
st.markdown("""
<div class="header-container">
    <div class="header-content">
        <h1 class="main-title">House Price Prediction System</h1>
        <p class="subtitle">Advanced Machine Learning Model for Real Estate Valuation</p>
        <div class="developer-info">
            <strong>Developed by:</strong> Arun Pandey Laudari<br>
            <strong>Model:</strong> Linear Regression with Standard Scaling | <strong>Accuracy:</strong> Optimized for  Housing Market
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Main content layout
col1, col2 = st.columns([3, 2])

with col1:
    # Property Details Form
    st.markdown("""
    <div class="form-section">
        <h3 class="section-title">Property Specifications</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Basic Property Details
    st.markdown("#### Basic Information")
    col1_1, col1_2 = st.columns(2)
    
    with col1_1:
        area = st.number_input(
            "Total Area (sq ft)",
            min_value=500.0,
            max_value=20000.0,
            value=7500.0,
            step=100.0,
            help="Enter the total built-up area in square feet"
        )
        
        bedrooms = st.selectbox(
            "Number of Bedrooms",
            options=[1, 2, 3, 4, 5],
            index=3,
            help="Total number of bedrooms"
        )
        
        bathrooms = st.selectbox(
            "Number of Bathrooms",
            options=[1, 2, 3],
            index=1,
            help="Total number of bathrooms"
        )

    with col1_2:
        stories = st.selectbox(
            "Number of Stories",
            options=[1, 2, 3],
            index=1,
            help="Total floors in the building"
        )
        
        parking = st.selectbox(
            "Parking Spaces",
            options=[0, 1, 2, 3],
            index=2,
            help="Number of dedicated parking spaces"
        )

    # Location & Amenities
    st.markdown("#### Location & Amenities")
    col2_1, col2_2 = st.columns(2)
    
    with col2_1:
        mainroad = st.selectbox(
            "Main Road Access",
            options=["No", "Yes"],
            help="Direct access to main road"
        )
        
        airconditioning = st.selectbox(
            "Air Conditioning",
            options=["No", "Yes"],
            help="Central or split AC system available"
        )

    with col2_2:
        prefarea = st.selectbox(
            "Preferred Area",
            options=["No", "Yes"],
            help="Located in a preferred/premium locality"
        )
        
        furnishingstatus = st.selectbox(
            "Furnishing Status",
            options=["Unfurnished", "Semi-furnished", "Furnished"],
            help="Current furnishing condition"
        )

    # Prediction Button
    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("Generate Price Prediction", type="primary")

with col2:
    # Property Summary Panel
    st.markdown("""
    <div class="form-section">
        <h3 class="section-title">Property Summary</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="property-summary">
        <div class="summary-item">
            <span class="summary-label">Total Area:</span>
            <span class="summary-value">{area:,.0f} sq ft</span>
        </div>
        <div class="summary-item">
            <span class="summary-label">Configuration:</span>
            <span class="summary-value">{bedrooms}BHK, {bathrooms} Bath</span>
        </div>
        <div class="summary-item">
            <span class="summary-label">Stories:</span>
            <span class="summary-value">{stories} Floor(s)</span>
        </div>
        <div class="summary-item">
            <span class="summary-label">Parking:</span>
            <span class="summary-value">{parking} Space(s)</span>
        </div>
        <div class="summary-item">
            <span class="summary-label">Main Road:</span>
            <span class="summary-value">{mainroad}</span>
        </div>
        <div class="summary-item">
            <span class="summary-label">Air Conditioning:</span>
            <span class="summary-value">{airconditioning}</span>
        </div>
        <div class="summary-item">
            <span class="summary-label">Preferred Area:</span>
            <span class="summary-value">{prefarea}</span>
        </div>
        <div class="summary-item">
            <span class="summary-label">Furnishing:</span>
            <span class="summary-value">{furnishingstatus}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Model Information
    st.markdown("""
    <div class="model-info">
        <h4>Model Information</h4>
        <p><strong>Algorithm:</strong> Linear Regression</p>
        <p><strong>Preprocessing:</strong> Standard Scaling (Z-score normalization)</p>
        <p><strong>Features:</strong> 9 property characteristics</p>
        <p><strong>Target:</strong> Log-transformed price (₹)</p>
        <p><strong>Developer:</strong> Arun Pandey Laudari</p>
    </div>
    """, unsafe_allow_html=True)

# Prediction Results
if predict_clicked:
    # Process inputs
    mainroad_val = 1 if mainroad == "Yes" else 0
    airconditioning_val = 1 if airconditioning == "Yes" else 0
    prefarea_val = 1 if prefarea == "Yes" else 0
    furnishing_map = {"Unfurnished": 0, "Semi-furnished": 1, "Furnished": 2}
    furnishingstatus_val = furnishing_map[furnishingstatus]

    # Create prediction dataframe
    feature_columns = [
        "area", "bedrooms", "bathrooms", "stories",
        "mainroad", "airconditioning", "parking",
        "prefarea", "furnishingstatus"
    ]
    new_data = pd.DataFrame([[
        area, bedrooms, bathrooms, stories,
        mainroad_val, airconditioning_val, parking,
        prefarea_val, furnishingstatus_val
    ]], columns=feature_columns)

    # Make prediction
    new_scaled = scaler.transform(new_data)
    pred_log = model.predict(new_scaled)
    predicted_price = np.expm1(pred_log[0])

    # Display results
    st.markdown("""
    <div class="form-section">
        <h3 class="section-title">Prediction Results</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="prediction-result">
        <h2 class="price-value">₹ {predicted_price:,.2f}</h2>
        <p class="price-label">Estimated Market Value</p>
    </div>
    """, unsafe_allow_html=True)

    # Analysis metrics
    price_per_sqft = predicted_price / area
    total_rooms = bedrooms + bathrooms
    
    st.markdown(f"""
    <div class="analysis-grid">
        <div class="analysis-card">
            <div class="analysis-value">₹ {price_per_sqft:,.0f}</div>
            <div class="analysis-label">Price per sq ft</div>
        </div>
        <div class="analysis-card">
            <div class="analysis-value">{total_rooms}</div>
            <div class="analysis-label">Total Rooms</div>
        </div>
        <div class="analysis-card">
            <div class="analysis-value">{stories}</div>
            <div class="analysis-label">Stories</div>
        </div>
        <div class="analysis-card">
            <div class="analysis-value">{parking}</div>
            <div class="analysis-label">Parking</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Additional insights
    st.markdown("#### Market Analysis")
    
    premium_features = []
    if mainroad == "Yes":
        premium_features.append("Main road connectivity")
    if airconditioning == "Yes":
        premium_features.append("Air conditioning")
    if prefarea == "Yes":
        premium_features.append("Premium location")
    if furnishingstatus == "Furnished":
        premium_features.append("Fully furnished")
    
    if premium_features:
        features_text = ", ".join(premium_features)
        st.info(f"**Premium Features Detected:** {features_text}")
    
    if price_per_sqft > 1000:
        st.success("**Market Position:** Above average price per sq ft indicates premium property")
    elif price_per_sqft < 500:
        st.info("**Market Position:** Below average price per sq ft indicates budget-friendly property")
    else:
        st.info("**Market Position:** Average market price per sq ft")

# Footer
st.markdown("""
<div class="footer">
    <p><strong>House Price Prediction System</strong> | Developed by <strong>Arun Pandey Laudari</strong></p>
    <p>© 2024 | Machine Learning Model for Real Estate Valuation | All Rights Reserved</p>
    <p><em>Disclaimer: Predictions are estimates based on historical data and market trends. 
    Actual prices may vary based on current market conditions and additional factors not included in the model.</em></p>
</div>
""", unsafe_allow_html=True)