import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ==========================================
# CONFIG & LOAD
# ==========================================
st.set_page_config(page_title="PHN No-Show Predictor", layout="wide")
MODEL_PATH = './model/Panhandle_Health_Network_XGBoost_model.pkl'

@st.cache_resource
def load_model():
    # In real life, use try/except
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    # Loading the raw CSV just to simulate 'incoming' data
    return pd.read_csv('./dataset/Panhandle_Health_Network_appointment_analysis.csv')

model = load_model()
df = load_data()

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.image("./dataset/phn600.png", use_container_width=True)
st.sidebar.title("Configuration")

risk_threshold = st.sidebar.slider("Intervention Threshold", 0.0, 1.0, 0.45, 
    help="Patients with probability higher than this will be flagged.")

clinic_filter = st.sidebar.selectbox("Select Clinic/Provider", ["All"] + list(df['Provider'].unique()))

# ==========================================
# MAIN DASHBOARD
# ==========================================
st.title("Panhandle Health Network: Operational Dashboard")
st.markdown("### Daily Appointment Risk Monitor")

#Preprocessing for Inference
df['Date.diff'] = pd.to_numeric(df['Date.diff'], errors='coerce') #Ensure numeric
#Encode Function
cat_cols = ['Gender', 'Neighbourhood', 'Provider', 'Insurance_type', 
            'Appointment_time_of_day', 'ZIP_code']

# Create a copy for display
display_df = df.copy()

# Simple mock encoding for prediction
input_df = df.copy()
# Create features needed
input_df['DayOfWeek'] = 0 # Placeholder
input_df['Age_Group'] = 0 # Placeholder
input_df['Interaction_LeadTime_Distance'] = input_df['Lead_time_days'] * input_df['Distance_to_clinic_miles']
input_df['Month'] = 1

#Select only numeric/compatible columns for the mock inference
pred_cols_numeric = [
    'Age', 'Hypertension', 'Diabetes', 'Alcoholism', 'Handcap', 
    'SMS_received', 'Distance_to_clinic_miles', 'Has_transportation', 
    'Past_no_shows', 'Visit_frequency', 'Lead_time_days', 'Copay',
    'Interaction_LeadTime_Distance'
]

#Create initial inference dataframe with numerics
dummy_X = input_df[pred_cols_numeric].copy()

#Handle Categoricals & Column Ordering
#Get the exact feature names and order from the trained model
try:
    needed_cols = model.get_booster().feature_names
except:
    needed_cols = model.feature_names_in_

for col in needed_cols:
    if col not in dummy_X.columns:
        dummy_X[col] = 0

dummy_X = dummy_X[needed_cols]

probs = model.predict_proba(dummy_X)[:, 1]
display_df['Risk_Score'] = probs

#KPI METRICS
if clinic_filter != "All":
    display_df = display_df[display_df['Provider'] == clinic_filter]

high_risk = display_df[display_df['Risk_Score'] > risk_threshold]
potential_loss = len(high_risk) * 200 # $200 per visit

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Appointments", len(display_df))
col2.metric("High Risk Patients", len(high_risk), delta_color="inverse")
col3.metric("Projected Revenue Risk", f"${potential_loss:,.0f}")
col4.metric("Intervention Capacity", "85%", help="Staff availability for calls")

#VISUALS & ACTION LIST
st.divider()

c1, c2 = st.columns([2, 1])

with c1:
    st.subheader("High Risk Patient List (Action Required)")
    st.dataframe(
        high_risk[['PatientId', 'AppointmentDay', 'Provider', 'Risk_Score', 'Phone_Number'] if 'Phone_Number' in high_risk else 
                  ['PatientId', 'AppointmentDay', 'Provider', 'Risk_Score', 'Past_no_shows', 'Has_transportation']]
        .sort_values('Risk_Score', ascending=False)
        .style.background_gradient(subset=['Risk_Score'], cmap='Reds'),
        use_container_width=True
    )
    st.caption("*Sorted by probability of No-Show. Prioritize calls for top rows.*")

with c2:
    st.subheader("Risk Drivers")
    #Histogram of Risk Scores
    fig = px.histogram(display_df, x="Risk_Score", nbins=20, title="Population Risk Distribution",
                       color_discrete_sequence=['#3b82f6'])
    fig.add_vline(x=risk_threshold, line_dash="dash", line_color="red", annotation_text="Threshold")
    st.plotly_chart(fig, use_container_width=True)
    
    #Impact Factor
    st.info("""
    **Suggested Actions:**
    1. **Transport:** Offer free rideshare voucher to patients with `Has_transportation=0`.
    2. **Reminders:** Double SMS for patients with `Past_no_shows > 2`.
    """)

#BIAS MONITORING
with st.expander("Senior Leadership: Fairness & Bias Audit"):
    st.write("Ensuring the model does not disproportionately flag vulnerable insurance groups.")
    
    bias_df = display_df.groupby('Insurance_type')['Risk_Score'].mean().reset_index()
    fig_bias = px.bar(bias_df, x='Insurance_type', y='Risk_Score', 
                      title="Average Risk Score by Payer", color='Risk_Score')
    st.plotly_chart(fig_bias, use_container_width=True)