import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from xgboost import XGBClassifier

# ==========================================
# 1. CONFIGURATION & CONSTANTS
# ==========================================
DATA_PATH = './dataset/Panhandle_Health_Network_appointment_analysis.csv'
MODEL_PATH = './model/Panhandle_Health_Network_XGBoost_model.pkl'
COST_OF_NO_SHOW = 200 # Average cost in USD
COST_OF_INTERVENTION = 15 # Cost of SMS + Staff Call + Transport Help
RANDOM_SEED = 67

# ==========================================
# 2. DATA LOADING & PREPROCESSING
# ==========================================
def load_and_clean_data(filepath):
    print("Loading data")
    df = pd.read_csv(filepath)
    
    #Convert Dates
    df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
    df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])
    
    #Target Encoding: No-Show as 1, if Showed_up is 'False', they missed the appointment -> 1
    df['target'] = df['Showed_up'].apply(lambda x: 1 if x == False else 0)
    
    return df

def feature_engineering(df):
    print("Engineering features")
    
    #Temporal Features
    df['DayOfWeek'] = df['AppointmentDay'].dt.day_name()
    df['Month'] = df['AppointmentDay'].dt.month
    
    #Interaction Features
    #High lead time + Low Copay might indicate Medicaid patients with transport issues
    df['Interaction_LeadTime_Distance'] = df['Lead_time_days'] * df['Distance_to_clinic_miles']
    
    #Binning Age
    df['Age_Group'] = pd.cut(df['Age'], bins=[0, 18, 35, 55, 100], labels=['Child', 'Young Adult', 'Adult', 'Senior'])
    
    return df

def prepare_for_modeling(df):
    print("Preparing the dataset for modeling :D")
    #Select Columns for Training
    #Dropping IDs and original dates (using derived features instead)
    drop_cols = ['PatientId', 'AppointmentID', 'ScheduledDay', 'AppointmentDay', 'Showed_up', 'target']
    
    # Columns to Keep (Categorical & Numerical)
    categorical_cols = ['Gender', 'Neighbourhood', 'Provider', 'Insurance_type', 
                        'Appointment_time_of_day', 'DayOfWeek', 'Age_Group', 'ZIP_code']
    
    numerical_cols = ['Age', 'Hypertension', 'Diabetes', 'Alcoholism', 'Handcap', 
                      'SMS_received', 'Distance_to_clinic_miles', 'Has_transportation', 
                      'Past_no_shows', 'Visit_frequency', 'Lead_time_days', 'Copay',
                      'Interaction_LeadTime_Distance']
    
    X = df[categorical_cols + numerical_cols].copy()
    y = df['target']
    
    #Label Encoding for XGBoost
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        
    return X, y

# ==========================================
# 3. MODEL TRAINING (XGBoost)
# ==========================================
def train_model(X, y):
    print("Splitting data and training XGBoost")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
    
    #Calculate scale_pos_weight for imbalance, No-shows are usually minority
    ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
    
    model = XGBClassifier(
        objective='binary:logistic',
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        scale_pos_weight=ratio, # Handling Class Imbalance
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=RANDOM_SEED
    )
    
    model.fit(X_train, y_train)
    
    # Save for the dashboard
    joblib.dump(model, MODEL_PATH)
    
    return model, X_test, y_test

# ==========================================
# 4. EVALUATION & BUSINESS IMPACT
# ==========================================
def evaluate_financial_impact(y_true, y_prob, threshold=0.5):
    """
    Calculates ROI based on PHN constraints.
    """
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Scenarios
    # TP: Predicted No-show, was No-show. We intervene. Assumed 50% success rate of intervention.
    #     Gain: (0.5 * COST_OF_NO_SHOW) - COST_OF_INTERVENTION
    # FP: Predicted No-show, actually showed. We wasted money on intervention.
    #     Loss: -COST_OF_INTERVENTION
    # FN: Predicted Show, actually No-show. We did nothing.
    #     Loss: -COST_OF_NO_SHOW (Opportunity Cost)
    # TN: Predicted Show, actually Showed.
    #     Gain: 0
    
    success_rate_intervention = 0.50 
    
    revenue_recovered = tp * (COST_OF_NO_SHOW * success_rate_intervention)
    intervention_costs = (tp + fp) * COST_OF_INTERVENTION
    
    net_savings = revenue_recovered - intervention_costs
    
    print(f"\n--- FINANCIAL IMPACT (Threshold {threshold}) ---")
    print(f"Interventions Triggered: {tp + fp}")
    print(f"Successful Interventions (Est): {int(tp * success_rate_intervention)}")
    print(f"Cost of Interventions: ${intervention_costs:,.2f}")
    print(f"Revenue Recovered: ${revenue_recovered:,.2f}")
    print(f"NET SAVINGS: ${net_savings:,.2f}")
    
    return net_savings

def run_analysis():
    #Load
    df = load_and_clean_data(DATA_PATH)
    df = feature_engineering(df)

    #EDA Snippet
    print("EDA SUMMARY")
    print(f"Overall No-Show Rate: {df['target'].mean():.2%}")
    print(f"No-Show Rate by Insurance:\n{df.groupby('Insurance_type')['target'].mean()}")
    
    #Train
    X, y = prepare_for_modeling(df)
    model, X_test, y_test = train_model(X, y)
    
    #Standard Metrics
    y_prob = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"\nModel ROC-AUC: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, (y_prob >= 0.5).astype(int)))
    
    #Business Optimization (Finding optimal threshold)
    print("Optimizing Threshold for Profit...")
    best_savings = -float('inf')
    best_thresh = 0
    
    for thresh in np.arange(0.1, 0.9, 0.05):
        savings = evaluate_financial_impact(y_test, y_prob, threshold=thresh)
        if savings > best_savings:
            best_savings = savings
            best_thresh = thresh
            
    print(f"\n>>> OPTIMAL DEPLOYMENT SETTINGS: Threshold {best_thresh:.2f} yields ${best_savings:,.2f} savings on test set.")
    
    #SHAP Values
    print("\nGenerating SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    #Create a simple plot
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("Feature Importance (SHAP)")
    plt.tight_layout()
    plt.savefig('shap_summary.png')
    print("SHAP summary saved to 'shap_summary.png'")

    #Fairness Audit Check
    print("\n--- FAIRNESS AUDIT (False Positive Rate by Insurance) ---")
    X_test_audit = X_test.copy()
    X_test_audit['target'] = y_test
    X_test_audit['pred'] = (y_prob >= best_thresh).astype(int)
    
    #Group by Insurance type
    audit = X_test_audit.groupby('Insurance_type').apply(
        lambda x: ((x['pred'] == 1) & (x['target'] == 0)).sum() / len(x)
    )
    print("Risk of over-policing (False Positive Rate) by Group:")
    print(audit)

if __name__ == "__main__":
    run_analysis()