import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ----------------------------------------------------
# 1. PAGE CONFIGURATION & THEME
# ----------------------------------------------------
st.set_page_config(
    page_title="CMAPSS Telemetry Predictor",
    page_icon="✈️",
    layout="wide"
)

st.markdown("""
<style>
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    .main-title {
        color: #58a6ff;
        text-align: center;
        margin-top: 15px;
        margin-bottom: 5px;
        font-size: 28px;
    }
    .sub-title {
        color: #8b949e;
        text-align: center;
        margin-bottom: 25px;
        font-size: 13px;
    }
    .feature-group-header {
        color: #58a6ff;
        border-bottom: 1px solid #30363d;
        padding-bottom: 5px;
        margin-top: 15px;
        margin-bottom: 10px;
        font-size: 14px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# 2. CACHED MODEL LOADER
# ----------------------------------------------------
@st.cache_resource
def load_prediction_models():
    """Loads the pre-trained scaler, regressor, and classifier."""
    scaler_path = 'models/scaler.joblib'
    reg_path = 'models/rf_regressor.joblib'
    class_path = 'models/rf_classifier.joblib'
    
    if not (os.path.exists(scaler_path) and os.path.exists(reg_path) and os.path.exists(class_path)):
        return None, None, None
        
    scaler = joblib.load(scaler_path)
    rf_reg = joblib.load(reg_path)
    rf_class = joblib.load(class_path)
    return scaler, rf_reg, rf_class

scaler, rf_reg, rf_class = load_prediction_models()

# ----------------------------------------------------
# 3. INTERFACE LAYOUT & PREDICTOR
# ----------------------------------------------------
st.markdown("<h1 class='main-title'>✈️ CMAPSS Single-Cycle Diagnostic Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Enter raw values for all 24 individual engine features below to calculate Remaining Useful Life and Health Status</p>", unsafe_allow_html=True)

if scaler is None or rf_reg is None or rf_class is None:
    st.error("🚨 Model files not found! Please run the training script (`train_and_save_models.py`) first.")
    st.stop()

# Default values mapped to standard healthy baseline preset (Sea Level)
BASELINE_DEFAULTS = {
    'time_in_cycles': '1.0', 'setting_1': '0.0005', 'setting_2': '0.0002', 'TRA': '100.0',
    'T2': '518.67', 'T24': '642.30', 'T30': '1585.81', 'T50': '1401.15',
    'P2': '14.62', 'P15': '21.61', 'P30': '553.97', 'Nf': '2388.03', 'Nc': '9054.75', 'epr': '1.30',
    'Ps30': '47.31', 'phi': '521.92', 'NRf': '2388.03', 'NRc': '8136.67', 'BPR': '8.41',
    'htBleed': '391.99', 'Nf_dmd': '2388.00', 'PCNfR_dmd': '100.00', 'W31': '38.95', 'W32': '23.37'
}

# Standardized Feature Descriptions for Help Tips
HELP_TIPS = {
    'time_in_cycles': "Cumulative operational cycles run by the engine (Age)",
    'setting_1': "Operational setting 1 (Altitude)",
    'setting_2': "Operational setting 2 (Mach Speed)",
    'TRA': "Throttle Resolver Angle % (Operational Setting 3)",
    'T2': "Total temperature at Fan Inlet (Rankine)",
    'T24': "Total temperature at LPC Outlet (Rankine)",
    'T30': "Total temperature at HPC Outlet (Rankine)",
    'T50': "Total temperature at LPT Outlet (Rankine)",
    'P2': "Pressure at Fan Inlet (psia)",
    'P15': "Total pressure in bypass-duct (psia)",
    'P30': "Total pressure at HPC Outlet (psia)",
    'Nf': "Physical Fan Speed (RPM)",
    'Nc': "Physical Core Speed (RPM)",
    'epr': "Engine Pressure Ratio",
    'Ps30': "Static pressure at HPC Outlet (psia)",
    'phi': "Ratio of fuel flow to Ps30 (pps/psia)",
    'NRf': "Corrected Fan Speed (RPM)",
    'NRc': "Corrected Core Speed (RPM)",
    'BPR': "Bypass Ratio",
    'htBleed': "Bleed Enthalpy",
    'Nf_dmd': "Demanded Fan Speed (RPM)",
    'PCNfR_dmd': "Demanded Corrected Fan Speed (RPM)",
    'W31': "HPT Coolant Bleed (lbm/s)",
    'W32': "LPT Coolant Bleed (lbm/s)"
}

# Create a clean 4-column layout for the 24 inputs
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("<div class='feature-group-header'>🧭 Flight Controls</div>", unsafe_allow_html=True)
    val_cycles = st.text_input("Time in Cycles (Age)", value=BASELINE_DEFAULTS['time_in_cycles'], help=HELP_TIPS['time_in_cycles'])
    val_s1 = st.text_input("Altitude (setting_1)", value=BASELINE_DEFAULTS['setting_1'], help=HELP_TIPS['setting_1'])
    val_s2 = st.text_input("Mach Speed (setting_2)", value=BASELINE_DEFAULTS['setting_2'], help=HELP_TIPS['setting_2'])
    val_s3 = st.text_input("Throttle Angle (TRA)", value=BASELINE_DEFAULTS['TRA'], help=HELP_TIPS['TRA'])
    
    st.markdown("<div class='feature-group-header'>💨 Fan Inlets</div>", unsafe_allow_html=True)
    val_t2 = st.text_input("Fan Inlet Temp (T2)", value=BASELINE_DEFAULTS['T2'], help=HELP_TIPS['T2'])
    val_p2 = st.text_input("Fan Inlet Press (P2)", value=BASELINE_DEFAULTS['P2'], help=HELP_TIPS['P2'])

with col2:
    st.markdown("<div class='feature-group-header'>🌡️ Temperature Channels</div>", unsafe_allow_html=True)
    val_t24 = st.text_input("LPC Outlet Temp (T24)", value=BASELINE_DEFAULTS['T24'], help=HELP_TIPS['T24'])
    val_t30 = st.text_input("HPC Outlet Temp (T30)", value=BASELINE_DEFAULTS['T30'], help=HELP_TIPS['T30'])
    val_t50 = st.text_input("LPT Outlet Temp (T50)", value=BASELINE_DEFAULTS['T50'], help=HELP_TIPS['T50'])
    
    st.markdown("<div class='feature-group-header'>🌀 Bypass Pressure</div>", unsafe_allow_html=True)
    val_p15 = st.text_input("Bypass Duct Press (P15)", value=BASELINE_DEFAULTS['P15'], help=HELP_TIPS['P15'])
    val_p30 = st.text_input("HPC Outlet Press (P30)", value=BASELINE_DEFAULTS['P30'], help=HELP_TIPS['P30'])
    val_ps30 = st.text_input("HPC Static Press (Ps30)", value=BASELINE_DEFAULTS['Ps30'], help=HELP_TIPS['Ps30'])

with col3:
    st.markdown("<div class='feature-group-header'>⚡ Rotational Speeds</div>", unsafe_allow_html=True)
    val_nf = st.text_input("Physical Fan Speed (Nf)", value=BASELINE_DEFAULTS['Nf'], help=HELP_TIPS['Nf'])
    val_nc = st.text_input("Physical Core Speed (Nc)", value=BASELINE_DEFAULTS['Nc'], help=HELP_TIPS['Nc'])
    val_nrf = st.text_input("Corrected Fan Speed (NRf)", value=BASELINE_DEFAULTS['NRf'], help=HELP_TIPS['NRf'])
    val_nrc = st.text_input("Corrected Core Speed (NRc)", value=BASELINE_DEFAULTS['NRc'], help=HELP_TIPS['NRc'])
    
    st.markdown("<div class='feature-group-header'>⚙️ Pressure & Bypass</div>", unsafe_allow_html=True)
    val_epr = st.text_input("Pressure Ratio (epr)", value=BASELINE_DEFAULTS['epr'], help=HELP_TIPS['epr'])
    val_bpr = st.text_input("Bypass Ratio (BPR)", value=BASELINE_DEFAULTS['BPR'], help=HELP_TIPS['BPR'])

with col4:
    st.markdown("<div class='feature-group-header'>💧 Fuel & Enthalpy</div>", unsafe_allow_html=True)
    val_phi = st.text_input("Fuel-to-Pressure (phi)", value=BASELINE_DEFAULTS['phi'], help=HELP_TIPS['phi'])
    val_htbleed = st.text_input("Bleed Enthalpy (htBleed)", value=BASELINE_DEFAULTS['htBleed'], help=HELP_TIPS['htBleed'])
    
    st.markdown("<div class='feature-group-header'>💨 Demands & Coolants</div>", unsafe_allow_html=True)
    val_nfdmd = st.text_input("Demanded Fan Speed (Nf_dmd)", value=BASELINE_DEFAULTS['Nf_dmd'], help=HELP_TIPS['Nf_dmd'])
    val_pcnfr = st.text_input("Demanded Corr Speed (PCNfR)", value=BASELINE_DEFAULTS['PCNfR_dmd'], help=HELP_TIPS['PCNfR_dmd'])
    val_w31 = st.text_input("HPT Coolant Bleed (W31)", value=BASELINE_DEFAULTS['W31'], help=HELP_TIPS['W31'])
    val_w32 = st.text_input("LPT Coolant Bleed (W32)", value=BASELINE_DEFAULTS['W32'], help=HELP_TIPS['W32'])

# Predict button in centered column
st.markdown("<br>", unsafe_allow_html=True)
predict_clicked = st.button("Predict Engine Status & RUL", type="primary", use_container_width=True)

if predict_clicked:
    st.markdown("---")
    
    try:
        # 1. Parse and validate all 24 inputs to float values
        features_dict = {
            'time_in_cycles': float(val_cycles),
            'setting_1': float(val_s1),
            'setting_2': float(val_s2),
            'TRA': float(val_s3),
            'T2': float(val_t2),
            'T24': float(val_t24),
            'T30': float(val_t30),
            'T50': float(val_t50),
            'P2': float(val_p2),
            'P15': float(val_p15),
            'P30': float(val_p30),
            'Nf': float(val_nf),
            'Nc': float(val_nc),
            'epr': float(val_epr),
            'Ps30': float(val_ps30),
            'phi': float(val_phi),
            'NRf': float(val_nrf),
            'NRc': float(val_nrc),
            'BPR': float(val_bpr),
            'htBleed': float(val_htbleed),
            'Nf_dmd': float(val_nfdmd),
            'PCNfR_dmd': float(val_pcnfr),
            'W31': float(val_w31),
            'W32': float(val_w32)
        }
        
        # 2. Convert to DataFrame matching active schema order
        ACTIVE_FEATURES = [
            'time_in_cycles', 'setting_1', 'setting_2', 'TRA',
            'T2', 'T24', 'T30', 'T50', 'P2', 'P15', 'P30', 'Nf', 'Nc', 'epr',
            'Ps30', 'phi', 'NRf', 'NRc', 'BPR', 'htBleed', 'Nf_dmd', 'PCNfR_dmd',
            'W31', 'W32'
        ]
        
        df_input = pd.DataFrame([features_dict])[ACTIVE_FEATURES]
        
        # 3. Scale inputs using standardized joblib scaler
        X_scaled = pd.DataFrame(scaler.transform(df_input), columns=df_input.columns)
        
        # 4. Predict continuous RUL (capped between 0 and 125)
        pred_rul = float(rf_reg.predict(X_scaled)[0])
        pred_rul_clipped = max(0.0, min(125.0, pred_rul))
        
        # 5. Predict 3-class health status
        pred_class_id = int(rf_class.predict(X_scaled)[0])
        pred_probs = rf_class.predict_proba(X_scaled)[0]
        
        # 6. Display Diagnostics Outputs
        st.success("✅ Analysis Complete!")
        
        col_out1, col_out2 = st.columns(2)
        
        with col_out1:
            st.metric("Predicted Remaining Useful Life (RUL)", f"{pred_rul_clipped:.1f} Cycles")
            
        with col_out2:
            if pred_class_id == 0:
                st.markdown("### Health Status: 🟢 Normal / Good")
                st.info(f"Engine operating stably. Confidence: **{pred_probs[0]:.1%}**")
            elif pred_class_id == 1:
                st.markdown("### Health Status: 🟡 Warning / Moderate")
                st.warning(f"Early-stage wear detected. Confidence: **{pred_probs[1]:.1%}**")
            else:
                st.markdown("### Health Status: 🔴 Critical / Urgent EOL")
                st.error(f"High failure risk! Maintenance required. Confidence: **{pred_probs[2]:.1%}**")
                
    except ValueError as ve:
        st.error(f"❌ Input Parsing Error: Please verify that all fields contain valid numerical digits. Detail: {ve}")
    except Exception as e:
        st.error(f"❌ Diagnostic Inference Error: {e}")
