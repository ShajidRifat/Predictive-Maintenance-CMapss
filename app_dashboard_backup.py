import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import time
import os

# ----------------------------------------------------
# 1. PAGE CONFIGURATION & THEME STYLING
# ----------------------------------------------------
st.set_page_config(
    page_title="Jet Engine Prognostics Control Panel",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Injection for a Premium dark control room UI (Dark mode, neon glows)
st.markdown("""
<style>
    /* Global Background and Fonts */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    }
    
    /* SideBar UI Customization */
    section[data-testid="stSidebar"] {
        background-color: #161b22 !important;
        border-right: 1px solid #30363d;
    }
    
    /* Metrics / Metric Cards Customization */
    div[data-testid="metric-container"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease-in-out;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        border-color: #58a6ff;
    }
    
    /* Neon glowing Health Badges */
    .health-badge {
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        text-align: center;
        display: inline-block;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .badge-good {
        background-color: rgba(46, 160, 67, 0.15);
        color: #3fb950;
        border: 1px solid #2ea043;
        box-shadow: 0 0 10px rgba(46, 160, 67, 0.3);
        animation: pulse-green 2s infinite;
    }
    .badge-warn {
        background-color: rgba(210, 153, 34, 0.15);
        color: #d29922;
        border: 1px solid #d29922;
        box-shadow: 0 0 10px rgba(210, 153, 34, 0.3);
    }
    .badge-critical {
        background-color: rgba(248, 81, 73, 0.15);
        color: #f85149;
        border: 1px solid #f85149;
        box-shadow: 0 0 10px rgba(248, 81, 73, 0.5);
        animation: flash-red 1s infinite;
    }
    
    /* Keyframe Animations */
    @keyframes pulse-green {
        0% { box-shadow: 0 0 5px rgba(46, 160, 67, 0.3); }
        50% { box-shadow: 0 0 15px rgba(46, 160, 67, 0.6); }
        100% { box-shadow: 0 0 5px rgba(46, 160, 67, 0.3); }
    }
    @keyframes flash-red {
        0%, 100% { opacity: 1; box-shadow: 0 0 15px rgba(248, 81, 73, 0.6); }
        50% { opacity: 0.6; box-shadow: 0 0 5px rgba(248, 81, 73, 0.2); }
    }
    
    /* Header decoration */
    .header-panel {
        background: linear-gradient(90deg, #161b22 0%, #0d1117 100%);
        padding: 20px;
        border-radius: 8px;
        border-bottom: 2px solid #30363d;
        margin-bottom: 25px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# 2. CACHED MODEL LOADER & TELEMETRY CONSTANTS
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

# Hardcoded Baseline Regime Means (pre-computed from mega training set)
REGIME_MEANS = {
    1: {
        'setting_1': 0.000456, 'setting_2': 0.000152, 'TRA': 100.000000,
        'T2': 518.670000, 'T24': 642.302800, 'T30': 1585.810107, 'T50': 1401.148597,
        'P2': 14.620000, 'P15': 21.605200, 'P30': 553.972831, 'Nf': 2388.031057,
        'Nc': 9054.754688, 'epr': 1.299993, 'Ps30': 47.305297, 'phi': 521.923999,
        'NRf': 2388.030081, 'NRc': 8136.672556, 'BPR': 8.411848, 'htBleed': 391.989576,
        'Nf_dmd': 2388.000000, 'PCNfR_dmd': 100.000000, 'W31': 38.953137, 'W32': 23.371425,
    },
    2: {
        'setting_1': 10.002871, 'setting_2': 0.250514, 'TRA': 100.000000,
        'T2': 489.050000, 'T24': 604.540235, 'T30': 1497.597578, 'T50': 1303.818022,
        'P2': 10.520000, 'P15': 15.486689, 'P30': 394.684904, 'Nf': 2318.863546,
        'Nc': 8776.901174, 'epr': 1.259790, 'Ps30': 45.254476, 'phi': 371.742422,
        'NRf': 2388.063336, 'NRc': 8129.931450, 'BPR': 8.632418, 'htBleed': 368.539816,
        'Nf_dmd': 2319.000000, 'PCNfR_dmd': 100.000000, 'W31': 28.618751, 'W32': 17.170930,
    },
    3: {
        'setting_1': 20.002993, 'setting_2': 0.700492, 'TRA': 100.000000,
        'T2': 491.190000, 'T24': 607.222003, 'T30': 1481.253005, 'T50': 1246.656166,
        'P2': 9.350000, 'P15': 13.647936, 'P30': 334.697539, 'Nf': 2323.910449,
        'Nc': 8720.697660, 'epr': 1.078273, 'Ps30': 44.238981, 'phi': 315.012288,
        'NRf': 2388.040907, 'NRc': 8059.641865, 'BPR': 9.202503, 'htBleed': 364.268566,
        'Nf_dmd': 2324.000000, 'PCNfR_dmd': 100.000000, 'W31': 24.523541, 'W32': 14.712753,
    },
    4: {
        'setting_1': 25.002952, 'setting_2': 0.620492, 'TRA': 60.000000,
        'T2': 462.540000, 'T24': 536.656028, 'T30': 1259.078065, 'T50': 1044.827161,
        'P2': 7.050000, 'P15': 9.022959, 'P30': 175.531622, 'Nf': 1915.320756,
        'Nc': 8008.226581, 'epr': 0.939714, 'Ps30': 36.615198, 'phi': 164.663871,
        'NRf': 2028.213078, 'NRc': 7872.828949, 'BPR': 10.875131, 'htBleed': 306.364055,
        'Nf_dmd': 1915.000000, 'PCNfR_dmd': 84.930000, 'W31': 14.305152, 'W32': 8.588414,
    },
    5: {
        'setting_1': 35.003165, 'setting_2': 0.840523, 'TRA': 100.000000,
        'T2': 449.440000, 'T24': 555.467097, 'T30': 1362.801779, 'T50': 1125.164335,
        'P2': 5.480000, 'P15': 7.996966, 'P30': 194.585309, 'Nf': 2222.932013,
        'Nc': 8348.231170, 'epr': 1.020787, 'Ps30': 41.772865, 'phi': 183.154869,
        'NRf': 2388.021901, 'NRc': 8066.691779, 'BPR': 9.301597, 'htBleed': 333.214419,
        'Nf_dmd': 2223.000000, 'PCNfR_dmd': 100.000000, 'W31': 14.877594, 'W32': 8.924749,
    },
    6: {
        'setting_1': 42.003131, 'setting_2': 0.840525, 'TRA': 100.000000,
        'T2': 445.000000, 'T24': 549.379565, 'T30': 1350.224565, 'T50': 1121.280712,
        'P2': 3.910000, 'P15': 5.711670, 'P30': 138.710644, 'Nf': 2211.902529,
        'Nc': 8318.955712, 'epr': 1.019927, 'Ps30': 41.929513, 'phi': 130.607592,
        'NRf': 2387.978916, 'NRc': 8083.133958, 'BPR': 9.348368, 'htBleed': 329.952880,
        'Nf_dmd': 2212.000000, 'PCNfR_dmd': 100.000000, 'W31': 10.617419, 'W32': 6.370506,
    },
}

REGIME_DESCRIPTIONS = {
    1: "Sea Level ground tests (100% Throttle, 0k ft, 0.0 Mach)",
    2: "Low Altitude flight/climb (100% Throttle, 10k ft, 0.25 Mach)",
    3: "Medium Altitude climb/flight (100% Throttle, 20k ft, 0.70 Mach)",
    4: "Mid-High Altitude cruise (60% Throttle - IDLE load, 25k ft, 0.62 Mach)",
    5: "High Altitude Cruise (100% Throttle, 35k ft, 0.84 Mach)",
    6: "Extreme Altitude Cruise (100% Throttle, 42k ft, 0.84 Mach)",
}

# Features expected by our scaler (excluding dropped constant 'farB')
ACTIVE_FEATURES = [
    'time_in_cycles', 'setting_1', 'setting_2', 'TRA',
    'T2', 'T24', 'T30', 'T50', 'P2', 'P15', 'P30', 'Nf', 'Nc', 'epr',
    'Ps30', 'phi', 'NRf', 'NRc', 'BPR', 'htBleed', 'Nf_dmd', 'PCNfR_dmd',
    'W31', 'W32'
]

# ----------------------------------------------------
# 3. INTERACTIVE SIMULATION UTILS
# ----------------------------------------------------
@st.cache_data
def get_preloaded_engine(engine_name):
    """Loads a specific pre-configured engine unit for the flight player."""
    columns = [
        'unit_number', 'time_in_cycles', 'setting_1', 'setting_2', 'TRA',
        'T2', 'T24', 'T30', 'T50', 'P2', 'P15', 'P30', 'Nf', 'Nc', 'epr',
        'Ps30', 'phi', 'NRf', 'NRc', 'BPR', 'farB', 'htBleed', 'Nf_dmd',
        'PCNfR_dmd', 'W31', 'W32'
    ]
    
    if engine_name == "FD001 - Engine 1 (Standard, Sea Level)":
        df = pd.read_csv("CMaps/test_FD001.txt", sep=r"\s+", header=None, names=columns)
        engine_data = df[df['unit_number'] == 1].copy()
        eol = len(engine_data) + 112  # cycles_run (31) + actual_rul (112)
        desc = "Baseline engine operating in a single constant Sea Level regime (HPC Degradation)."
    elif engine_name == "FD002 - Engine 10 (Multi-Regime, HPC Wear)":
        df = pd.read_csv("CMaps/test_FD002.txt", sep=r"\s+", header=None, names=columns)
        engine_data = df[df['unit_number'] == 10].copy()
        eol = len(engine_data) + 79   # cycles_run (84) + actual_rul (79)
        desc = "Complex engine experiencing multiple altitude/thrust transitions (HPC Degradation)."
    elif engine_name == "FD003 - Engine 24 (Sea Level, Multi-Fault)":
        df = pd.read_csv("CMaps/test_FD003.txt", sep=r"\s+", header=None, names=columns)
        engine_data = df[df['unit_number'] == 24].copy()
        eol = len(engine_data) + 9    # cycles_run (475) + actual_rul (9)
        desc = "Extremely long-running engine at Sea Level facing joint HPC & Fan degradations."
    else: # FD004 - Engine 40
        df = pd.read_csv("CMaps/test_FD004.txt", sep=r"\s+", header=None, names=columns)
        engine_data = df[df['unit_number'] == 40].copy()
        eol = len(engine_data) + 10   # cycles_run (266) + actual_rul (10)
        desc = "The ultimate challenge: multi-regime, multi-fault (HPC & Fan) simultaneous failures."
        
    engine_data['EOL'] = eol
    engine_data['True_RUL'] = eol - engine_data['time_in_cycles']
    return engine_data.drop(columns=['farB']), desc

# ----------------------------------------------------
# 4. APP HEADER & LAYOUT
# ----------------------------------------------------
st.markdown("""
    <div class="header-panel">
        <h1 style='margin: 0; color: #58a6ff; font-size: 32px; letter-spacing: 0.5px;'>
            ✈️ Jet Engine Prognostics Control Panel
        </h1>
        <p style='margin: 5px 0 0 0; color: #8b949e; font-size: 14px;'>
            Real-time RUL Regression & Health State Classification Dashboard • Powered by Random Forest Mega-Models
        </p>
    </div>
""", unsafe_allow_html=True)

# Error handling if models aren't trained yet
if scaler is None or rf_reg is None or rf_class is None:
    st.error("🚨 Serialized model assets not found in `models/` folder!")
    st.info("Please verify that the model training pipeline (`train_and_save_models.py`) has finished execution.")
    st.stop()

# ----------------------------------------------------
# 5. SIDEBAR: SIMULATION SELECTION
# ----------------------------------------------------
st.sidebar.markdown("<h2 style='color: #58a6ff; font-size: 18px;'>🎛️ SYSTEM CONTROLLER</h2>", unsafe_allow_html=True)

mode = st.sidebar.radio(
    "Select Operating Mode",
    ["📁 Preloaded Flight Player", "🔧 Manual Telemetry Overrides"]
)

st.sidebar.markdown("---")

if mode == "📁 Preloaded Flight Player":
    st.sidebar.markdown("<p style='color: #8b949e; font-size: 12px;'>Choose a pre-configured engine unit to simulate flight playback.</p>", unsafe_allow_html=True)
    engine_choice = st.sidebar.selectbox(
        "Choose Preloaded Engine",
        [
            "FD001 - Engine 1 (Standard, Sea Level)",
            "FD002 - Engine 10 (Multi-Regime, HPC Wear)",
            "FD003 - Engine 24 (Sea Level, Multi-Fault)",
            "FD004 - Engine 40 (Multi-Regime, Multi-Fault)"
        ]
    )
    
    engine_df, engine_desc = get_preloaded_engine(engine_choice)
    max_cycles = len(engine_df)
    
    st.sidebar.info(f"**Description:** {engine_desc}")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("<h3 style='font-size: 14px;'>✈️ FLIGHT SIMULATION PLAYER</h3>", unsafe_allow_html=True)
    
    # Session state for playback control
    if "current_cycle_idx" not in st.session_state:
        st.session_state.current_cycle_idx = 1
        
    play_col, reset_col = st.sidebar.columns(2)
    play_clicked = play_col.button("▶️ Play Flight")
    reset_clicked = reset_col.button("🔄 Reset")
    
    if reset_clicked:
        st.session_state.current_cycle_idx = 1
        st.rerun()
        
    playback_speed = st.sidebar.slider("Playback Speed (s)", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
    
    cycle_slider = st.sidebar.slider(
        "Current Time Cycle",
        min_value=1,
        max_value=max_cycles,
        value=st.session_state.current_cycle_idx,
        key="cycle_slider_val"
    )
    st.session_state.current_cycle_idx = cycle_slider
    
    # Auto-play loop logic
    if play_clicked:
        while st.session_state.current_cycle_idx < max_cycles:
            st.session_state.current_cycle_idx += 1
            st.rerun()
            time.sleep(playback_speed)

else:  # Manual Telemetry Overrides Mode
    st.sidebar.markdown("<p style='color: #8b949e; font-size: 12px;'>Select a flight phase to prepopulate sensor inputs, then scale the degradation.</p>", unsafe_allow_html=True)
    regime_id = st.sidebar.selectbox(
        "Active Flight Phase / Operating Condition",
        [1, 2, 3, 4, 5, 6],
        format_func=lambda x: f"Regime {x}: {REGIME_DESCRIPTIONS[x].split('(')[0]}"
    )
    
    st.sidebar.info(f"**Regime Details:** {REGIME_DESCRIPTIONS[regime_id]}")
    
    st.sidebar.markdown("---")
    st.sidebar.success("💡 Prepopulate Baseline: Changing the preset regime above will automatically shift all telemetry sliders on the main control desk to standard values!")

# ----------------------------------------------------
# 6. FEATURE CONSTRUCTION & INFERENCE ENGINE
# ----------------------------------------------------
if mode == "📁 Preloaded Flight Player":
    # Extract feature vector from chosen row in the demo dataset
    selected_row = engine_df[engine_df['time_in_cycles'] == st.session_state.current_cycle_idx].iloc[0]
    
    # Feature vector for modeling (24 features, matching scale names)
    current_features = pd.DataFrame([selected_row[ACTIVE_FEATURES]])
    
    true_rul = int(selected_row['True_RUL'])
    current_cycle = int(selected_row['time_in_cycles'])
    total_life = int(selected_row['EOL'])
    current_regime = 1  # Standard placeholder, but dynamically inferred below:
    # Estimate current regime
    s1, s2, s3 = selected_row['setting_1'].round(1), selected_row['setting_2'].round(2), selected_row['TRA'].round(0)
    for r_id, m in REGIME_MEANS.items():
        if abs(s1-round(m['setting_1'], 1)) < 2.0 and abs(s2-round(m['setting_2'], 2)) < 0.1 and abs(s3-m['TRA']) < 5.0:
            current_regime = r_id
            break

else:  # Manual Overrides Mode
    st.markdown("""
        <div style='background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; margin-bottom: 25px;'>
            <h3 style='margin: 0 0 10px 0; color: #58a6ff; font-size: 18px;'>🔧 Engine Telemetry Control Desk</h3>
            <p style='margin: 0; color: #8b949e; font-size: 13px;'>
                Prepopulate baseline telemetry by switching flight phases in the sidebar, then manually adjust individual channel sliders below to see real-time predictions:
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    col_ctrl1, col_ctrl2 = st.columns(2)
    
    baseline = REGIME_MEANS[regime_id].copy()
    
    with col_ctrl1:
        st.markdown("<h4 style='color: #58a6ff; font-size: 14px; margin-bottom: 12px;'>🧭 Flight Control Settings</h4>", unsafe_allow_html=True)
        manual_cycle = st.slider("Time in Cycles (Age)", min_value=1, max_value=400, value=150, key="man_cycle")
        s1 = st.slider("Altitude (setting_1 in k ft)", min_value=0.0, max_value=45.0, value=float(baseline['setting_1']), step=0.1, key="man_s1")
        s2 = st.slider("Mach Speed (setting_2)", min_value=0.0, max_value=0.9, value=float(baseline['setting_2']), step=0.01, key="man_s2")
        s3 = st.slider("Throttle Resolver Angle (TRA %)", min_value=50.0, max_value=105.0, value=float(baseline['TRA']), step=1.0, key="man_s3")
        
    with col_ctrl2:
        st.markdown("<h4 style='color: #58a6ff; font-size: 14px; margin-bottom: 12px;'>🌡️ Active Thermal & Pressure Telemetry</h4>", unsafe_allow_html=True)
        t30 = st.slider("HPC Outlet Temp (T30 in Rankine)", min_value=1200.0, max_value=1700.0, value=float(baseline['T30']), step=1.0, key="man_t30")
        t50 = st.slider("LPT Outlet Temp (T50 in Rankine)", min_value=950.0, max_value=1550.0, value=float(baseline['T50']), step=1.0, key="man_t50")
        ps30 = st.slider("HPC Static Pressure (Ps30 in psia)", min_value=30.0, max_value=55.0, value=float(baseline['Ps30']), step=0.1, key="man_ps30")
        phi = st.slider("Fuel-to-Pressure Ratio (phi)", min_value=100.0, max_value=600.0, value=float(baseline['phi']), step=1.0, key="man_phi")
        htBleed = st.slider("Bleed Enthalpy (htBleed)", min_value=280.0, max_value=420.0, value=float(baseline['htBleed']), step=1.0, key="man_htbleed")
        
    # Construct telemetry vector from manual inputs
    baseline['time_in_cycles'] = manual_cycle
    baseline['setting_1'] = s1
    baseline['setting_2'] = s2
    baseline['TRA'] = s3
    baseline['T30'] = t30
    baseline['T50'] = t50
    baseline['Ps30'] = ps30
    baseline['phi'] = phi
    baseline['htBleed'] = htBleed
    
    # Proportional scaling for physical consistency in background sensors
    wear_frac = max(0.0, (t30 - REGIME_MEANS[regime_id]['T30']) / 85.0)
    baseline['Nc'] -= wear_frac * 150.0
    baseline['Nf'] -= wear_frac * 100.0
    baseline['W31'] += wear_frac * 5.0
    baseline['W32'] += wear_frac * 3.0
    
    current_features = pd.DataFrame([baseline])[ACTIVE_FEATURES]
    
    true_rul = None
    current_cycle = manual_cycle
    total_life = None
    current_regime = regime_id

# ----------------------------------------------------
# 7. RUN ML MODEL PREDICTIONS
# ----------------------------------------------------
# A. Apply standard scaling
current_scaled = pd.DataFrame(scaler.transform(current_features), columns=current_features.columns)

# B. Run classification inference (3 classes: 0=Healthy, 1=Warning, 2=Critical)
pred_class_id = int(rf_class.predict(current_scaled)[0])
pred_probs = rf_class.predict_proba(current_scaled)[0]

# C. Run RUL regression inference (clip output at 0 and max 125 cycles)
pred_rul = float(rf_reg.predict(current_scaled)[0])
pred_rul_clipped = max(0, min(125, pred_rul))

# ----------------------------------------------------
# 8. LAYOUT ROW 1: METRICS & HEAVY TARGET KPIS
# ----------------------------------------------------
col_kpi1, col_kpi2, col_kpi3 = st.columns([1.2, 1.2, 1])

with col_kpi1:
    st.markdown("<p style='color: #8b949e; margin-bottom: 2px; font-weight: bold;'>📉 PROGNOSTICS (REGRESSION)</p>", unsafe_allow_html=True)
    rul_delta = None
    if true_rul is not None:
        rul_error = pred_rul_clipped - true_rul
        rul_delta = f"Error: {rul_error:+.1f} cycles (True: {true_rul})"
        
    st.metric(
        label="Predicted Remaining Useful Life (RUL)",
        value=f"{pred_rul_clipped:.1f} Cycles",
        delta=rul_delta,
        delta_color="off" if rul_delta is None else ("normal" if abs(rul_error) < 10 else "inverse")
    )

with col_kpi2:
    st.markdown("<p style='color: #8b949e; margin-bottom: 2px; font-weight: bold;'>🩺 DIAGNOSTICS (CLASSIFICATION)</p>", unsafe_allow_html=True)
    
    # Map class prediction to glowing indicator badge
    if pred_class_id == 0:
        badge_html = '<div class="health-badge badge-good">🟢 Good / Healthy</div>'
        sub_desc = f"Stable operation • Conf: {pred_probs[0]:.1%}"
    elif pred_class_id == 1:
        badge_html = '<div class="health-badge badge-warn">🟡 Warning / Moderate</div>'
        sub_desc = f"Early wear signatures • Conf: {pred_probs[1]:.1%}"
    else:
        badge_html = '<div class="health-badge badge-critical">🚨 Critical / EOL Urgent</div>'
        sub_desc = f"Imminent failure risk! • Conf: {pred_probs[2]:.1%}"
        
    st.markdown(f"""
        <div style='background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; height: 100px; display: flex; flex-direction: column; justify-content: center;'>
            <div style='margin-bottom: 8px;'>{badge_html}</div>
            <div style='color: #8b949e; font-size: 12px; font-weight: 500;'>{sub_desc}</div>
        </div>
    """, unsafe_allow_html=True)

with col_kpi3:
    st.markdown("<p style='color: #8b949e; margin-bottom: 2px; font-weight: bold;'>🧭 JET FLIGHT METRICS</p>", unsafe_allow_html=True)
    st.markdown(f"""
        <div style='background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; height: 100px;'>
            <div style='color: #8b949e; font-size: 11px; text-transform: uppercase;'>Current Flight Phase</div>
            <div style='font-size: 16px; font-weight: bold; color: #58a6ff; margin-top: 4px;'>Regime {current_regime}</div>
            <div style='color: #8b949e; font-size: 11px; margin-top: 4px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;'>Cycle Runtime: <b>{current_cycle}</b></div>
        </div>
    """, unsafe_allow_html=True)

# ----------------------------------------------------
# 9. LAYOUT ROW 2: LIVE TELEMETRY GRAPHS (PLOTLY)
# ----------------------------------------------------
st.markdown("### 📊 Live Telemetry Sensors and Settings Analysis")

if mode == "📁 Preloaded Flight Player":
    # Plot historical trends leading up to the current cycle
    history_df = engine_df[engine_df['time_in_cycles'] <= current_cycle].copy()
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        # Plot Temp Sensors
        fig_temp = go.Figure()
        fig_temp.add_trace(go.Scatter(x=history_df['time_in_cycles'], y=history_df['T30'], mode='lines', name='T30 (HPC Outlet Temp)', line=dict(color='#ff7f0e', width=2)))
        fig_temp.add_trace(go.Scatter(x=history_df['time_in_cycles'], y=history_df['T50'], mode='lines', name='T50 (LPT Outlet Temp)', line=dict(color='#d62728', width=2)))
        fig_temp.update_layout(
            title=dict(text="Thermal Telemetry Trends (T30 & T50)", font=dict(color="#c9d1d9")),
            xaxis=dict(title=dict(text="Time (cycles)", font=dict(color="#8b949e")), gridcolor="#21262d", tickfont=dict(color="#8b949e")),
            yaxis=dict(title=dict(text="Temperature (Rankine)", font=dict(color="#8b949e")), gridcolor="#21262d", tickfont=dict(color="#8b949e")),
            paper_bgcolor="#161b22",
            plot_bgcolor="#161b22",
            margin=dict(l=40, r=40, t=40, b=40),
            legend=dict(font=dict(color="#c9d1d9")),
            height=300
        )
        st.plotly_chart(fig_temp, use_container_width=True)

    with col_chart2:
        # Plot Pressure & Bleed
        fig_press = go.Figure()
        fig_press.add_trace(go.Scatter(x=history_df['time_in_cycles'], y=history_df['Ps30'], mode='lines', name='Ps30 (HPC Outlet Static Press)', line=dict(color='#2ca02c', width=2)))
        fig_press.add_trace(go.Scatter(x=history_df['time_in_cycles'], y=history_df['htBleed'], mode='lines', name='htBleed (Bleed Enthalpy)', line=dict(color='#17becf', width=2)))
        fig_press.update_layout(
            title=dict(text="Pressure & Bleed Enthalpy Trends", font=dict(color="#c9d1d9")),
            xaxis=dict(title=dict(text="Time (cycles)", font=dict(color="#8b949e")), gridcolor="#21262d", tickfont=dict(color="#8b949e")),
            yaxis=dict(title=dict(text="Pressure (psia) / Enthalpy", font=dict(color="#8b949e")), gridcolor="#21262d", tickfont=dict(color="#8b949e")),
            paper_bgcolor="#161b22",
            plot_bgcolor="#161b22",
            margin=dict(l=40, r=40, t=40, b=40),
            legend=dict(font=dict(color="#c9d1d9")),
            height=300
        )
        st.plotly_chart(fig_press, use_container_width=True)

else:  # Manual overrides charts
    # For manual overrides, show a bar comparing the current scaled feature vector vs the baseline (healthy) regime averages!
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        # Temperature comparisons
        fig_temp_comp = go.Figure(data=[
            go.Bar(name='Baseline (Healthy)', x=['T24', 'T30', 'T50'], y=[REGIME_MEANS[regime_id]['T24'], REGIME_MEANS[regime_id]['T30'], REGIME_MEANS[regime_id]['T50']], marker_color='#30363d'),
            go.Bar(name='Current (Wear Adjusted)', x=['T24', 'T30', 'T50'], y=[current_features['T24'].iloc[0], current_features['T30'].iloc[0], current_features['T50'].iloc[0]], marker_color='#ff7f0e')
        ])
        fig_temp_comp.update_layout(
            title=dict(text="Thermal Telemetry Drift Comparison", font=dict(color="#c9d1d9")),
            xaxis=dict(gridcolor="#21262d", tickfont=dict(color="#8b949e")),
            yaxis=dict(title=dict(text="Temperature (Rankine)", font=dict(color="#8b949e")), gridcolor="#21262d", tickfont=dict(color="#8b949e")),
            paper_bgcolor="#161b22",
            plot_bgcolor="#161b22",
            barmode='group',
            margin=dict(l=40, r=40, t=40, b=40),
            legend=dict(font=dict(color="#c9d1d9")),
            height=300
        )
        st.plotly_chart(fig_temp_comp, use_container_width=True)
        
    with col_chart2:
        # Rotational speed degradation
        fig_speed_comp = go.Figure(data=[
            go.Bar(name='Baseline (Healthy)', x=['Nc (Core Speed)', 'Nf (Fan Speed)'], y=[REGIME_MEANS[regime_id]['Nc'], REGIME_MEANS[regime_id]['Nf']], marker_color='#30363d'),
            go.Bar(name='Current (Wear Adjusted)', x=['Nc (Core Speed)', 'Nf (Fan Speed)'], y=[current_features['Nc'].iloc[0], current_features['Nf'].iloc[0]], marker_color='#17becf')
        ])
        fig_speed_comp.update_layout(
            title=dict(text="Rotational Speeds Efficiency Decay", font=dict(color="#c9d1d9")),
            xaxis=dict(gridcolor="#21262d", tickfont=dict(color="#8b949e")),
            yaxis=dict(title=dict(text="RPM / Operational Speeds", font=dict(color="#8b949e")), gridcolor="#21262d", tickfont=dict(color="#8b949e")),
            paper_bgcolor="#161b22",
            plot_bgcolor="#161b22",
            barmode='group',
            margin=dict(l=40, r=40, t=40, b=40),
            legend=dict(font=dict(color="#c9d1d9")),
            height=300
        )
        st.plotly_chart(fig_speed_comp, use_container_width=True)

# ----------------------------------------------------
# 10. LAYOUT ROW 3: DETAILED TABULAR DATA AND CONTRIBUTIONS
# ----------------------------------------------------
col_tab1, col_tab2 = st.columns([1.5, 1])

with col_tab1:
    st.markdown("### 📋 Current Operational Telemetry Vector (Raw & Scaled)")
    
    # Create DataFrame of raw, baseline, and scaled features to show under the hood details
    comparison_dict = []
    for col in ACTIVE_FEATURES:
        raw_val = current_features[col].iloc[0]
        scaled_val = current_scaled[col].iloc[0]
        
        # Determine drift status
        reg_base = REGIME_MEANS[current_regime].get(col, raw_val)
        drift = raw_val - reg_base
        if abs(drift) < 0.001:
            status = "✅ Stable Baseline"
        elif drift > 0:
            status = f"🔺 +{drift:.2f} Drift"
        else:
            status = f"🔻 {drift:.2f} Drift"
            
        comparison_dict.append({
            "Sensor Feature": col,
            "Raw Telemetry": f"{raw_val:.4f}",
            "Operating Baseline": f"{reg_base:.4f}",
            "Physical Drift Status": status,
            "Scaled Model Input": f"{scaled_val:+.4f}"
        })
        
    st.dataframe(pd.DataFrame(comparison_dict), height=250, use_container_width=True)

with col_tab2:
    st.markdown("### 🌲 Top Feature Drivers (Model Importance)")
    
    # Calculate feature importances from the regressor
    importances = rf_reg.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Top 10 features
    top_features = [ACTIVE_FEATURES[i] for i in indices[:8]]
    top_importances = [importances[i] for i in indices[:8]]
    
    fig_imp = go.Figure(go.Bar(
        x=top_importances[::-1],
        y=top_features[::-1],
        orientation='h',
        marker_color='#58a6ff',
        text=[f"{val:.1%}" for val in top_importances[::-1]],
        textposition='auto'
    ))
    
    fig_imp.update_layout(
        title=dict(text="Most Influential Engine Metrics for RUL", font=dict(color="#c9d1d9")),
        xaxis=dict(title=dict(text="Importance Weight", font=dict(color="#8b949e")), gridcolor="#21262d", tickfont=dict(color="#8b949e")),
        yaxis=dict(gridcolor="#21262d", tickfont=dict(color="#8b949e")),
        paper_bgcolor="#161b22",
        plot_bgcolor="#161b22",
        margin=dict(l=40, r=40, t=40, b=40),
        height=250
    )
    st.plotly_chart(fig_imp, use_container_width=True)
