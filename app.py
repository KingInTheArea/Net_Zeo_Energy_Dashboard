# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Net-Zero Carbon Architect", layout="wide")

st.title("Building performance optimisation platform")
# st.markdown("This system chains geometric layouts with structural material thermal scaling to predict final energy outcomes.")

# 1. Load Meta Data for Dropdown Selectors
@st.cache_data
def load_all_meta_data():
    df1 = pd.read_csv('Pareto.csv', header=[0, 1], encoding='latin1')
    df1.columns = ['iteration','generation','category','total_energy','discomfort_hours','cooling_energy','window_to_wall','orientation','facade_type','shading_type','window_open_pct','unnamed'][:len(df1.columns)]
    
    df2 = pd.read_csv('2nd Optimization Results.csv', header=[0, 1], encoding='latin1')
    df2.columns = ['iteration','generation','category','total_energy','discomfort_hours','cooling_energy','external_wall','flat_roof','glazing_type','partition_wall','unnamed'][:len(df2.columns)]
    
    return (
        sorted(df1['facade_type'].unique()), sorted(df1['shading_type'].unique()),
        sorted(df2['external_wall'].unique()), sorted(df2['flat_roof'].unique()),
        sorted(df2['glazing_type'].unique()), sorted(df2['partition_wall'].unique())
    )

facades, shadings, ext_walls, roofs, glazings, partitions = load_all_meta_data()

# 2. Load Pre-trained Machine Learning Models
@st.cache_resource
def load_all_assets():
    with open('encoder_layout.pkl', 'rb') as f: enc_l = pickle.load(f)
    with open('encoder_materials.pkl', 'rb') as f: enc_m = pickle.load(f)
    
    ml_layout = {
        'total': pickle.load(open('model_layout_total_energy.pkl', 'rb')),
        'cool': pickle.load(open('model_layout_cooling_energy.pkl', 'rb')),
        'discom': pickle.load(open('model_layout_discomfort_hours.pkl', 'rb'))
    }
    ml_mat = {
        'total': pickle.load(open('model_mat_total_energy.pkl', 'rb')),
        'cool': pickle.load(open('model_mat_cooling_energy.pkl', 'rb')),
        'discom': pickle.load(open('model_mat_discomfort_hours.pkl', 'rb'))
    }
    return enc_l, enc_m, ml_layout, ml_mat

try:
    enc_l, enc_m, ml_layout, ml_mat = load_all_assets()
except FileNotFoundError:
    st.error(" run `train_models.py`")
    st.stop()

# 3. Sidebar UI: Two Distinct Sections In One Streamlined View
st.sidebar.header("Geometric Configuration")
wwr = st.sidebar.slider("Window to Wall Ratio (%)", 10, 50, 30, 5)
orient = st.sidebar.slider("Site Orientation (°)", 0, 270, 180, 90)
w_open = st.sidebar.slider("% External Window Opens", 50, 80, 60, 10)
facade = st.sidebar.selectbox("Facade Layout Matrix", options=facades)
shading = st.sidebar.selectbox("Local Shading Framework", options=shadings)

st.sidebar.markdown("--")
st.sidebar.header("Structural Materials")
st.sidebar.info("optimised materials selected")

# Automatically default dropdown selections to the best performing options
def_wall_idx = ext_walls.index('AAC + INSULATION')
def_roof_idx = roofs.index('RCC+INSULATION+TILE')
def_glaz_idx = glazings.index('GLAZING-10')
def_part_idx = partitions.index('Partation AAC Wall')

wall = st.sidebar.selectbox("External Wall Assembly", options=ext_walls, index=def_wall_idx)
roof = st.sidebar.selectbox("Flat Roof Insulation", options=roofs, index=def_roof_idx)
glazing = st.sidebar.selectbox("Window Glazing Glass", options=glazings, index=def_glaz_idx)
partition = st.sidebar.selectbox("Internal Partitioning Wall", options=partitions, index=def_part_idx)

# 4. CHANNED MATHEMATICAL COMPUTATION ENGINE
# Stage A: Predict Layout Baseline Energy
inp_layout = pd.DataFrame([{'window_to_wall': wwr, 'orientation': orient, 'facade_type': facade, 'shading_type': shading, 'window_open_pct': w_open}])
inp_layout[['facade_type', 'shading_type']] = enc_l.transform(inp_layout[['facade_type', 'shading_type']])

geo_base_total = ml_layout['total'].predict(inp_layout)[0]
geo_base_cooling = ml_layout['cool'].predict(inp_layout)[0]
geo_base_discom = ml_layout['discom'].predict(inp_layout)[0]

# Stage B: Predict Material Selection and evaluate vs standard uninsulated base configuration (Row 0)
inp_mat = pd.DataFrame([{'external_wall': wall, 'flat_roof': roof, 'glazing_type': glazing, 'partition_wall': partition}])
inp_mat[['external_wall', 'flat_roof', 'glazing_type', 'partition_wall']] = enc_m.transform(inp_mat[['external_wall', 'flat_roof', 'glazing_type', 'partition_wall']])

mat_pred_total = ml_mat['total'].predict(inp_mat)[0]
mat_pred_cooling = ml_mat['cool'].predict(inp_mat)[0]
mat_pred_discom = ml_mat['discom'].predict(inp_mat)[0]

# Normalization constants (the uninsulated base case simulation value)
BASE_MAT_TOTAL = 11160.17
BASE_MAT_COOL = 23620.97
BASE_MAT_DISCOM = 1450.65

# Generate multipliers
mult_total = mat_pred_total / BASE_MAT_TOTAL
mult_cooling = mat_pred_cooling / BASE_MAT_COOL
mult_discom = mat_pred_discom / BASE_MAT_DISCOM

# Calculate Final Cumulative Chain Output
final_total = geo_base_total * mult_total
final_cooling = geo_base_cooling * mult_cooling
final_discom = geo_base_discom * mult_discom

# 5. USER INTERFACE METRICS PRESENTATION
st.subheader("Cumulative System Performance Metrics")
m1, m2, m3 = st.columns(3)

with m1:
    st.metric(
        label=" Cumulative Site Energy", 
        value=f"{final_total:,.2f} kWh", 
        delta=f"{(mult_total-1)*100:+.1f}% Material Effect"
    )
with m2:
    st.metric(
        label=" Cumulative Cooling Demand", 
        value=f"{final_cooling:,.2f} kWh", 
        delta=f"{(mult_cooling-1)*100:+.1f}% Material Effect",
        delta_color="inverse"
    )
with m3:
    st.metric(
        label=" Thermal Discomfort", 
        value=f"{final_discom:,.1f} Hours", 
        delta=f"{(mult_discom-1)*100:+.1f}% Material Effect",
        delta_color="inverse"
    )

# 6. EXPLANATION CONTAINER FOR TEAM MEETINGS
st.markdown("---")
with st.expander("How this calculation stacks"):
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"""
        ** Geometric Baseline Performance**
        * Your choices for Orientation, WWR, and Shading established a spatial baseline.
        * **Baseline Energy Assessment:** `{geo_base_total:,.2f} kWh`
        """)
    with col_b:
        st.markdown(f"""
        ** Material Thermal Envelope Correction**
        * The selected structural materials change structural efficiency by a multiplier of **{mult_total:.2f}x** compared to standard brick.
        * **Optimized Thermal Outcome:** `{final_total:,.2f} kWh`
        """)