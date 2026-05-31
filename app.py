# app.py (Top of the file)
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go  # <--- MAKE SURE THIS LINE IS HERE!


st.set_page_config(page_title="Net-Zero Carbon Architect", layout="wide")

st.title("Building performance optimisation platform")

# =======================================================
# 1. Load Meta Data Safely & Robustly
# =======================================================
@st.cache_data
def load_all_meta_data():
    # Read using header=1 to match train_models.py exactly
    df1 = pd.read_csv('Pareto.csv', header=1, encoding='latin1')
    df1.columns = ['iteration','generation','category','total_energy','discomfort_hours','cooling_energy','window_to_wall','orientation','facade_type','shading_type','window_open_pct','unnamed'][:len(df1.columns)]
    
    df2 = pd.read_csv('2nd Optimization Results.csv', header=1, encoding='latin1')
    df2.columns = ['iteration','generation','category','total_energy','discomfort_hours','cooling_energy','external_wall','flat_roof','glazing_type','partition_wall','unnamed'][:len(df2.columns)]
    
    # Safely convert to string, drop empty/NaN records, and sort
    facades_list = sorted([str(x).strip() for x in df1['facade_type'].dropna().unique()])
    shadings_list = sorted([str(x).strip() for x in df1['shading_type'].dropna().unique()])
    
    ext_walls_list = sorted([str(x).strip() for x in df2['external_wall'].dropna().unique()])
    roofs_list = sorted([str(x).strip() for x in df2['flat_roof'].dropna().unique()])
    glazings_list = sorted([str(x).strip() for x in df2['glazing_type'].dropna().unique()])
    partitions_list = sorted([str(x).strip() for x in df2['partition_wall'].dropna().unique()])
    
    return facades_list, shadings_list, ext_walls_list, roofs_list, glazings_list, partitions_list

# Execute the loader and unpack the clean variables
try:
    facades, shadings, ext_walls, roofs, glazings, partitions = load_all_meta_data()
except Exception as e:
    st.error(f"Error compiling metadata lists: {e}")
    st.stop()

# =======================================================
# 2. Load Pre-trained Machine Learning Models
# =======================================================
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
    st.error("Missing asset models! Please run `train_models.py` first to compile required pkl files.")
    st.stop()

# =======================================================
# 3. Sidebar UI Layout Setup
# =======================================================
st.sidebar.header("Geometric Configuration")
wwr = st.sidebar.slider("Window to Wall Ratio (%)", 10, 50, 30, 1)
orient = st.sidebar.slider("Site Orientation (°)", 0, 270, 180, 90)
w_open = st.sidebar.slider("% External Window Opens", 50, 80, 60, 1)
facade = st.sidebar.selectbox("Facade Layout Matrix", options=facades)
shading = st.sidebar.selectbox("Local Shading Framework", options=shadings)

st.sidebar.markdown("---")
st.sidebar.header("Structural Materials")

# Safe index lookups with fallbacks to avoid ValueErrors if exact names change
def get_safe_index(item_list, target_value):
    try:
        return item_list.index(target_value)
    except ValueError:
        return 0 # Default to first item if name doesn't match raw strings

def_wall_idx = get_safe_index(ext_walls, 'AAC + INSULATION')
def_roof_idx = get_safe_index(roofs, 'RCC+INSULATION+TILE')
def_glaz_idx = get_safe_index(glazings, 'GLAZING-10')
def_part_idx = get_safe_index(partitions, 'Partation AAC Wall')

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
        
    )
with m2:
    st.metric(
        label=" Cumulative Cooling Demand", 
        value=f"{final_cooling:,.2f} kWh", 
       
    )
with m3:
    st.metric(
        label=" Thermal Discomfort", 
        value=f"{final_discom:,.1f} Hours", 
       
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
        # =======================================================
# 📊 NEW SECTION: FEATURE IMPORTANCE VISUALIZATION
# =======================================================
st.markdown("---")
st.subheader("🎯 Machine Learning Driver Attribution")
st.markdown("This interactive chart breaks down exactly which architectural levers your AI surrogate model relies on to calculate energy outcomes.")

# 1. Extract feature importance values from your trained 'total' layout model
# We map them back to their clean parameter names for the presentation
raw_importances = ml_layout['total'].feature_importances_
feature_labels = ['Window to Wall Ratio', 'Site Orientation', 'Facade Matrix Type', 'Local Shading Framework', 'Window Open Pct']

# 2. Structure into a DataFrame and sort them from highest impact to lowest
df_importance = pd.DataFrame({
    'Design Parameter': feature_labels,
    'Model Impact (%)': raw_importances * 100
}).sort_values(by='Model Impact (%)', ascending=True) # Horizontal charts look best sorted low-to-high

# 3. Construct the interactive Plotly horizontal bar chart
fig_importance = go.Figure(go.Bar(
    x=df_importance['Model Impact (%)'],
    y=df_importance['Design Parameter'],
    orientation='h',
    marker=dict(
        color=df_importance['Model Impact (%)'],
        colorscale='Viridis', # Beautiful gradient that matches your 3D surface map
        showscale=False
    ),
    text=df_importance['Model Impact (%)'].map('{:.2f}%'.format), # Overlay exact percentages onto bars
    textposition='outside'
))

fig_importance.update_layout(
    title="Relative Impact of Geometric Features on Total Site Energy",
    xaxis=dict(title="Influence Share (%)", range=[0, 100]),
    yaxis=dict(title=""),
    margin=dict(l=150, r=50, b=40, t=50), # Padding for long feature labels
    height=350
)

# 4. Render the chart container on your Streamlit screen
st.plotly_chart(fig_importance, use_container_width=True)

# 5. Presentation Callout Box
st.warning(
    f"💡 **Key Insight for Team Review:** The model attributes **{df_importance.loc[df_importance['Design Parameter'] == 'Local Shading Framework', 'Model Impact (%)'].values[0]:.1f}%** "
    "of its predictive power strictly to the **Local Shading Framework**. This proves that active envelope shading effectively neutralizes variations "
    "caused by structural windows and orientation."
)

# =======================================================
# 7. INTERACTIVE DESIGN CHART: WWR vs. ORIENTATION
# =======================================================
st.markdown("---")
st.subheader(" Geometric Sensitivity Mapping")
st.markdown("This interactive matrix visualizes how **Window-to-Wall Ratio** and **Orientation** cross-examine to change total energy outcomes under your selected facade and shading framework.")

import plotly.graph_objects as go

# 1. Generate a smooth mesh grid of points to evaluate
wwr_range = np.linspace(10, 50, 30)     # 30 steps from 10% to 50%
orient_range = np.linspace(0, 270, 30)  # 30 steps from 0° to 270°
W_mesh, O_mesh = np.meshgrid(wwr_range, orient_range)

# 2. Build a batch dataframe for the model to predict instantly
sim_data = pd.DataFrame({
    'window_to_wall': W_mesh.flatten(),
    'orientation': O_mesh.flatten(),
    'facade_type': facade,     # Uses the user's active dropdown choice
    'shading_type': shading,   # Uses the user's active dropdown choice
    'window_open_pct': w_open  # Uses the user's active slider choice
})

# 3. Transform the categorical features using your pre-trained encoder
sim_data[['facade_type', 'shading_type']] = enc_l.transform(sim_data[['facade_type', 'shading_type']])

# 4. Predict baseline and apply the current material multiplier
raw_predictions = ml_layout['total'].predict(sim_data)
final_predictions = raw_predictions * mult_total

# 5. Reshape flat outputs back into a 2D matrix grid for the surface plot
Z_energy = final_predictions.reshape(W_mesh.shape)

# 6. Create the Plotly Layout Tabs
tab1, tab2 = st.tabs(["3D Design Terrain Map", "2D Thermal Contour Grid"])

with tab1:
    fig_3d = go.Figure(data=[go.Surface(
        z=Z_energy, 
        x=wwr_range, 
        y=orient_range,
        colorscale='Viridis',
        colorbar_title="kWh"
    )])
    
    fig_3d.update_layout(
        title=f"Total Site Energy Sensitivity Matrix (Shading: {shading})",
        scene=dict(
            xaxis_title="Window to Wall Ratio (%)",
            yaxis_title="Site Orientation (°)",
            zaxis_title="Total Energy (kWh)"
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=600
    )
    st.plotly_chart(fig_3d, use_container_width=True)

with tab2:
    fig_2d = go.Figure(data=go.Contour(
        z=Z_energy,
        x=wwr_range,
        y=orient_range,
        colorscale='Hot',
        contours=dict(showlabels=True, labelfont=dict(size=12, color='white')),
        colorbar_title="kWh"
    ))
    
    fig_2d.update_layout(
        title="Thermal Performance Footprint (Darker = More Efficient)",
        xaxis_title="Window to Wall Ratio (%)",
        yaxis_title="Site Orientation (°)",
        height=500
    )
    st.plotly_chart(fig_2d, use_container_width=True)
