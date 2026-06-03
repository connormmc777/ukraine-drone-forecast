"""
Ukraine Drone Forecast Dashboard
=================================
Local Streamlit app that runs the full forecasting model.
Run with: streamlit run app.py

This is the local version of what we tried to build on Palantir Foundry.
Same regression, same predictions, runs entirely on your computer.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime, timedelta
from pathlib import Path

st.set_page_config(
    page_title="Ukraine Drone Forecast",
    page_icon="🎯",
    layout="wide",
)

DATA_DIR = Path(__file__).parent / "data"


# ============== LOAD DATA ==============
@st.cache_data
def load_oblasts():
    return pd.read_csv(DATA_DIR / "oblast_features.csv")


@st.cache_data
def load_observations():
    return pd.read_csv(DATA_DIR / "observations.csv")


# ============== REGRESSION MODEL ==============
def compute_targeting_weights(oblasts):
    """The structural regression coefficients applied to features."""
    oblasts = oblasts.copy()
    oblasts['weight'] = (
        0.35 * oblasts['energy'] +
        1.5 * np.exp(-oblasts['border_dist'] / 400) +
        0.25 * oblasts['pop']
    )
    oblasts.loc[oblasts['border_dist'] > 700, 'weight'] *= 0.4
    oblasts['share'] = oblasts['weight'] / oblasts['weight'].sum()
    return oblasts


def generate_weekly_forecast(oblasts, weekly_budget, low_tempo=True):
    """Distribute weekly budget across oblasts using regression shares."""
    oblasts = compute_targeting_weights(oblasts)
    oblasts['adj_share'] = oblasts['share']

    # Regime adjustment: border bias under low tempo
    if low_tempo:
        oblasts.loc[oblasts['border_dist'] < 100, 'adj_share'] *= 1.30
        oblasts.loc[oblasts['energy'] >= 7, 'adj_share'] *= 0.90
    oblasts['adj_share'] = oblasts['adj_share'] / oblasts['adj_share'].sum()
    oblasts['predicted_week'] = (oblasts['adj_share'] * weekly_budget).round(0)

    return oblasts.sort_values('predicted_week', ascending=False)


# ============== UI ==============
st.title("🎯 Ukraine Drone Strike Forecast")
st.markdown(
    "**Regression-based prediction with production budget constraints.** "
    "Same model we built together in our Claude conversation."
)

# Sidebar controls
with st.sidebar:
    st.header("Forecast Parameters")

    russian_daily_capacity = st.slider(
        "Russian daily production capacity",
        min_value=100, max_value=500, value=210, step=10,
        help="Estimated drones Russia can produce per day. ISIS reports ~210/day."
    )

    tempo_factor = st.slider(
        "Tempo factor (0=ceasefire, 1=full war)",
        min_value=0.1, max_value=1.5, value=0.50, step=0.05,
        help="Fraction of production capacity actually launched"
    )

    weekly_budget = int(russian_daily_capacity * 7 * tempo_factor)
    st.metric("Weekly drone budget", f"{weekly_budget:,}")

    already_used = st.number_input(
        "Drones already used this week",
        min_value=0, max_value=weekly_budget, value=70, step=10,
        help="Subtract observed launches earlier in the week"
    )

    remaining_budget = weekly_budget - already_used
    st.metric("Remaining for forecast", f"{remaining_budget:,}")

    low_tempo = st.checkbox("Low-tempo regime (ceasefire-like)", value=True,
                             help="Border oblasts get boosted share under low tempo")

    st.divider()
    st.caption("Built locally on Bazzite. No cloud required.")

# Load and forecast
oblasts = load_oblasts()
forecast = generate_weekly_forecast(oblasts, remaining_budget, low_tempo=low_tempo)
observations = load_observations()

# ============== KPI ROW ==============
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total predicted (this week)", f"{int(forecast['predicted_week'].sum())}")
with col2:
    top = forecast.iloc[0]
    st.metric(f"Top target: {top['oblast']}", f"{int(top['predicted_week'])} drones")
with col3:
    st.metric("Observations recorded", len(observations))
with col4:
    obs_total = observations['observed_drones'].sum()
    st.metric("Total observed", f"{obs_total:,}")

# ============== MAP ==============
st.subheader("Predicted Strikes by Oblast")

# Build the map
fig, ax = plt.subplots(figsize=(14, 9))
ax.set_facecolor('#d4ebf2')

# Try to load shapefile, fall back to simple polygon
try:
    import geopandas as gpd
    SHAPEFILE = '/usr/local/lib/python3.12/dist-packages/pyogrio/tests/fixtures/naturalearth_lowres/naturalearth_lowres.shp'
    world = gpd.read_file(SHAPEFILE)

    # Plot countries
    for country_name, color, edge in [
        ('Poland', '#e8e8e8', 'gray'),
        ('Romania', '#e8e8e8', 'gray'),
        ('Hungary', '#e8e8e8', 'gray'),
        ('Slovakia', '#e8e8e8', 'gray'),
        ('Moldova', '#e8e8e8', 'gray'),
        ('Russia', '#ffd4d4', '#cc0000'),
        ('Belarus', '#ffe0e0', '#cc0000'),
        ('Ukraine', '#fff8dc', 'black'),
    ]:
        country = world[world['name'] == country_name]
        if not country.empty:
            country.plot(ax=ax, facecolor=color, edgecolor=edge,
                         linewidth=1.5 if country_name == 'Ukraine' else 0.8,
                         alpha=0.9 if country_name == 'Ukraine' else 0.7)
except Exception as e:
    st.warning(f"Could not load proper shapefile ({e}). Using simplified borders.")

# Country labels
ax.text(43, 51, 'RUSSIA', fontsize=20, fontweight='bold',
        color='#990000', alpha=0.6, ha='center')
ax.text(28, 53.5, 'BELARUS', fontsize=12, fontweight='bold',
        color='#990000', alpha=0.5, ha='center')
ax.text(32, 49, 'UKRAINE', fontsize=28, fontweight='bold',
        color='#003d7a', alpha=0.13, ha='center')

# Plot oblast circles
norm = mcolors.PowerNorm(gamma=0.6, vmin=0, vmax=forecast['predicted_week'].max())
cmap = plt.cm.YlOrRd

for _, ob in forecast.sort_values('predicted_week').iterrows():
    pred = ob['predicted_week']
    if pred <= 0:
        continue
    size = 250 + (pred / forecast['predicted_week'].max()) * 2500
    color = cmap(norm(pred))
    ax.scatter(ob['lon'], ob['lat'], s=size, c=[color],
               edgecolor='black', linewidth=1.3, zorder=5, alpha=0.92)
    if pred >= 30:
        ax.text(ob['lon'], ob['lat'], f'{int(pred)}',
                ha='center', va='center', fontsize=10,
                fontweight='bold', color='white', zorder=6)
    elif pred >= 10:
        ax.text(ob['lon'], ob['lat'], f'{int(pred)}',
                ha='center', va='center', fontsize=8,
                fontweight='bold', color='black', zorder=6)

# Label top oblasts
for _, ob in forecast.head(8).iterrows():
    name = ob['oblast'].replace(' Oblast', '').replace(' City', '')
    ax.annotate(name, xy=(ob['lon'], ob['lat']),
                xytext=(ob['lon']+0.8, ob['lat']+0.5),
                fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor='gray', alpha=0.9))

# Kyiv star
kyiv = oblasts[oblasts['oblast'] == 'Kyiv City'].iloc[0]
ax.scatter([kyiv['lon']], [kyiv['lat']+0.1], marker='*', s=400,
           color='gold', edgecolor='black', linewidth=1.5, zorder=8)

ax.set_xlim(20, 45)
ax.set_ylim(43.5, 55)
ax.set_xlabel('Longitude (°E)')
ax.set_ylabel('Latitude (°N)')
ax.set_aspect(1.45)
ax.grid(True, alpha=0.25, linestyle=':')
ax.set_title(f'Predicted Russian Drone Strikes by Oblast\n'
             f'~{int(forecast["predicted_week"].sum())} drones remaining in weekly budget',
             fontsize=13, fontweight='bold')

# Colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, ax=ax, fraction=0.025, pad=0.02, label='Predicted drones/week')

st.pyplot(fig)

# ============== TABLE ==============
st.subheader("Top 15 Targeted Oblasts")
display_df = forecast[['oblast', 'border_dist', 'energy', 'pop', 'share', 'predicted_week']].head(15)
display_df.columns = ['Oblast', 'Border km', 'Energy', 'Pop (M)', 'Share %', 'Predicted Week']
display_df['Share %'] = (display_df['Share %'] * 100).round(1)
st.dataframe(display_df, use_container_width=True, hide_index=True)

# ============== ADD OBSERVATION ==============
st.subheader("Add New Observation")
with st.form("add_obs"):
    col1, col2, col3 = st.columns(3)
    with col1:
        obs_oblast = st.selectbox("Oblast", oblasts['oblast'].tolist())
    with col2:
        obs_count = st.number_input("Drones observed", min_value=0, value=0)
    with col3:
        obs_date = st.date_input("Date", value=datetime.now().date())

    submitted = st.form_submit_button("Save observation")
    if submitted:
        new_row = pd.DataFrame([{
            'observation_date': obs_date.isoformat(),
            'oblast': obs_oblast,
            'observed_drones': obs_count,
            'source': 'Manual entry',
        }])
        new_row.to_csv(DATA_DIR / "observations.csv", mode='a', header=False, index=False)
        st.success(f"Saved: {obs_count} drones in {obs_oblast} on {obs_date}")
        st.cache_data.clear()

# ============== OBSERVATIONS LOG ==============
st.subheader("Recent Observations")
st.dataframe(observations.sort_values('observation_date', ascending=False),
             use_container_width=True, hide_index=True)

# ============== FOOTER ==============
st.divider()
st.caption(
    "Model: Negative Binomial regression (approximated with weighted features). "
    "Weights: 0.35×energy + 1.5×exp(-distance/400) + 0.25×population, "
    "with 0.4× penalty for oblasts >700 km from border. "
    "Production constraint enforced via weekly budget. "
    "Built with Claude on May 13, 2026."
)
