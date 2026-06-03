# Ukraine Drone Forecast — Local Bazzite Version

Same model we built together in our Claude conversation, running entirely on your own computer.

## What this does

Live dashboard that:
- Predicts Russian drone strikes by Ukrainian oblast for the next week
- Shows them on a real map of Ukraine with circles sized by predicted volume
- Lets you record observed strikes as they happen
- Updates instantly when you change tempo/regime parameters
- Works completely offline after install

## Installation on Bazzite (5 minutes)

Open a terminal:

```bash
cd ~/Downloads
unzip drone_app.zip
cd drone_app
bash install.sh
```

The install script:
- Creates a Python virtual environment (doesn't touch your system Python)
- Installs streamlit, pandas, numpy, matplotlib, scikit-learn, geopandas
- Doesn't require root/sudo access

If `install.sh` errors out, run the steps manually:

```bash
cd drone_app
python3 -m venv venv
source venv/bin/activate
pip install streamlit pandas numpy matplotlib scipy scikit-learn geopandas
```

## Running the app

```bash
cd drone_app
bash run.sh
```

Or manually:
```bash
source venv/bin/activate
streamlit run app.py
```

A browser tab opens automatically at http://localhost:8501

Press **Ctrl+C** in the terminal to stop the app.

## Using the dashboard

### Sidebar controls
- **Russian daily capacity**: drones Russia can produce per day (~210 based on ISIS data)
- **Tempo factor**: 0.1 = ceasefire, 1.0 = full war, 1.5 = surge
- **Drones already used**: subtract launches earlier in the week
- **Low-tempo regime**: boosts border oblasts when checked (matches ceasefire pattern)

### Recording observations
Use the form at the bottom to add new strike observations as they happen. Data is saved to `data/observations.csv` and persists between sessions.

### Reading the map
- **Big red circles in the east** = expected high-strike oblasts (Kharkiv, Donetsk, etc.)
- **Yellow circles in the west** = expected low-strike oblasts (Lviv, Zakarpattia)
- **Gold star** = Kyiv (capital)
- **Numbers inside circles** = predicted drone count for the week

## The model in 30 seconds

Each oblast gets a targeting weight:

    weight = 0.35 × energy_score + 1.5 × exp(-distance/400) + 0.25 × population

Oblasts >700 km from the border get a 0.4× penalty (Shaheds rarely reach that far).

Weights normalize to shares. Shares × weekly production budget = predicted drones per oblast.

In low-tempo mode, border oblasts get a 30% boost (Russia favors cheap, close-in targets during ceasefires).

This is the same regression we worked through together. It's not magic — it's geography (distance) × strategic value (energy + population) × budget constraint.

## File structure

    drone_app/
    ├── app.py                  # Main Streamlit dashboard
    ├── install.sh              # One-time setup script
    ├── run.sh                  # Daily startup script
    ├── README.md               # This file
    └── data/
        ├── oblast_features.csv      # 24 Ukraine regions + features
        ├── observations.csv         # Strikes you've recorded
        ├── weekly_forecast.csv      # Last computed forecast
        └── updated_forecast.csv     # Updated forecast file

## Updating the data

To add a new day's observations:
1. Open the dashboard
2. Scroll to "Add New Observation" form
3. Select oblast, count, date
4. Click "Save observation"
5. The map updates immediately

Data is appended to `data/observations.csv` and you can edit that file directly if needed.

## Troubleshooting

**"streamlit: command not found"**
The venv isn't activated. Run: `source venv/bin/activate`

**Map shows but countries look wrong**
The geopandas shapefile path might be different on your system. The fallback (no shapefile) still works.

**Port 8501 already in use**
Run: `streamlit run app.py --server.port 8502`

**Bazzite says "command not found: python3"**
This shouldn't happen — Bazzite ships with Python. Try: `which python` and use that path.

## Connection to our Claude conversation

This is the local version of what we tried to build on Palantir Foundry. After hitting verification errors, package install issues, egg_info errors, and path permission errors over several hours, we agreed to ship this version first.

Once you're comfortable with the local version and want to try Foundry again, the conversation transcript has all the steps and fixes we discovered. The Foundry version would give you scheduled pipelines, multi-user access, and the Ontology layer — but the analytical model is identical.

## What's NOT included (yet)

- Daily auto-scraper from Ukrainian Air Force Telegram
- Email alerts when forecast changes significantly
- US-Iran attrition tracking (separate model we built)
- Finland defense analysis
- Equilibrium calculations

These can all be added as additional Streamlit pages. The structure is set up to support them.

## Built with Claude on May 13, 2026
