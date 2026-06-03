# Conversation Summary: Ukraine Drone Forecast Project

Date: May 13, 2026
Session length: Extended conversation building drone forecast models

## What we built together

A statistical model predicting Russian drone strikes against Ukrainian regions, extended into:
- Two-front war analysis (Russia + Finland)
- US-Iran attrition modeling
- Global drone production equilibrium
- Football analogies for the same math
- Defensive net calculations
- Single-use vs reusable interceptor economics
- Adversarial adaptation and Goodhart's Law
- Production-budget constraint forecasting

## The core regression equation

For oblast i, expected drones launched:

  w_i = 0.35 × energy_score + 1.5 × exp(-distance/400) + 0.25 × population_M

For oblasts >700 km from border: w_i × 0.4

Share: s_i = w_i / sum(w_j)

Weekly prediction: drones_i = s_i × weekly_national_total

## Key numbers from our forecast

Locked predictions for May 11-17, 2026:
- Total: ~630 drones (after ceasefire underspend)
- Top targets: Kyiv City (45), Dnipropetrovsk (45), Kharkiv (50)
- Russian production ceiling: ~210/day = ~1,470/week max
- Ceasefire-adjusted realistic budget: ~700/week

## Observations made during conversation

May 8: 67 drones (pre-ceasefire violation)
May 9: 43 drones + 1 Iskander (ceasefire start)
May 10: 27 drones (ceasefire active)
Total observed: ~137 drones over 3 nights

Original model predicted ~3,500 for same period. Reality 25× lower because of ceasefire.

Spatial accuracy still strong: Pearson r = +0.75 between predicted and observed share even with total volume way off.

## Key insights from conversation

1. **Production budget binds**: Russia ~210 drones/day, no stockpiling visible. Can't suddenly attack 2 fronts.

2. **Geography is destiny**: distance from border drives 75% of variance. Western Ukraine effectively safe.

3. **Cost-exchange favors defenders**: Ukraine using Gepards (reusable, $30/round) against Shaheds ($30K) is 1000:1 favorable. Patriot use is 111:1 unfavorable.

4. **Single-use vs reusable distinction critical**:
   - Attacker pays full drone cost every launch
   - Defender pays ammunition cost for reusable systems
   - Defender pays full replacement only when platforms destroyed
   - This is why Ukraine survives despite production disadvantage

5. **Adversarial adaptation bounded by physics**: Russia can shift timing within a week but cannot exceed production capacity, cannot easily change geography.

6. **Aggregation reduces adversarial noise**: weekly forecasts beat daily ones because Russia can move drones Tuesday→Friday but cannot change weekly totals without violating production.

7. **Two-front war takes 12-18 months to build capacity for**: Russia can't attack Finland at scale until 2027+

8. **Equilibrium math** for defender production:
   P_def = K × (P_atk × N_atk / N_def) × (C_atk / C_def)
   
   Taiwan defense requires 535K/yr allied production. Current ~110K. Gap: 5× scaling needed.

9. **China is the structural keystone**: Even adversaries (Russia, Iran) depend on Chinese components 70-80%. China itself depends on Taiwan chips. The whole system is interconnected.

10. **Foundry vs local tradeoff**: Foundry has steep learning curve. Local Streamlit version gives 80% of value in 15 minutes. We built the local version after struggling with Foundry setup.

## Foundry attempt status

Got partway through setup:
- ✅ Signed up for AIP Developer Tier
- ✅ Got enrollment at drones.usw-16.palantirfoundry.com
- ✅ Created project folder Ukraine_Forecast
- ✅ Created package folder ukraine_forecast/
- ✅ Fixed setup.py syntax errors
- ✅ Added statsmodels and scikit-learn to meta.yaml
- ✅ Pipeline.py imports working
- ✅ Transform decorator recognized
- ❌ Stuck on: output path permission (need to use paths inside repo's project)

To resume Foundry work:
1. Find repository's project path from breadcrumb
2. Use relative Output("name") instead of absolute paths
3. Upload historical_strikes.csv to Foundry Files
4. Match Input() path to the uploaded dataset location

## Files in this package

drone_app/
├── app.py                    # Streamlit dashboard
├── install.sh                # Bazzite setup script
├── run.sh                    # Daily startup script  
├── README.md                 # Setup instructions
├── CONVERSATION_SUMMARY.md   # This file
└── data/
    ├── oblast_features.csv         # 24 Ukrainian regions
    ├── observations.csv            # Recorded strike observations
    ├── weekly_forecast.csv         # Locked weekly forecast
    └── updated_forecast.csv        # Latest forecast computation

## Next steps to consider

1. Run the local Streamlit version on Bazzite (start here)
2. Record daily observations from Ukrainian Air Force Telegram
3. After a week, score model accuracy with real data
4. Return to Foundry attempt with conversation transcript as reference
5. Build out auto-scraper if you want fully automated updates
6. Add the Finland model as a second Streamlit page
7. Add the US-Iran attrition model as a third page

## Things to tell your Finnish tutor

The fishing analogy:
- Russia is a fisherman with one bucket of bait per week
- Different lakes (Ukrainian regions) are different distances from the dock
- Closer lakes get more bait (Kharkiv, Donetsk)
- Farther lakes barely see any (Lviv, Zakarpattia)
- Russia can't fish two oceans at once (Ukraine + Finland)
- Building a bigger bait bucket takes time (12-18 months)

The math is regression of past patterns plus respecting physical constraints. It's not magic — it's geography times strategic value times production capacity.
