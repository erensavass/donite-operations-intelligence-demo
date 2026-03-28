# Donite Operations Intelligence Demo

This project was built as a personal initiative to explore how data-driven decision support can be applied to industrial manufacturing environments.

I wanted to simulate a practical analytics workflow for a thermoforming / CNC trimming setting using synthetic data. Instead of focusing on a highly complex model, I aimed to build an explainable prototype that combines monitoring, predictive signals, root-cause analysis, and simple operational recommendations.

## Why I built this
After reviewing the public profile of a manufacturing company working in areas such as thermoforming and CNC trimming, I wanted to create a small prototype showing how I would think about:
- process monitoring
- scrap and rework analysis
- short-horizon failure risk
- basic maintenance signals
- actionable operational recommendations

## Important note
This project does **not** use real company data.  
It uses a fully synthetic dataset created only for demonstration purposes.

## Features
- KPI monitoring dashboard  
- Failure-risk estimation  
- Root-cause style analysis for scrap drivers  
- Recommendation engine  
- What-if simulation for process changes  

## Files
- `app.py` → Streamlit dashboard  
- `requirements.txt` → dependencies  
- `dashboard_screenshot.png` → app screenshot  

## Run locally
```bash
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py