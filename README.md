# 📈 Graph Digitizer (Pix2Plot)

Upload a graph image → set 2-point calibration on each axis → extract (click or auto by color) → **download CSV**.  
Beginner-friendly, laptop-only workflow—no mobile needed.

![app-screenshot](docs/screenshot.png) <!-- optional: replace with your own -->

---

## ✨ Features
- **Upload PNG/JPG** of a plot.
- **2-point axis calibration** (supports linear, semilogx, semilogy, loglog).
- **Digitize modes**:
  - Manual: click along the curve.
  - Auto-by-color (HSV): detect points with OpenCV.
- **Export**: CSV of (x, y) + optional `recipe.json` (reproducible settings).
- Works fully **offline** on Windows.

---

## 🧱 Tech Stack
Streamlit, NumPy, pandas, OpenCV, Pillow, `streamlit-drawable-canvas`.

---

## 🚀 Quickstart (Windows, PowerShell)

```powershell
# 1) clone your repo (or cd into your project folder)
cd C:\Users\YOURNAME\Documents\prototype

# 2) create & activate venv
python -m venv .venv
.venv\Scripts\Activate.ps1

# 3) install deps (pinned to avoid canvas/streamlit breakage)
pip install -r requirements.txt

# 4) run the app
streamlit run app.py
