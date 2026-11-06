@echo off
cd /d "%~dp0"
echo Installing Python requirements...
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo Could not install requirements. Ensure Python and internet access are available.
  pause
  exit /b 1
)
echo Launching IMDb Movie Analytics Dashboard...
python -m streamlit run app.py
pause
