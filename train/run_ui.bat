@echo off
setlocal

cd /d "%~dp0"

echo Starting Streamlit UI...
echo If the first port is busy, it will try the next one.

python -m streamlit run ui.py --server.address 127.0.0.1 --server.port 8501

endlocal
