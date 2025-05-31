@echo off
REM Create virtual environment
python -m venv venv

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Upgrade pip
python -m pip install --upgrade pip

REM Install PyTorch and torchvision with CUDA 12.1 support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

REM Optional: install extra useful packages
pip install matplotlib torchsummary

echo.
echo ✅ Virtual environment created and PyTorch with CUDA 12.1 installed!
pause
