@echo off
cd /d "%~dp0scripts"
"%~dp0..\.venv-gpu\Scripts\python.exe" train_resnet50.py --skip-download
