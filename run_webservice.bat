@echo off
REM Activate your virtual environment
call D:\python_envs\yoloenv\Scripts\activate.bat

REM Change to your project directory
cd D:\pyton_projects\FlowCount2

REM Run your application
python app.py

REM Keep window open if there are errors (optional)
pause