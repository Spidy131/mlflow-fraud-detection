@echo off

cd /d C:\Users\user\python_codes\mlflow-fraud-detection

call mlflow-env\Scripts\activate.bat

python run_monitoring.py >> monitoring_scheduler.log 2>&1