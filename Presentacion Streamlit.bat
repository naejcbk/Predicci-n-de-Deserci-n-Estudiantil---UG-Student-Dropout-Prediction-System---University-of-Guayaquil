@echo off
title Lanzador del Dashboard de Desercion - UG
echo ====================================================
echo    INICIANDO DASHBOARD DE DESERCION ESTUDIANTIL
echo           Jean Carlos Burgos - UG 2026
echo ====================================================
echo.
echo Espera un momento mientras se carga el modelo de IA...
echo.

:: Ejecutar Streamlit usando el modulo de Python
python -m streamlit run Streamlit.py

:: Si ocurre un error, la ventana no se cerrara inmediatamente
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Hubo un problema al iniciar la aplicacion.
    echo Asegurate de estar en la carpeta correcta y tener instalado Streamlit.
    pause
)