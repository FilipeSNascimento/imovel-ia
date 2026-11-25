@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Instalando/verificando bibliotecas do Imovel IA...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo Erro ao instalar dependencias. Verifique se o Python esta instalado e no PATH.
    pause
    exit /b 1
)
echo Iniciando aplicacao Imovel IA...
python -m streamlit run app.py
pause
