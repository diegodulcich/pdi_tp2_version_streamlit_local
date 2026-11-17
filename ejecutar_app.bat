@echo off
setlocal

rem Cambia a la carpeta del proyecto para que las rutas relativas funcionen.
cd /d "%~dp0"

rem uv crea o reutiliza el entorno virtual y ejecuta Streamlit con Python 3.12.
where uv >nul 2>nul
if errorlevel 1 (
    echo No se encontro uv. Instala uv y vuelve a ejecutar este archivo.
    pause
    exit /b 1
)

uv run --python 3.12 streamlit run app.py --server.address 127.0.0.1
if errorlevel 1 (
    echo La aplicacion se cerro con un error. Revisa el mensaje anterior.
    pause
    exit /b 1
)

endlocal
