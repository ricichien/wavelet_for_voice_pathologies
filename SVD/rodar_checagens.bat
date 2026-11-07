@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ==============================================================
echo  Checagens para a defesa - validacao cruzada por falante
echo  Pasta: %cd%
echo ==============================================================
echo.

if not exist "venv\Scripts\activate.bat" (
    echo ERRO: nao achei a pasta venv aqui. Mova este .bat para a pasta do projeto
    echo WavletDetectionPatologies, a mesma onde esta o rodar_pipeline.bat.
    pause
    exit /b 1
)
call "venv\Scripts\activate.bat"

if not exist "checagens_defesa.py" (
    echo ERRO: coloque o checagens_defesa.py nesta pasta.
    pause
    exit /b 1
)
if not exist "svd_features_mfcc_wavelet_temporal.csv" (
    echo ERRO: nao achei svd_features_mfcc_wavelet_temporal.csv nesta pasta.
    pause
    exit /b 1
)
if not exist "svd_features_healthy.csv" (
    echo ERRO: nao achei svd_features_healthy.csv nesta pasta.
    pause
    exit /b 1
)

echo Pode demorar algumas horas. Se interromper, ao rodar de novo ele continua
echo de onde parou. Deixe o PC na tomada e sem suspender.
echo.
python checagens_defesa.py

echo.
echo ==============================================================
echo  Fim. O resumo esta em checagens_defesa_relatorio.txt
echo ==============================================================
pause
