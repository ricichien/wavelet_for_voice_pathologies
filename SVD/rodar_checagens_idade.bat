@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ==============================================================
echo  Checagem de idade e comparacao com grupos pareados
echo  Pasta: %cd%
echo ==============================================================
echo.

if not exist "venv\Scripts\activate.bat" (
    echo ERRO: nao achei a pasta venv aqui. Este .bat precisa ficar na pasta do
    echo projeto WavletDetectionPatologies, junto com o rodar_pipeline.bat.
    pause
    exit /b 1
)
call "venv\Scripts\activate.bat"

if not exist "checagens_idade.py" (
    echo ERRO: coloque o checagens_idade.py nesta pasta.
    pause
    exit /b 1
)
if not exist "checagens_defesa.py" (
    echo ERRO: coloque tambem o checagens_defesa.py nesta pasta. O de idade usa ele.
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

echo Pode levar de 1 a 2 horas. Se interromper, ao rodar de novo ele continua
echo de onde parou. Deixe o PC na tomada e sem suspender.
echo.
python checagens_idade.py

echo.
echo ==============================================================
echo  Fim. Me envie o arquivo checagens_idade_relatorio.txt
echo ==============================================================
pause
