@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ==============================================================
echo  Experimentos finais - idade, sexo e periodo pareados
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

for %%F in (experimentos_finais.py checagens_idade.py checagens_defesa.py svd_features_mfcc_wavelet_temporal.csv svd_features_healthy.csv) do (
    if not exist "%%F" (
        echo ERRO: nao achei %%F nesta pasta.
        pause
        exit /b 1
    )
)

echo Deve levar de 15 a 40 minutos. Se interromper, ao rodar de novo
echo ele continua de onde parou. Deixe o PC na tomada e sem suspender.
echo.
python experimentos_finais.py

echo.
echo ==============================================================
echo  Fim. Me envie o arquivo experimentos_finais_relatorio.txt
echo ==============================================================
pause
