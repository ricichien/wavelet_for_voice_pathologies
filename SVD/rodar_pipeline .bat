@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ==============================================================
echo  Pipeline SVD - WPT temporal
echo  Pasta: %cd%
echo ==============================================================
echo.

REM --- 1) venv ---
if not exist "venv\Scripts\activate.bat" (
    echo [1/3] Criando o venv...
    python -m venv venv
    if errorlevel 1 (
        echo ERRO ao criar o venv. O Python esta instalado e no PATH?
        pause
        exit /b 1
    )
) else (
    echo [1/3] venv ja existe.
)
call "venv\Scripts\activate.bat"

REM --- 2) bibliotecas: so instala se faltar alguma ---
python -c "import numpy, pandas, scipy, librosa, pywt, sklearn, xgboost, catboost, imblearn" >nul 2>&1
if errorlevel 1 (
    echo [2/3] Faltam bibliotecas. Instalando, pode demorar uns minutos...
    pip install numpy pandas scipy librosa PyWavelets scikit-learn xgboost catboost imbalanced-learn
    if errorlevel 1 (
        echo ERRO ao instalar. Veja a mensagem acima.
        pause
        exit /b 1
    )
) else (
    echo [2/3] Bibliotecas ok.
)

REM --- 3) acha o script ---
set SCRIPT=
if exist "src\svd_pipeline_wpt_temporal_HEALTHY.py" set SCRIPT=src\svd_pipeline_wpt_temporal_HEALTHY.py
if not defined SCRIPT if exist "src\svd_pipeline_wpt_temporal.py" set SCRIPT=src\svd_pipeline_wpt_temporal.py
if not defined SCRIPT (
    echo.
    echo ERRO: nao achei nenhum script na pasta src desta pasta.
    echo Este .bat precisa ficar na pasta do projeto WavletDetectionPatologies,
    echo a que tem a pasta src dentro. Conteudo desta pasta:
    echo.
    dir /b
    echo.
    pause
    exit /b 1
)

if not exist "svd_features_mfcc_wavelet_temporal.csv" (
    echo.
    echo AVISO: nao achei svd_features_mfcc_wavelet_temporal.csv nesta pasta.
    echo Sem ele, o script vai tentar reprocessar as 9 patologias, o que demora.
    echo Se nao era isso, feche esta janela e mova o .bat para a pasta certa.
    echo Para continuar mesmo assim:
    pause
)

echo [3/3] Rodando %SCRIPT%
echo.
python %SCRIPT%

echo.
echo ==============================================================
echo  Fim da execucao. Role para cima para ver os resultados.
echo ==============================================================
pause
