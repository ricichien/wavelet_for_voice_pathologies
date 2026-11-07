# -*- coding: utf-8 -*-
"""
Pipeline SVD - Saarbruecken Voice Database
Extracao de atributos MFCC + Wavelet + WPT temporal a partir de audio
bruto (.nsp), classificacao supervisionada e comparacao entre
representacoes com foco na WPT e na dinamica temporal.

Segue a metodologia descrita nas Subsecoes 3.3, 3.4 e 3.5 do TCC:
  - familias de wavelet: db4, sym8, coif5
  - atributos por sub-banda: energia, entropia, desvio padrao
  - split treino/teste por FALANTE (SprecherID), nao por gravacao,
    para evitar vazamento de dados (data leakage) entre treino e teste
  - padronizacao (StandardScaler), SMOTE no treino
  - classificadores: LR, RF, SVM, MLP, XGBoost, CatBoost, Voting, Stacking

REQUISITOS (instalar antes de rodar):
    pip install -r requirements.txt     # na raiz do projeto; requer Python >= 3.10

COMO RODAR:
    python src/svd_pipeline_wpt_temporal.py
"""

# IMPORTANTE: limita o numero de threads internas usadas por numpy/scipy/
# librosa (BLAS/MKL/OpenBLAS) ANTES de importar essas bibliotecas. Sem isso,
# cada processo paralelo tenta usar TODOS os nucleos da maquina para suas
# proprias operacoes internas -- com N_JOBS processos rodando ao mesmo
# tempo, isso multiplica o numero de threads ativas (N_JOBS x nucleos) e
# pode saturar RAM/CPU a ponto de travar o sistema operacional.
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import io
import csv
import time
import random
import struct
import zipfile
import warnings
import multiprocessing as mp

import numpy as np
import pandas as pd
import librosa
import pywt
from scipy.stats import skew, kurtosis

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False


# ---------------------------------------------------------------------------
# 1) CONFIGURACAO - caminhos dos .zip e classes selecionadas
# ---------------------------------------------------------------------------

# Os caminhos sao resolvidos a partir da raiz do projeto para que o pipeline
# rode de qualquer diretorio de trabalho (o fonte vive em src/, os dados em
# dataset/).
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

# NAO descompacte os .zip. collect_zip_items() le os .nsp direto de dentro do
# arquivo compactado (modulo zipfile), junto com o overview.csv que mapeia
# AufnahmeID -> SprecherID -- e esse mapeamento que sustenta o split por
# falante. Uma pasta descompactada nao seria encontrada por este codigo.
PATHOLOGY_ZIPS = {
    # neurologico/motor
    os.path.join(DATASET_DIR, "Dysarthrophonie.zip"): "dysarthrophonie",
    os.path.join(DATASET_DIR, "Rekurrensparese.zip"): "rekurrensparese",
    os.path.join(DATASET_DIR, "Spasmodische_Dysphonie.zip"): "spasmodische_dysphonie",
    os.path.join(DATASET_DIR, "Vox_senilis.zip"): "vox_senilis",
    os.path.join(DATASET_DIR, "Zentral-laryngale_Bewegungsstörung.zip"): "zentral_laryngale",
    # nao-neurologico (contraste)
    os.path.join(DATASET_DIR, "Psychogene_Dysphonie.zip"): "psychogene_dysphonie",
    os.path.join(DATASET_DIR, "Laryngitis.zip"): "laryngitis",
    os.path.join(DATASET_DIR, "Funktionelle_Dysphonie.zip"): "funktionelle_dysphonie",
    os.path.join(DATASET_DIR, "Balbuties.zip"): "balbuties",
}

# Qual tarefa binaria rodar:
#   "saudavel_vs_patologico" -> healthy (controle) contra as 9 patologias juntas
#   "neuro_vs_nao_neuro"     -> so patologias: neurologico contra nao-neurologico
TAREFA = "saudavel_vs_patologico"

LABEL_TO_GROUP = {
    "dysarthrophonie": "neurologico",
    "rekurrensparese": "neurologico",
    "spasmodische_dysphonie": "neurologico",
    "vox_senilis": "neurologico",
    "zentral_laryngale": "neurologico",
    "psychogene_dysphonie": "nao_neurologico",
    "laryngitis": "nao_neurologico",
    "funktionelle_dysphonie": "nao_neurologico",
    "balbuties": "nao_neurologico",
}

# Configuracao usada na maquina Windows do TCC (9 classes). Mantida aqui para
# referencia: basta repor os .zip em dataset/ e acrescentar as entradas acima.
#   nucleo neurologico motor: Vox_senilis, Zentral-laryngale_Bewegungsstoerung,
#       Rekurrensparese, Dysarthrophonie, Spasmodische_Dysphonie
#   grupo de contraste nao neurologico: Funktionelle_Dysphonie,
#       Psychogene_Dysphonie, Laryngitis, Balbuties

# Se quiser tambem baixar e incluir a pasta "healthy" como controle,
# adicione o caminho do zip aqui com o rotulo "healthy".
# ATENCAO: o zip "healthy.zip" da SVD tem ~6 GB e milhares de falantes;
# recomenda-se usar apenas uma amostra dele (ver SAMPLE_HEALTHY_N abaixo).
HEALTHY_ZIP = os.environ.get("AVFAD_SVD_HEALTHY_ZIP", os.path.join(BASE_DIR, "data", "SVD", "healthy.zip"))
# Numero de FALANTES saudaveis sorteados. As 9 patologias somam ~550 falantes,
# entao ~550 deixa os dois lados com um numero parecido de falantes (e de
# gravacoes). Menos que isso funciona, mas desbalanceia a tarefa.
SAMPLE_HEALTHY_N = 550

# O healthy.zip e processado em lotes de falantes (em vez de ler tudo na RAM
# de uma vez, o que ja travou sua maquina antes). 100 falantes ~ 1 GB de audio.
HEALTHY_BATCH_SPEAKERS = 100
HEALTHY_FEATURES_CSV = "svd_features_healthy.csv"

# Wavelets utilizadas (Subsecao 3.5 do TCC)
WAVELET_FAMILIES = ["db4", "sym8", "coif5"]
WAVELET_LEVEL = 4  # nivel seguro para os 3 filtros na janela de analise definida abaixo

# Parametros de analise por janela curta -- usados tanto para MFCC quanto
# para a Wavelet, garantindo comparacao justa entre as duas representacoes
# (ver discussao: a versao anterior decompunha a wavelet no sinal inteiro
# de uma vez, perdendo a resolucao temporal que e a propria justificativa
# teorica de usa-la; agora ambas operam sobre os mesmos frames curtos)
TARGET_SR = 16000        # taxa de reamostragem comum (fala: 16 kHz e padrao)
FRAME_MS = 50             # duracao do frame em milissegundos
HOP_MS = 20               # deslocamento entre frames (overlap de 60%)

# Parametros do ENVELOPE de amplitude (hipotese de modulacao lenta / tremor)
# ver extract_envelope_features() para justificativa teorica
ENVELOPE_FRAME_MS = 25
ENVELOPE_HOP_MS = 10
TREMOR_BAND_HZ = (3.0, 12.0)  # faixa classica de tremor vocal na literatura
ENVELOPE_WAVELET_LEVEL = 2    # nivel alvo; reduzido automaticamente se o
                              # audio for curto demais (ver funcao abaixo)

# Wavelet Packet Transform (WPT) aplicada ao sinal inteiro -- ver
# extract_wpt_features() para a justificativa metodologica (baseada na
# literatura de classificacao de patologia vocal, que tipicamente usa
# niveis 5-8; nivel 6 e seguro para toda a faixa de duracoes do dataset)
WPT_LEVEL = 6

# WPT temporal: reaplica a WPT em janelas curtas, permitindo medir
# como a energia de cada sub-banda varia ao longo da gravacao.
# Mantemos nivel 6 para comparacao direta com o WPT global ja testado.
WPT_TEMPORAL_FRAME_MS = 50
WPT_TEMPORAL_HOP_MS = 20

N_MFCC = 13
# numero de filtros mel. O padrao do librosa (128) e excessivo para a janela
# usada aqui: com n_fft = 800 (50 ms a 16 kHz) ha 401 bins de FFT, e 32 dos
# 128 filtros acabam cobrindo apenas 2 bins cada -- bandas vizinhas quase
# identicas, que inflam a redundancia do banco sem acrescentar informacao.
# Com 40 filtros o suporte mediano sobe para ~12 bins. 40 tambem e a faixa
# convencional para fala a 16 kHz (HTK/Kaldi usam 23-40).
# NOTA: nenhuma das duas configuracoes gera filtros vazios nesta janela --
# isso foi verificado empiricamente, nao ha log(0) envolvido.
N_MELS = 40
OUTPUT_FEATURES_CSV = "svd_features_mfcc_wavelet_temporal.csv"
OUTPUT_RESULTS_CSV = "svd_results_wpt_temporal_summary.csv"
OUTPUT_CONFUSION_TXT = "svd_confusion_matrices.txt"
RANDOM_STATE = 42

# numero de processos paralelos para extracao de atributos.
#
# ATENCAO: os.cpu_count() no Windows retorna o numero de THREADS LOGICAS,
# nao de nucleos fisicos -- em CPUs com hyperthreading/E-cores (ex: Intel
# i5-14600K: 14 nucleos fisicos, 20 threads logicas), isso pode levar a um
# numero de processos maior do que a CPU aguenta confortavelmente,
# especialmente combinado com bibliotecas que tambem usam multithreading
# internamente (ver limitacao de threads BLAS acima).
#
# Por seguranca, o padrao usa uma fracao conservadora dos nucleos logicos.
# Ajuste manualmente conforme a estabilidade da sua maquina:
#   - maquinas mais simples / notebooks: N_JOBS = 4
#   - i5-14600K ou similar (14 nucleos fisicos): N_JOBS = 8 costuma ser seguro
#   - se quiser arriscar mais performance: N_JOBS = 10-12, mas monitore o
#     Gerenciador de Tarefas (uso de RAM e CPU) na primeira execucao
N_JOBS = 8


# ---------------------------------------------------------------------------
# 2) PARSER NATIVO DE ARQUIVOS .nsp (formato Kay CSL / DS16)
# ---------------------------------------------------------------------------
# A SVD distribui os audios em formato .nsp, que nao e lido nativamente por
# librosa/soundfile. O formato e um container tipo IFF, com chunks HEDR
# (metadados: taxa de amostragem) e SDA_ (dados de audio PCM 16-bit).
# Este parser le o .nsp diretamente da memoria (bytes), sem precisar de
# ferramenta externa de conversao.

def parse_nsp_bytes(data: bytes):
    """Decodifica bytes de um arquivo .nsp e retorna (samples, sample_rate).

    samples: np.ndarray float32 normalizado em [-1, 1]
    sample_rate: int
    """
    if data[0:4] != b"FORM" or data[4:8] != b"DS16":
        raise ValueError("Arquivo nao esta no formato NSP/DS16 esperado")

    offset = 12
    sample_rate = None
    audio_bytes = None

    while offset < len(data) - 8:
        tag = data[offset:offset + 4]
        size = struct.unpack("<I", data[offset + 4:offset + 8])[0]
        chunk_data = data[offset + 8:offset + 8 + size]

        if tag == b"HEDR":
            sample_rate = struct.unpack("<i", chunk_data[20:24])[0]
        elif tag == b"SDA_":
            audio_bytes = chunk_data

        offset += 8 + size
        if size % 2 == 1:  # padding para alinhamento par
            offset += 1

    if sample_rate is None or audio_bytes is None:
        raise ValueError("Nao foi possivel localizar chunks HEDR/SDA_ no arquivo")

    samples = np.frombuffer(audio_bytes, dtype="<i2").astype(np.float32) / 32768.0
    return samples, sample_rate


# ---------------------------------------------------------------------------
# 3) EXTRACAO DE ATRIBUTOS - MFCC e WAVELET
# ---------------------------------------------------------------------------

def extract_mfcc_features(y: np.ndarray, sr: int) -> dict:
    """Atributos MFCC (media e desvio padrao por coeficiente) + duracao.

    Usa janela/hop explicitos (FRAME_MS/HOP_MS) para que a analise fique
    diretamente comparavel a extracao wavelet por janela curta.
    """
    duration_sec = len(y) / sr
    n_fft = int(FRAME_MS / 1000 * sr)
    hop_length = int(HOP_MS / 1000 * sr)

    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=N_MFCC, n_mels=N_MELS, n_fft=n_fft, hop_length=hop_length
    )
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    feats = {"duration_sec": duration_sec}
    for i in range(N_MFCC):
        feats[f"mfcc_{i}_mean"] = mfcc_mean[i]
        feats[f"mfcc_{i}_std"] = mfcc_std[i]
    return feats


def _shannon_entropy(coeffs: np.ndarray, eps: float = 1e-12) -> float:
    """Entropia de Shannon normalizada da energia dos coeficientes."""
    energy = coeffs ** 2
    total = np.sum(energy) + eps
    p = energy / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p + eps)))


def extract_wavelet_features(y: np.ndarray, sr: int, wavelet: str,
                              level: int = WAVELET_LEVEL) -> dict:
    """Atributos estatisticos por sub-banda da DWT, calculados por JANELA
    CURTA e depois agregados entre janelas -- mesma logica usada no MFCC.

    Para cada sub-banda, calcula ao longo das janelas (frames):
      - log-energia:  media, desvio, SKEWNESS, KURTOSE
      - entropia:     media, desvio, SKEWNESS, KURTOSE
      - DELTA (variacao entre frames consecutivos) da log-energia:
        media e desvio do delta, e da delta-delta (segunda derivada)

    Justificativa (discussao Secao 4 do TCC):
      - skewness/kurtose capturam assimetria e "picos" na distribuicao de
        energia ao longo do tempo -- a media/desvio isoladas tendem a
        diluir eventos transientes isolados (ex.: micro-tremores),
        exatamente o tipo de padrao que a media aritmetica "acha" e
        homogeneiza.
      - delta/delta-delta capturam a TAXA DE VARIACAO da energia entre
        frames sucessivos, replicando a pratica classica de MFCC+Delta+
        Delta-Delta em reconhecimento de fala/locutor, e dando ao
        classificador acesso direto a dinamica temporal (nao so o nivel).
    """
    frame_length = int(FRAME_MS / 1000 * sr)
    hop_length = int(HOP_MS / 1000 * sr)

    if len(y) < frame_length:
        frames = y.reshape(1, -1)
    else:
        # ATENCAO: com axis=0 o librosa retorna (n_frames, frame_length) --
        # cada LINHA e um frame completo. Indexar por coluna (frames[:, i])
        # pegaria a i-esima AMOSTRA de todos os frames, ou seja, o sinal
        # subamostrado no passo do hop, e nao o i-esimo trecho de audio.
        frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length, axis=0)

    n_frames = frames.shape[0]
    n_subbands = level + 1  # cA_level + cD_level ... cD_1

    log_energies = np.zeros((n_frames, n_subbands))
    entropies = np.zeros((n_frames, n_subbands))

    for i in range(n_frames):
        # .copy() e obrigatorio aqui: librosa.util.frame() retorna uma VIEW
        # somente-leitura (stride tricks) sobre o array original. Uma fatia
        # de uma linha dessa view ja vem contigua, entao np.ascontiguousarray
        # NAO faz copia nesse caso (so copia quando precisa reordenar bytes)
        # e o array continua read-only -- pywt/scipy entao falham ao tentar
        # escrever nesse buffer ("buffer source array is read-only").
        # .copy() forca alocacao de memoria nova e gravavel, resolvendo isso
        # de forma garantida, independente do layout de memoria.
        frame = frames[i, :].copy()
        coeffs = pywt.wavedec(frame, wavelet, level=level)
        for j, c in enumerate(coeffs):
            energy = np.sum(c ** 2)
            log_energies[i, j] = np.log(energy + 1e-10)
            entropies[i, j] = _shannon_entropy(c)

    subband_labels = [f"{wavelet}_cA{level}"] + [f"{wavelet}_cD{level - k}" for k in range(level)]

    feats = {}
    for j, label in enumerate(subband_labels):
        le_col = log_energies[:, j]
        en_col = entropies[:, j]

        # estatisticas de nivel (mean/std/skew/kurtose)
        feats[f"{label}_logenergy_mean"] = float(np.mean(le_col))
        feats[f"{label}_logenergy_std"] = float(np.std(le_col))
        feats[f"{label}_entropy_mean"] = float(np.mean(en_col))
        feats[f"{label}_entropy_std"] = float(np.std(en_col))

        if n_frames >= 3:
            feats[f"{label}_logenergy_skew"] = float(skew(le_col))
            feats[f"{label}_logenergy_kurtosis"] = float(kurtosis(le_col))
            feats[f"{label}_entropy_skew"] = float(skew(en_col))
            feats[f"{label}_entropy_kurtosis"] = float(kurtosis(en_col))
        else:
            feats[f"{label}_logenergy_skew"] = 0.0
            feats[f"{label}_logenergy_kurtosis"] = 0.0
            feats[f"{label}_entropy_skew"] = 0.0
            feats[f"{label}_entropy_kurtosis"] = 0.0

        # delta e delta-delta da log-energia (dinamica temporal entre frames)
        if n_frames >= 2:
            delta = np.diff(le_col)
            feats[f"{label}_logenergy_delta_mean"] = float(np.mean(delta))
            feats[f"{label}_logenergy_delta_std"] = float(np.std(delta))
        else:
            delta = np.array([0.0])
            feats[f"{label}_logenergy_delta_mean"] = 0.0
            feats[f"{label}_logenergy_delta_std"] = 0.0

        if len(delta) >= 2:
            delta2 = np.diff(delta)
            feats[f"{label}_logenergy_delta2_mean"] = float(np.mean(delta2))
            feats[f"{label}_logenergy_delta2_std"] = float(np.std(delta2))
        else:
            feats[f"{label}_logenergy_delta2_mean"] = 0.0
            feats[f"{label}_logenergy_delta2_std"] = 0.0

    return feats


def extract_amplitude_envelope(y: np.ndarray, sr: int):
    """Extrai o envelope de amplitude (energia RMS por frame curto) do
    sinal inteiro, retornando (envelope, envelope_sr).

    O envelope e uma serie temporal de "baixa taxa de amostragem" que
    descreve como a intensidade da voz varia ao longo de toda a gravacao
    -- diferente do MFCC/Wavelet por frame (que descrevem a estrutura
    espectral fina dentro de cada janela de ~50ms), o envelope captura
    flutuacoes MAIS LENTAS de amplitude, na escala de dezenas de Hz.

    Isso e relevante porque o tremor vocal associado a Parkinson tende a
    se manifestar como modulacao de amplitude/frequencia na faixa de
    aproximadamente 3-12 Hz (Little et al., 2007) -- uma escala temporal
    muito mais lenta do que qualquer coisa capturavel dentro de uma unica
    janela de analise espectral de 25-50ms. Analisar o envelope
    separadamente permite investigar essa modulacao diretamente.
    """
    frame_length = int(ENVELOPE_FRAME_MS / 1000 * sr)
    hop_length = int(ENVELOPE_HOP_MS / 1000 * sr)
    frame_length = max(frame_length, 2)
    hop_length = max(hop_length, 1)

    envelope = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    envelope_sr = sr / hop_length
    return envelope, envelope_sr


def extract_tremor_fft_features(envelope: np.ndarray, envelope_sr: float) -> dict:
    """Atributos diretos de tremor via espectro de Fourier do envelope.

    Calcula, dentro da faixa classica de tremor vocal (3-12 Hz):
      - energia relativa na faixa de tremor (proporcao da energia total
        do envelope que se concentra nessa faixa)
      - frequencia de pico dentro da faixa
      - magnitude do pico
    Alem de uma medida geral de "profundidade de modulacao" (coeficiente
    de variacao do envelope: desvio padrao / media).
    """
    feats = {}

    mean_env = np.mean(envelope)
    std_env = np.std(envelope)
    feats["envelope_modulation_depth"] = float(std_env / (mean_env + 1e-10))

    n = len(envelope)
    if n < 4:
        # sinal curto demais para uma FFT com resolucao util
        feats["tremor_band_energy_ratio"] = 0.0
        feats["tremor_peak_freq_hz"] = 0.0
        feats["tremor_peak_magnitude"] = 0.0
        return feats

    env_centered = envelope - mean_env
    fft_vals = np.abs(np.fft.rfft(env_centered))
    freqs = np.fft.rfftfreq(n, d=1 / envelope_sr)

    total_energy = float(np.sum(fft_vals ** 2)) + 1e-12
    tremor_mask = (freqs >= TREMOR_BAND_HZ[0]) & (freqs <= TREMOR_BAND_HZ[1])

    if tremor_mask.sum() == 0:
        # gravacao curta demais para resolver a faixa de tremor na FFT
        feats["tremor_band_energy_ratio"] = 0.0
        feats["tremor_peak_freq_hz"] = 0.0
        feats["tremor_peak_magnitude"] = 0.0
        return feats

    tremor_energy = float(np.sum(fft_vals[tremor_mask] ** 2))
    feats["tremor_band_energy_ratio"] = tremor_energy / total_energy

    peak_idx = int(np.argmax(fft_vals[tremor_mask]))
    feats["tremor_peak_freq_hz"] = float(freqs[tremor_mask][peak_idx])
    feats["tremor_peak_magnitude"] = float(fft_vals[tremor_mask][peak_idx])

    return feats


def extract_envelope_wavelet_features(envelope: np.ndarray, wavelet: str,
                                       target_level: int = ENVELOPE_WAVELET_LEVEL) -> dict:
    """Decompoe o ENVELOPE (nao o audio bruto) via DWT, extraindo
    energia/entropia por sub-banda de modulacao.

    Diferente da extract_wavelet_features() (que decompoe o audio bruto
    em janelas curtas, capturando estrutura espectral fina), esta funcao
    decompoe o envelope de amplitude inteiro de uma vez -- apropriado
    aqui porque o envelope ja e, por natureza, uma serie de baixa taxa de
    amostragem e o fenomeno de interesse (modulacao/tremor) se estende ao
    longo de toda a gravacao, nao dentro de uma janela curta.

    Gravacoes muito curtas podem nao comportar o nivel de decomposicao
    alvo -- nesse caso, o nivel e reduzido automaticamente (e as colunas
    de sub-bandas mais profundas, que nao puderam ser calculadas, sao
    preenchidas com zero) para manter o vetor de atributos com tamanho
    constante entre todas as gravacoes.
    """
    subband_labels = ([f"envwav_{wavelet}_cA{target_level}"] +
                       [f"envwav_{wavelet}_cD{target_level - k}" for k in range(target_level)])

    feats = {f"{label}_logenergy": 0.0 for label in subband_labels}
    feats.update({f"{label}_entropy": 0.0 for label in subband_labels})

    n = len(envelope)
    if n < 4:
        return feats

    dec_len = pywt.Wavelet(wavelet).dec_len
    max_level = pywt.dwt_max_level(n, dec_len)
    safe_level = max(0, min(target_level, max_level))

    if safe_level == 0:
        # decomposicao nao e viavel; usa o envelope inteiro como "banda unica"
        energy = np.sum(envelope ** 2)
        feats[f"{subband_labels[0]}_logenergy"] = float(np.log(energy + 1e-10))
        feats[f"{subband_labels[0]}_entropy"] = _shannon_entropy(envelope)
        return feats

    coeffs = pywt.wavedec(envelope, wavelet, level=safe_level)
    # coeffs tem safe_level+1 elementos; se safe_level < target_level,
    # preenche apenas as primeiras sub-bandas (as demais ficam em 0.0)
    computed_labels = ([f"envwav_{wavelet}_cA{safe_level}"] +
                        [f"envwav_{wavelet}_cD{safe_level - k}" for k in range(safe_level)])

    for label, c in zip(computed_labels, coeffs):
        # usa o rotulo alvo correspondente por posicao (aproximacao e o
        # primeiro detalhe mais grosseiro tendem a ser comparaveis mesmo
        # com niveis diferentes)
        target_label = subband_labels[computed_labels.index(label)]
        energy = np.sum(c ** 2)
        feats[f"{target_label}_logenergy"] = float(np.log(energy + 1e-10))
        feats[f"{target_label}_entropy"] = _shannon_entropy(c)

    return feats


def _zero_crossing_rate(c: np.ndarray) -> float:
    """Proporcao de mudancas de sinal no vetor de coeficientes -- medida
    complementar a energia/entropia, que a literatura de classificacao de
    patologias vocais aponta como particularmente discriminante em
    alguns tipos de disfonia (ver discussao da Secao 4 do TCC).
    """
    if len(c) < 2:
        return 0.0
    signs = np.sign(c)
    signs[signs == 0] = 1
    return float(np.mean(np.abs(np.diff(signs)) > 0))


def extract_wpt_features(y: np.ndarray, wavelet: str, level: int = WPT_LEVEL) -> dict:
    """Wavelet Packet Transform (WPT) aplicada ao SINAL INTEIRO (nao em
    janelas curtas), replicando a metodologia predominante na literatura
    de classificacao de patologias vocais -- inclusive estudos que usam a
    propria Saarbruecken Voice Database.

    Diferente da DWT (extract_wavelet_features), que decompoe
    recursivamente apenas o canal de aproximacao, a WPT decompoe TAMBEM
    os canais de detalhe em cada nivel, produzindo uma arvore completa e
    balanceada de sub-bandas (2^level nos no nivel escolhido). Isso da
    controle muito mais fino sobre quais faixas de frequencia estao
    sendo analisadas, e e a abordagem usada nos trabalhos que reportam os
    melhores resultados nessa tarefa (energia/entropia extraidas dos nos
    de arvores WPT de nivel 5 a 8).

    O nivel 6 foi escolhido por ser viavel mesmo para as gravacoes mais
    curtas do dataset (min. ~0.15s), nas tres familias de wavelet usadas.
    """
    wp = pywt.WaveletPacket(data=y, wavelet=wavelet, mode="periodization", maxlevel=level)
    nodes = wp.get_level(level, order="freq")  # ordenados por frequencia crescente

    feats = {}
    for i, node in enumerate(nodes):
        c = node.data
        energy = np.sum(c ** 2)
        label = f"wpt_{wavelet}_n{i:02d}"
        feats[f"{label}_logenergy"] = float(np.log(energy + 1e-10))
        feats[f"{label}_entropy"] = _shannon_entropy(c)
        feats[f"{label}_zcr"] = _zero_crossing_rate(c)

    return feats


def extract_all_wpt_features(y: np.ndarray) -> dict:
    """Combina a WPT das 3 familias de wavelet em um unico vetor."""
    feats = {}
    for wavelet in WAVELET_FAMILIES:
        feats.update(extract_wpt_features(y, wavelet))
    return feats


def extract_wpt_temporal_features(
    y: np.ndarray,
    wavelet: str,
    level: int = WPT_LEVEL,
) -> dict:
    """WPT em janelas curtas + caracteristicas de dinamica temporal.

    Para cada sub-banda da WPT:
      - log-energia media e desvio;
      - energia relativa media e desvio;
      - ZCR medio e desvio;
      - delta da log-energia: media e desvio.

    A energia relativa e calculada dentro de cada frame, reduzindo a
    dependencia do volume absoluto da gravacao. O delta fornece ao modelo
    acesso explicito a variacao da energia entre frames consecutivos.
    """
    frame_length = max(int(WPT_TEMPORAL_FRAME_MS / 1000 * TARGET_SR), 2)
    hop_length = max(int(WPT_TEMPORAL_HOP_MS / 1000 * TARGET_SR), 1)

    if len(y) < frame_length:
        frames = y.reshape(1, -1)
    else:
        # ver nota em extract_wavelet_features: axis=0 produz
        # (n_frames, frame_length), com um frame por LINHA
        frames = librosa.util.frame(
            y,
            frame_length=frame_length,
            hop_length=hop_length,
            axis=0,
        )

    n_frames = frames.shape[0]
    n_nodes = 2 ** level

    log_energies = np.zeros((n_frames, n_nodes), dtype=np.float32)
    relative_energies = np.zeros((n_frames, n_nodes), dtype=np.float32)
    zcrs = np.zeros((n_frames, n_nodes), dtype=np.float32)

    for i in range(n_frames):
        # ver comentario equivalente em extract_wavelet_features: .copy() e
        # necessario para garantir buffer gravavel (librosa.util.frame
        # retorna view read-only, e ascontiguousarray nao resolve isso)
        frame = frames[i, :].copy()

        wp = pywt.WaveletPacket(
            data=frame,
            wavelet=wavelet,
            mode="periodization",
            maxlevel=level,
        )
        nodes = wp.get_level(level, order="freq")

        node_energies = np.array(
            [np.sum(node.data ** 2) for node in nodes],
            dtype=np.float64,
        )
        total_energy = float(np.sum(node_energies)) + 1e-12

        # O nivel 6 e seguro para frames de ~800 amostras nas wavelets atuais.
        for j, node in enumerate(nodes):
            energy = node_energies[j]
            log_energies[i, j] = np.log(energy + 1e-10)
            relative_energies[i, j] = energy / total_energy
            zcrs[i, j] = _zero_crossing_rate(node.data)

    feats = {}

    for j in range(n_nodes):
        label = f"wptt_{wavelet}_n{j:02d}"

        le = log_energies[:, j]
        re = relative_energies[:, j]
        zcr = zcrs[:, j]

        # Nivel
        feats[f"{label}_logenergy_mean"] = float(np.mean(le))
        feats[f"{label}_logenergy_std"] = float(np.std(le))

        # Energia normalizada pela energia total de cada frame
        feats[f"{label}_relenergy_mean"] = float(np.mean(re))
        feats[f"{label}_relenergy_std"] = float(np.std(re))

        # Estrutura oscilatoria da sub-banda
        feats[f"{label}_zcr_mean"] = float(np.mean(zcr))
        feats[f"{label}_zcr_std"] = float(np.std(zcr))

        # Dinamica temporal
        if n_frames >= 2:
            delta = np.diff(le)
            feats[f"{label}_delta_mean"] = float(np.mean(delta))
            feats[f"{label}_delta_std"] = float(np.std(delta))
        else:
            feats[f"{label}_delta_mean"] = 0.0
            feats[f"{label}_delta_std"] = 0.0

    return feats


def extract_all_wpt_temporal_features(y: np.ndarray) -> dict:
    """Combina a WPT temporal das 3 familias de wavelet."""
    feats = {}
    for wavelet in WAVELET_FAMILIES:
        feats.update(extract_wpt_temporal_features(y, wavelet))
    return feats


def extract_envelope_features(y: np.ndarray, sr: int) -> dict:
    """Combina todos os atributos derivados do envelope de amplitude:
    tremor via FFT + decomposicao wavelet do envelope (nas 3 familias).
    """
    envelope, envelope_sr = extract_amplitude_envelope(y, sr)

    feats = extract_tremor_fft_features(envelope, envelope_sr)
    for wavelet in WAVELET_FAMILIES:
        feats.update(extract_envelope_wavelet_features(envelope, wavelet))

    return feats


def extract_all_features(y: np.ndarray, sr: int) -> dict:
    """Combina atributos MFCC e Wavelet (todas as familias) em um unico vetor.

    Reamostra o sinal para TARGET_SR antes de qualquer extracao, garantindo
    que MFCC e Wavelet operem sobre o mesmo sinal e com os mesmos parametros
    de janela/hop -- comparacao justa entre as duas representacoes.
    """
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR

    feats = extract_mfcc_features(y, sr)
    for wavelet in WAVELET_FAMILIES:
        feats.update(extract_wavelet_features(y, sr, wavelet))
    feats.update(extract_envelope_features(y, sr))
    feats.update(extract_all_wpt_features(y))
    feats.update(extract_all_wpt_temporal_features(y))
    return feats


# ---------------------------------------------------------------------------
# 4) LEITURA DOS .zip E MONTAGEM DO DATASET
# ---------------------------------------------------------------------------

def read_overview_csv(zf: zipfile.ZipFile) -> dict:
    """Le o overview.csv dentro do zip e retorna {AufnahmeID: SprecherID}."""
    candidates = [n for n in zf.namelist() if n.endswith("overview.csv")]
    if not candidates:
        return {}

    with zf.open(candidates[0]) as f:
        reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))
        mapping = {}
        for row in reader:
            aufnahme_id = row.get("AufnahmeID", "").strip()
            sprecher_id = row.get("SprecherID", "").strip()
            if aufnahme_id:
                mapping[aufnahme_id] = sprecher_id
        return mapping


def _extract_worker(item: tuple) -> dict | None:
    """Funcao executada em processo paralelo: decodifica o .nsp e extrai
    todos os atributos (MFCC + Wavelet) de uma unica gravacao.

    Recebe uma tupla (raw_bytes, label, speaker_id, filename) para poder
    ser distribuida entre processos via multiprocessing.Pool.
    """
    raw_bytes, label, speaker_id, filename = item
    try:
        y, sr = parse_nsp_bytes(raw_bytes)
        if len(y) < sr * 0.1:  # descarta audios menores que 100ms
            return None

        feats = extract_all_features(y, sr)
        feats["label"] = label
        feats["speaker_id"] = speaker_id
        feats["filename"] = filename
        return feats
    except Exception as e:
        print(f"    erro ao processar {filename}: {e}")
        return None


def _recording_id(name: str) -> str:
    """AufnahmeID = prefixo numerico do nome do arquivo (ex.: 1580-a_n.nsp)."""
    return os.path.basename(name).split("-")[0]


def list_zip_speakers(zip_path: str) -> list:
    """Lista (ordenada) dos falantes distintos presentes no zip, sem ler audio.
    Tambem avisa se alguma gravacao nao tem falante no overview.csv: nesse
    caso o codigo cairia no AufnahmeID como se fosse um falante, e varias
    gravacoes da MESMA pessoa poderiam ir uma para treino e outra para teste
    (vazamento por falante).
    """
    with zipfile.ZipFile(zip_path) as zf:
        speaker_map = read_overview_csv(zf)
        nsp_files = [n for n in zf.namelist() if n.lower().endswith(".nsp")]

    if not speaker_map:
        print("  [AVISO] overview.csv nao encontrado/vazio no zip: cada gravacao "
              "sera tratada como um falante diferente (risco de vazamento!).")
    else:
        sem_mapa = sum(1 for n in nsp_files if _recording_id(n) not in speaker_map)
        if sem_mapa:
            print(f"  [AVISO] {sem_mapa}/{len(nsp_files)} gravacoes sem falante no "
                  f"overview.csv (usando AufnahmeID no lugar).")

    return sorted({speaker_map.get(_recording_id(n), _recording_id(n)) for n in nsp_files})


def collect_zip_items(zip_path: str, label: str, max_speakers: int | None = None,
                      speaker_subset: set | None = None) -> list:
    """Le os bytes brutos de cada .nsp dentro do zip (rapido, I/O), sem
    ainda calcular nenhum atributo. Retorna uma lista de tuplas prontas
    para serem processadas em paralelo por _extract_worker.

    speaker_subset: se informado, le so as gravacoes desses falantes (usado
    para processar o healthy.zip em lotes).
    max_speakers: sorteia (semente fixa) esse numero de falantes. O sorteio
    evita pegar so os primeiros falantes do zip, que podem ser de uma mesma
    epoca/sessao de gravacao.
    """
    items = []

    if not os.path.exists(zip_path):
        print(f"  [AVISO] arquivo nao encontrado, pulando: {zip_path}")
        return items

    with zipfile.ZipFile(zip_path) as zf:
        speaker_map = read_overview_csv(zf)
        nsp_files = [n for n in zf.namelist() if n.lower().endswith(".nsp")]

        def spk_of(name: str) -> str:
            return speaker_map.get(_recording_id(name), _recording_id(name))

        if speaker_subset is not None:
            nsp_files = [n for n in nsp_files if spk_of(n) in speaker_subset]
        elif max_speakers is not None:
            todos = sorted({spk_of(n) for n in nsp_files})
            random.Random(RANDOM_STATE).shuffle(todos)
            escolhidos = set(todos[:max_speakers])
            nsp_files = [n for n in nsp_files if spk_of(n) in escolhidos]

        print(f"  [{label}] {len(nsp_files)} arquivos .nsp encontrados")

        for name in nsp_files:
            raw_bytes = zf.read(name)
            items.append((raw_bytes, label, spk_of(name), name))

    return items


def process_zip_parallel(zip_path: str, label: str, max_speakers: int | None = None,
                         speaker_subset: set | None = None) -> list:
    """Processa um unico .zip: le os bytes (rapido) e extrai atributos em
    paralelo (multiprocessing), retornando a lista de linhas prontas.

    Processar um zip de cada vez (em vez de carregar todos os .zip na
    memoria simultaneamente) mantem o consumo de RAM sob controle --
    importante em maquinas com muitos arquivos grandes (algumas pastas da
    SVD tem quase 2 GB de audio bruto).
    """
    items = collect_zip_items(zip_path, label, max_speakers=max_speakers,
                              speaker_subset=speaker_subset)
    if not items:
        return []

    total = len(items)
    rows = []
    t0 = time.time()

    with mp.Pool(processes=N_JOBS) as pool:
        for i, feats in enumerate(pool.imap_unordered(_extract_worker, items, chunksize=8), 1):
            if feats is not None:
                rows.append(feats)
            if i % 200 == 0 or i == total:
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0
                eta = (total - i) / rate if rate > 0 else 0
                print(f"    processados {i}/{total}  "
                      f"({rate:.1f} arquivos/s, ETA ~{eta/60:.1f} min)")

    return rows


def build_dataset() -> pd.DataFrame:
    """Percorre todos os .zip configurados, processando UM DE CADA VEZ
    (le bytes + extrai atributos em paralelo), e salva incrementalmente
    em CSV apos cada pasta -- assim, se o processo for interrompido, o
    progresso ja feito nao se perde.
    """
    all_rows = []
    first_write = True

    # remove CSV de execucao anterior, se existir -- como agora salvamos
    # incrementalmente (append), um arquivo antigo causaria mistura de
    # dados de execucoes diferentes ou colunas incompativeis
    if os.path.exists(OUTPUT_FEATURES_CSV):
        os.remove(OUTPUT_FEATURES_CSV)
        print(f"[INFO] CSV anterior removido: {OUTPUT_FEATURES_CSV}\n")

    print(f"Configuracao: N_JOBS={N_JOBS} processos paralelos\n")

    for zip_path, label in PATHOLOGY_ZIPS.items():
        print(f"Processando: {label}")
        rows = process_zip_parallel(zip_path, label)
        all_rows.extend(rows)

        # salva incrementalmente (append) para nao perder progresso em caso
        # de interrupcao/crash no meio do processamento
        if rows:
            df_partial = pd.DataFrame(rows)
            df_partial.to_csv(
                OUTPUT_FEATURES_CSV, mode="a", index=False, header=first_write
            )
            first_write = False
        print(f"  concluido: {len(rows)} gravacoes processadas\n")

    # o healthy NAO entra aqui: e processado a parte (load_or_build_healthy),
    # em lotes e com CSV proprio, para nao reprocessar as patologias.

    df = pd.DataFrame(all_rows)

    if df.empty or "label" not in df.columns:
        print("\n[ERRO] Nenhuma gravacao foi processada com sucesso.")
        print("Verifique se os caminhos em PATHOLOGY_ZIPS apontam para arquivos existentes.")
        return df

    print(f"\nDataset completo salvo em {OUTPUT_FEATURES_CSV} -- shape: {df.shape}")
    print(df["label"].value_counts())
    return df


def load_or_build_healthy() -> pd.DataFrame:
    """Carrega o CSV de atributos do grupo saudavel, ou o constroi.

    Sorteia SAMPLE_HEALTHY_N falantes (semente fixa) e processa em lotes de
    HEALTHY_BATCH_SPEAKERS falantes, anexando cada lote ao CSV. Assim o pico
    de RAM fica em ~1 GB, e se a execucao for interrompida o que ja foi feito
    fica salvo: ao rodar de novo, so os falantes que faltam sao processados.
    """
    done_speakers = set()
    if os.path.exists(HEALTHY_FEATURES_CSV):
        df_prev = pd.read_csv(HEALTHY_FEATURES_CSV, dtype={"speaker_id": str})
        done_speakers = set(df_prev["speaker_id"].unique())
        print(f"[healthy] CSV existente com {len(done_speakers)} falantes.")

    if not HEALTHY_ZIP or not os.path.exists(HEALTHY_ZIP):
        print(f"[ERRO] healthy.zip nao encontrado: {HEALTHY_ZIP}")
        return pd.DataFrame()

    todos = list_zip_speakers(HEALTHY_ZIP)
    print(f"[healthy] {len(todos)} falantes no zip; sorteando {min(SAMPLE_HEALTHY_N, len(todos))}.")
    random.Random(RANDOM_STATE).shuffle(todos)
    escolhidos = todos[:SAMPLE_HEALTHY_N]
    faltam = [spk for spk in escolhidos if spk not in done_speakers]

    for i in range(0, len(faltam), HEALTHY_BATCH_SPEAKERS):
        lote = set(faltam[i:i + HEALTHY_BATCH_SPEAKERS])
        print(f"\n[healthy] lote {i // HEALTHY_BATCH_SPEAKERS + 1}: {len(lote)} falantes")
        rows = process_zip_parallel(HEALTHY_ZIP, "healthy", speaker_subset=lote)
        if rows:
            pd.DataFrame(rows).to_csv(
                HEALTHY_FEATURES_CSV, mode="a", index=False,
                header=not os.path.exists(HEALTHY_FEATURES_CSV),
            )

    if not os.path.exists(HEALTHY_FEATURES_CSV):
        return pd.DataFrame()
    df_h = pd.read_csv(HEALTHY_FEATURES_CSV)  # releitura: tipos iguais aos do CSV das patologias
    print(f"[healthy] {len(df_h)} gravacoes, {df_h['speaker_id'].nunique()} falantes.")
    return df_h


# ---------------------------------------------------------------------------
# 5) PRE-PROCESSAMENTO (split por falante)
# ---------------------------------------------------------------------------

def split_by_speaker(df: pd.DataFrame, feature_cols: list, test_size: float = 0.2):
    """Divide treino/teste garantindo que o mesmo falante nao aparece nos dois.

    Usa GroupShuffleSplit com speaker_id como grupo, evitando vazamento de
    dados quando ha multiplas gravacoes por pessoa (ex: spasmodische_dysphonie).
    """
    X = df[feature_cols].values
    y = df["label"].values
    groups = df["speaker_id"].values

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y, groups))

    y_train, y_test = y[train_idx], y[test_idx]

    # aviso de seguranca: classes com poucos falantes podem cair inteiras
    # so no treino ou so no teste (split por falante nao consegue dividi-las)
    train_classes = set(y_train)
    test_classes = set(y_test)
    missing_in_train = test_classes - train_classes
    missing_in_test = train_classes - test_classes
    if missing_in_train:
        print(f"  [AVISO] classes ausentes no TREINO (poucos falantes): {missing_in_train}")
    if missing_in_test:
        print(f"  [AVISO] classes ausentes no TESTE (poucos falantes): {missing_in_test}")

    return X[train_idx], X[test_idx], y_train, y_test


def preprocess(X_train, X_test, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if HAS_SMOTE:
        try:
            smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=3)
            X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
        except ValueError as e:
            print(f"  [AVISO] SMOTE nao aplicado (provavelmente classe com poucas amostras): {e}")

    return X_train_scaled, X_test_scaled, y_train


# ---------------------------------------------------------------------------
# 6) MODELAGEM
# ---------------------------------------------------------------------------

def get_baseline_models() -> dict:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        ),
        "random_forest": RandomForestClassifier(
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
        "svm_rbf": SVC(
            kernel="rbf",
            probability=True,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
    }


def get_advanced_models() -> dict:
    models = {"mlp": MLPClassifier(max_iter=500, random_state=RANDOM_STATE)}
    if HAS_XGBOOST:
        models["xgboost"] = XGBClassifier(eval_metric="mlogloss", random_state=RANDOM_STATE)
    if HAS_CATBOOST:
        models["catboost"] = CatBoostClassifier(
        verbose=0,
        random_state=RANDOM_STATE,
        auto_class_weights="Balanced",
        thread_count=4,
        allow_writing_files=False,
    )
    return models


def build_ensemble(base_models: dict) -> VotingClassifier:
    return VotingClassifier(estimators=list(base_models.items()), voting="soft")


def build_stacking(base_models: dict) -> StackingClassifier:
    return StackingClassifier(
        estimators=list(base_models.items()),
        final_estimator=LogisticRegression(max_iter=1000),
    )


# ---------------------------------------------------------------------------
# 7) AVALIACAO
# ---------------------------------------------------------------------------

# acumula o texto de todas as matrizes de confusao geradas na execucao, para
# salvar em OUTPUT_CONFUSION_TXT ao final (o resumo em CSV guarda so as metricas
# escalares; a matriz precisa de um formato proprio)
CONFUSION_BLOCKS = []


def format_confusion_matrix(cm: np.ndarray, class_names: list) -> str:
    """Formata a matriz de confusao como texto alinhado.

    Linhas = classe REAL, colunas = classe PREVISTA. As colunas sao indicadas
    por indice ([0], [1], ...) porque os nomes das patologias sao longos demais
    para servirem de cabecalho; a legenda fica nas linhas.
    """
    idx = [f"[{i}]" for i in range(len(class_names))]
    cw = max(max(len(str(int(v))) for v in cm.flatten()), 3)
    rot = [f"[{i}] {n}" for i, n in enumerate(class_names)]
    w = max(len(r) for r in rot)

    # nota: o separador usa "/" em vez de "\" porque f-strings no Python
    # 3.10/3.11 nao permitem barra invertida dentro da parte de expressao
    # (essa restricao so foi removida no Python 3.12) -- "\" causaria
    # SyntaxError: f-string expression part cannot include a backslash
    rotulo_cabecalho = "real / previsto"
    linhas = [f"{rotulo_cabecalho:<{w}}  " + " ".join(f"{c:>{cw}}" for c in idx)]
    for i, r in enumerate(rot):
        marca = " " if cm[i].sum() else " <- classe ausente no teste"
        linhas.append(f"{r:<{w}}  " + " ".join(f"{int(v):>{cw}}" for v in cm[i]) + marca)
    return "\n".join(linhas)


def evaluate_model(model, X_test, y_test, name: str, representation: str,
                   class_names: list | None = None) -> dict:
    y_pred = model.predict(X_test)
    metrics = {
        "representation": representation,
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }
    print(f"  {name:<22} acc={metrics['accuracy']:.4f}  bal_acc={metrics['balanced_accuracy']:.4f}  f1={metrics['f1']:.4f}")

    if class_names is not None:
        # labels= fixa a matriz em todas as classes conhecidas, inclusive as que
        # nao aparecem no teste -- caso contrario a matriz encolhe silenciosamente
        # e deixa de mostrar que uma classe nao esta sendo avaliada
        cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(class_names)))
        bloco = (f"--- {representation} | {name} "
                 f"(acc={metrics['accuracy']:.4f}, f1={metrics['f1']:.4f}) ---\n"
                 + format_confusion_matrix(cm, class_names))
        print("\n".join("      " + l for l in bloco.splitlines()))
        CONFUSION_BLOCKS.append(bloco)

    return metrics


def run_pipeline_for_representation(df: pd.DataFrame, feature_cols: list, representation: str) -> list:
    """Roda o pipeline completo (split, preprocess, treino, avaliacao) para
    um conjunto especifico de colunas de atributos (MFCC, Wavelet ou hibrido).
    """
    print(f"\n=== Representacao: {representation} ({len(feature_cols)} atributos) ===")

    X_train, X_test, y_train, y_test = split_by_speaker(df, feature_cols)
    X_train, X_test, y_train = preprocess(X_train, X_test, y_train)

    # codifica rotulos (strings -> inteiros) uma unica vez; necessario para
    # XGBoost/CatBoost, e mantido consistente para todos os modelos e para
    # o ensemble/stacking, que combinam as saidas de varios classificadores
    label_encoder = LabelEncoder()
    label_encoder.fit(np.concatenate([y_train, y_test]))
    y_train = label_encoder.transform(y_train)
    y_test = label_encoder.transform(y_test)

    results = []
    baseline_models = get_baseline_models()
    advanced_models = get_advanced_models()
    all_models = {**baseline_models, **advanced_models}

    for name, model in all_models.items():
        try:
            model.fit(X_train, y_train)
            results.append(evaluate_model(model, X_test, y_test, name, representation,
                                          class_names=list(label_encoder.classes_)))
        except Exception as e:
            print(f"  [ERRO] {name} falhou: {type(e).__name__}: {e}")

    ensemble = build_ensemble(all_models)
    ensemble.fit(X_train, y_train)
    results.append(evaluate_model(ensemble, X_test, y_test, "ensemble_voting", representation,
                                  class_names=list(label_encoder.classes_)))

    stacking = build_stacking(all_models)
    stacking.fit(X_train, y_train)
    results.append(evaluate_model(stacking, X_test, y_test, "stacking", representation,
                                  class_names=list(label_encoder.classes_)))

    return results


# ---------------------------------------------------------------------------
# 8) EXECUCAO PRINCIPAL
# ---------------------------------------------------------------------------

def main():
    print("=== EXPERIMENTO WPT TEMPORAL ===")
    print("O arquivo de features usado aqui e separado do pipeline anterior.")
    print("Na primeira execucao, as features serao reconstruidas para incluir")
    print("WPT temporal, energia relativa e delta de energia.\\n")

    if os.path.exists(OUTPUT_FEATURES_CSV):
        print(f"Carregando dataset ja existente de {OUTPUT_FEATURES_CSV}...")
        df = pd.read_csv(OUTPUT_FEATURES_CSV)
    else:
        df = build_dataset()

    if df.empty:
        print("ERRO: dataset vazio. Verifique os caminhos dos .zip em PATHOLOGY_ZIPS.")
        return

    # --- define a tarefa binaria ---
    print(f"\nTAREFA: {TAREFA}")
    if TAREFA == "saudavel_vs_patologico":
        df_healthy = load_or_build_healthy()
        if df_healthy.empty:
            print("ERRO: nao foi possivel carregar/construir o grupo saudavel.")
            return
        df = pd.concat([df, df_healthy], ignore_index=True)
        df["label"] = np.where(df["label"] == "healthy", "saudavel", "patologico")
    elif TAREFA == "neuro_vs_nao_neuro":
        df = df[df["label"] != "healthy"].copy()
        df["label"] = df["label"].map(LABEL_TO_GROUP)
        df = df.dropna(subset=["label"])
    else:
        print(f"ERRO: TAREFA desconhecida: {TAREFA}")
        return

    print("Distribuicao das classes (gravacoes):")
    print(df["label"].value_counts())
    print("Falantes unicos por classe:")
    print(df.groupby("label")["speaker_id"].nunique())

    # separa colunas de atributos por tipo
    all_cols = [c for c in df.columns if c not in ("label", "speaker_id", "filename")]
    mfcc_cols = [c for c in all_cols if c.startswith("mfcc_") or c == "duration_sec"]
    wavelet_cols = [c for c in all_cols if any(c.startswith(w + "_") for w in WAVELET_FAMILIES)]
    envelope_cols = [c for c in all_cols if c.startswith("envelope_")
                      or c.startswith("tremor_") or c.startswith("envwav_")]
    wpt_cols = [c for c in all_cols if c.startswith("wpt_") or c.startswith("wptt_")]
    hybrid_cols = mfcc_cols + wavelet_cols
    hybrid_all_cols = mfcc_cols + wavelet_cols + envelope_cols
    hybrid_mfcc_wpt_cols = mfcc_cols + wpt_cols

    all_results = []

    # Experimentos principais deste arquivo:
    # 1) MFCC baseline
    # 2) WPT global + WPT temporal/energia relativa
    # 3) MFCC + WPT global + WPT temporal
    all_results += run_pipeline_for_representation(
        df, mfcc_cols, "MFCC"
    )
    all_results += run_pipeline_for_representation(
        df, wpt_cols, "WPT (global + temporal)"
    )
    all_results += run_pipeline_for_representation(
        df, hybrid_mfcc_wpt_cols, "Hibrido (MFCC + WPT temporal)"
    )

    results_df = pd.DataFrame(all_results).sort_values(
        ["representation", "accuracy"], ascending=[True, False]
    )
    print("\n=== RESUMO FINAL ===")
    print(results_df.to_string(index=False))
    out_csv = OUTPUT_RESULTS_CSV.replace(".csv", f"_{TAREFA}.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"\nResumo salvo em {out_csv}")

    if CONFUSION_BLOCKS:
        with open(OUTPUT_CONFUSION_TXT.replace(".txt", f"_{TAREFA}.txt"), "w", encoding="utf-8") as f:
            f.write("MATRIZES DE CONFUSAO\n"
                    "linhas = classe real, colunas = classe prevista\n\n")
            f.write("\n\n".join(CONFUSION_BLOCKS) + "\n")
        print("Matrizes de confusao salvas.")


if __name__ == "__main__":
    main()
