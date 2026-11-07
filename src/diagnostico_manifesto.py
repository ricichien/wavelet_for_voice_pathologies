from avfad_paths import asset, project_root, raw_dir
import pandas as pd
from collections import Counter

arquivo = asset(project_root(), "metadata_capev_avfad.csv")

df = pd.read_csv(arquivo)

pessoas = df.drop_duplicates("participant_id")

print("\n=== MANIFESTO ===")
print("Gravacoes:", len(df))
print("Participantes:", df["participant_id"].nunique())

print("\n=== PARTICIPANTES POR CLASSE ===")
print(pessoas["label"].value_counts(dropna=False))

print("\n=== SEXO POR CLASSE ===")
print(pd.crosstab(
    pessoas["sex"],
    pessoas["label"],
    margins=True,
    dropna=False
))

print("\n=== IDADE POR CLASSE ===")
pessoas["age"] = pd.to_numeric(
    pessoas["age"],
    errors="coerce"
)

print(pessoas.groupby("label")["age"].describe())

print("\n=== LOCAL DE COLETA POR CLASSE ===")
print(pd.crosstab(
    pessoas["visit_place"],
    pessoas["label"],
    margins=True,
    dropna=False
))

print("\n=== GRAVACOES POR TAREFA ===")
print(df["task"].value_counts().sort_index())

print("\n=== NUMERO DE FRASES POR PARTICIPANTE ===")
tarefas = df.groupby("participant_id")["task"].nunique()

print(tarefas.value_counts().sort_index())

print("\n=== PRIMEIRAS 15 LINHAS ===")
colunas = [
    "sample_id",
    "participant_id",
    "task",
    "label",
    "age",
    "sex",
    "visit_year",
    "visit_place",
]

print(df[colunas].head(15).to_string(index=False))