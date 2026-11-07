
from avfad_paths import asset, project_root, raw_dir
import csv
import io
import re
import wave
import zipfile

from pathlib import Path
from collections import Counter, defaultdict
from statistics import median

from openpyxl import load_workbook


ROOT = project_root()

XLSX_NAME = "AVFAD_01_00_00.xlsx"
SHEET_NAME = "AVFAD"

REPORT = asset(ROOT, "resumo_validacao.txt")
CSV_OUTPUT = asset(ROOT, "validacao_tarefa_002.csv")


def line(report, text=""):
    report.write(str(text) + "\n")


def clean(value):
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def number(value):
    if value is None:
        return None

    try:
        result = float(value)
        return int(result) if result.is_integer() else result
    except (ValueError, TypeError):
        return None


def find_workbook():
    matches = []

    for zip_path in raw_dir(ROOT).glob("AVFAD_01_00_00_*.zip"):
        with zipfile.ZipFile(zip_path, "r") as archive:
            for info in archive.infolist():
                basename = info.filename.replace("\\", "/").split("/")[-1]

                if basename.lower() == XLSX_NAME.lower():
                    matches.append((zip_path, info.filename))

    if len(matches) != 1:
        raise RuntimeError(
            f"Esperava encontrar uma planilha {XLSX_NAME}, "
            f"mas encontrei {len(matches)}."
        )

    zip_path, internal_path = matches[0]

    with zipfile.ZipFile(zip_path, "r") as archive:
        content = archive.read(internal_path)

    return io.BytesIO(content)


def load_metadata():
    workbook_file = find_workbook()

    workbook = load_workbook(
        workbook_file,
        read_only=True,
        data_only=True
    )

    if SHEET_NAME not in workbook.sheetnames:
        raise RuntimeError(
            f"Aba '{SHEET_NAME}' não encontrada. "
            f"Abas disponíveis: {workbook.sheetnames}"
        )

    sheet = workbook[SHEET_NAME]

    headers = [
        clean(value)
        for value in next(sheet.iter_rows(
            min_row=1,
            max_row=1,
            values_only=True
        ))
    ]

    indexes = {
        name: headers.index(name)
        for name in (
            "File ID",
            "CMVD-I Dimension 1 (numeric system)",
            "CMVD-I Dimension 1 (word system)",
            "Age",
            "Sex",
            "Visit place"
        )
        if name in headers
    }

    required = [
        "File ID",
        "CMVD-I Dimension 1 (numeric system)",
        "CMVD-I Dimension 1 (word system)",
        "Age",
        "Sex",
        "Visit place"
    ]

    missing_columns = [
        name for name in required
        if name not in indexes
    ]

    if missing_columns:
        raise RuntimeError(
            f"Colunas ausentes na planilha: {missing_columns}"
        )

    metadata = {}
    duplicate_ids = []
    invalid_labels = []
    inconsistent_labels = []

    for row_number, row in enumerate(
        sheet.iter_rows(min_row=2, values_only=True),
        start=2
    ):
        participant_id = clean(row[indexes["File ID"]])

        if not participant_id:
            continue

        participant_id = participant_id.upper()

        code = number(
            row[indexes["CMVD-I Dimension 1 (numeric system)"]]
        )

        clinical_label = clean(
            row[indexes["CMVD-I Dimension 1 (word system)"]]
        )

        if code is None:
            label = None
            invalid_labels.append(
                f"Linha {row_number}: {participant_id}, código vazio/inválido"
            )
        else:
            label = "Normal" if code == 0 else "Pathological"

            if clinical_label:
                text_is_normal = (
                    clinical_label.casefold() == "normal"
                )

                if text_is_normal != (code == 0):
                    inconsistent_labels.append(
                        f"{participant_id}: código={code}, "
                        f"texto={clinical_label}"
                    )

        if participant_id in metadata:
            duplicate_ids.append(participant_id)
            continue

        metadata[participant_id] = {
            "participant_id": participant_id,
            "numeric_code": code,
            "clinical_label": clinical_label,
            "label": label,
            "age": number(row[indexes["Age"]]),
            "sex": clean(row[indexes["Sex"]]),
            "visit_place": clean(row[indexes["Visit place"]])
        }

    workbook.close()

    return (
        metadata,
        duplicate_ids,
        invalid_labels,
        inconsistent_labels
    )


def inspect_audio_archives():
    audio = {}
    errors = []
    duplicates = []
    malformed_names = []

    formats = Counter()

    for zip_path in sorted(raw_dir(ROOT).glob("AVFAD_01_00_00_*.zip")):

        with zipfile.ZipFile(zip_path, "r") as archive:

            for info in archive.infolist():

                if info.is_dir():
                    continue

                basename = (
                    info.filename.replace("\\", "/").split("/")[-1]
                )

                if not basename.upper().endswith("002.WAV"):
                    continue

                match = re.fullmatch(
                    r"([A-Z]{3})002\.WAV",
                    basename.upper()
                )

                if not match:
                    malformed_names.append(info.filename)
                    continue

                participant_id = match.group(1)

                if participant_id in audio:
                    duplicates.append(participant_id)
                    continue

                try:
                    with archive.open(info) as raw:
                        with wave.open(raw, "rb") as wav:

                            channels = wav.getnchannels()
                            sample_rate = wav.getframerate()
                            sample_width = wav.getsampwidth()
                            frames = wav.getnframes()
                            compression = wav.getcomptype()

                            duration = (
                                frames / sample_rate
                                if sample_rate > 0 else 0
                            )

                            formats[
                                (
                                    channels,
                                    sample_rate,
                                    sample_width * 8,
                                    compression
                                )
                            ] += 1

                            audio[participant_id] = {
                                "audio_archive": zip_path.name,
                                "audio_path": info.filename,
                                "duration_s": round(duration, 4),
                                "sample_rate_hz": sample_rate,
                                "channels": channels,
                                "bits_per_sample": sample_width * 8,
                                "frames": frames,
                                "compression": compression
                            }

                except Exception as error:
                    errors.append(
                        f"{info.filename}: {error}"
                    )

    return audio, formats, errors, duplicates, malformed_names


def age_group(age):
    if age is None:
        return "Sem informação"
    if age < 30:
        return "Menos de 30"
    if age < 50:
        return "30 a 49"
    if age < 70:
        return "50 a 69"
    return "70 ou mais"


def cross_table(report, records, field, title):

    table = defaultdict(Counter)

    for record in records:
        value = record.get(field)
        category = clean(value) or "Sem informação"
        label = record.get("label") or "Rótulo inválido"

        table[category][label] += 1

    line(report, f"\n{title}")
    line(report, f"{'Categoria':<24} {'Normal':>10} {'Pathological':>15}")

    for category in sorted(table):
        normal = table[category]["Normal"]
        pathological = table[category]["Pathological"]

        line(
            report,
            f"{category:<24} {normal:>10} {pathological:>15}"
        )


def save_csv(metadata, audio, duplicates_metadata):
    fields = [
        "participant_id",
        "label",
        "numeric_code",
        "clinical_label",
        "age",
        "sex",
        "visit_place",
        "audio_found",
        "audio_archive",
        "audio_path",
        "duration_s",
        "sample_rate_hz",
        "channels",
        "bits_per_sample",
        "frames",
        "compression"
    ]

    all_ids = sorted(set(metadata) | set(audio))

    with open(
        CSV_OUTPUT,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for participant_id in all_ids:

            meta = metadata.get(participant_id, {})
            wav = audio.get(participant_id, {})

            writer.writerow({
                "participant_id": participant_id,
                "label": meta.get("label"),
                "numeric_code": meta.get("numeric_code"),
                "clinical_label": meta.get("clinical_label"),
                "age": meta.get("age"),
                "sex": meta.get("sex"),
                "visit_place": meta.get("visit_place"),
                "audio_found": participant_id in audio,
                "audio_archive": wav.get("audio_archive"),
                "audio_path": wav.get("audio_path"),
                "duration_s": wav.get("duration_s"),
                "sample_rate_hz": wav.get("sample_rate_hz"),
                "channels": wav.get("channels"),
                "bits_per_sample": wav.get("bits_per_sample"),
                "frames": wav.get("frames"),
                "compression": wav.get("compression")
            })


def main():

    print("Lendo metadados...")
    (
        metadata,
        duplicate_metadata,
        invalid_labels,
        inconsistent_labels
    ) = load_metadata()

    print("Verificando áudios 002...")
    (
        audio,
        formats,
        audio_errors,
        duplicate_audio,
        malformed_names
    ) = inspect_audio_archives()

    metadata_ids = set(metadata)
    audio_ids = set(audio)

    missing_audio = sorted(metadata_ids - audio_ids)
    extra_audio = sorted(audio_ids - metadata_ids)

    matched = [
        metadata[pid]
        for pid in sorted(metadata_ids & audio_ids)
    ]

    valid_metadata = [
        record for record in metadata.values()
        if record["label"] is not None
    ]

    normal_count = sum(
        r["label"] == "Normal"
        for r in valid_metadata
    )

    pathological_count = sum(
        r["label"] == "Pathological"
        for r in valid_metadata
    )

    with open(REPORT, "w", encoding="utf-8") as report:

        line(report, "VALIDAÇÃO DA AVFAD — TAREFA 002")
        line(report, "=" * 55)

        line(report, "\n1. METADADOS")
        line(report, f"Participantes na planilha: {len(metadata)}")
        line(report, f"Duplicados na planilha: {len(duplicate_metadata)}")
        line(report, f"Rótulos inválidos: {len(invalid_labels)}")
        line(report, f"Inconsistências código/texto: {len(inconsistent_labels)}")
        line(report, f"Normais: {normal_count}")
        line(report, f"Patológicos: {pathological_count}")

        code_counts = Counter(
            str(r["numeric_code"])
            for r in valid_metadata
        )

        line(report, "\nCódigos clínicos mais frequentes:")
        for code, count in code_counts.most_common(15):
            line(report, f"  Código {code}: {count}")

        if inconsistent_labels:
            line(report, "\nInconsistências:")
            for item in inconsistent_labels[:20]:
                line(report, f"  {item}")

        if invalid_labels:
            line(report, "\nRótulos inválidos:")
            for item in invalid_labels[:20]:
                line(report, f"  {item}")

        line(report, "\n2. ÁUDIOS")
        line(report, f"Arquivos 002 encontrados: {len(audio)}")
        line(report, f"Duplicados: {len(duplicate_audio)}")
        line(report, f"Nomes fora do padrão: {len(malformed_names)}")
        line(report, f"Erros de leitura: {len(audio_errors)}")

        line(report, "\nCorrespondência:")
        line(report, f"Participantes sem áudio 002: {len(missing_audio)}")
        line(report, f"Áudios sem participante na planilha: {len(extra_audio)}")
        line(report, f"Participantes com áudio e metadados: {len(matched)}")

        line(report, "\nFormatos:")
        for fmt, count in formats.most_common():
            channels, rate, bits, compression = fmt
            line(
                report,
                f"  {count} arquivos: {channels} canal(is), "
                f"{rate} Hz, {bits} bits, {compression}"
            )

        durations = [
            audio[pid]["duration_s"]
            for pid in audio
        ]

        if durations:
            line(report, "\nDurações:")
            line(report, f"  Mínima: {min(durations):.2f}s")
            line(report, f"  Mediana: {median(durations):.2f}s")
            line(report, f"  Máxima: {max(durations):.2f}s")

            unusual = [
                (pid, audio[pid]["duration_s"])
                for pid in sorted(audio)
                if audio[pid]["duration_s"] < 5
                or audio[pid]["duration_s"] > 30
            ]

            line(
                report,
                f"\nGravações para revisão (<5s ou >30s): {len(unusual)}"
            )

            for pid, duration in unusual[:30]:
                line(report, f"  {pid}: {duration:.2f}s")

        if missing_audio:
            line(report, "\nParticipantes sem áudio:")
            line(report, ", ".join(missing_audio[:50]))

        if extra_audio:
            line(report, "\nÁudios sem metadados:")
            line(report, ", ".join(extra_audio[:50]))

        if audio_errors:
            line(report, "\nErros:")
            for item in audio_errors[:20]:
                line(report, f"  {item}")

        line(report, "\n3. DISTRIBUIÇÃO DOS PARTICIPANTES")

        cross_table(report, matched, "sex", "Classe por sexo")
        cross_table(report, matched, "visit_place", "Classe por local de gravação")

        age_records = []

        for record in matched:
            copy = dict(record)
            copy["age_group"] = age_group(record["age"])
            age_records.append(copy)

        cross_table(
            report,
            age_records,
            "age_group",
            "Classe por faixa etária"
        )

        line(report, "\n4. RESUMO")
        line(report, f"IDs únicos na planilha: {len(metadata_ids)}")
        line(report, f"IDs únicos nos áudios: {len(audio_ids)}")
        line(report, f"Correspondências válidas: {len(matched)}")
        line(report, f"Classes válidas: {normal_count + pathological_count}")

    save_csv(metadata, audio, duplicate_metadata)

    print("\nConcluído.")
    print(f"Relatório: {REPORT}")
    print(f"CSV: {CSV_OUTPUT}")
    print("\nEnvie somente o conteúdo de resumo_validacao.txt.")


if __name__ == "__main__":
    main()
