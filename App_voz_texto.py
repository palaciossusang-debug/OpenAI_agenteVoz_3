"""
Agente de transcripción y extracción para incidencias de mantenimiento en socavones.

1) Lee audios desde ./incidencias_audio/  (wav/mp3/m4a/ogg)
2) Transcribe con OpenAI STT (gpt-4o-mini-transcribe o whisper-1)
3) Extrae campos estructurados con un LLM
4) Guarda CSV con: archivo, fecha, equipo, id_equipo, componente, modo_falla, sintomas,
   severidad, riesgo_seguridad, acciones_sugeridas, tiempo_fuera_servicio_est, ubicacion,
   resumen, transcripcion

Docs oficiales (Audio STT / modelos):
- Speech-to-Text (Audio API): https://platform.openai.com/docs/guides/audio
- Modelos STT (gpt-4o-transcribe / gpt-4o-mini-transcribe / whisper-1)
"""

import os
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from openai import OpenAI
from pydub import AudioSegment
from tqdm import tqdm

# ========= Config =========
#INPUT_DIR = Path("./incidencias_audio")
INPUT_DIR = Path(D:/Audio)
OUTPUT_DIR = Path("./salidas")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUTPUT_DIR / "incidencias_transcritas.csv"

# Elige el modelo de transcripción:
STT_MODEL = os.getenv("STT_MODEL", "gpt-4o-mini-transcribe")  # alternativas: "gpt-4o-transcribe", "whisper-1"
# Modelo para extracción estructurada y resúmenes:
NLP_MODEL = os.getenv("NLP_MODEL", "gpt-4o-mini")  # rápido y económico para tasks de extracción/resumen

# Extensiones de audio soportadas
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".ogg", ".flac"}

# ========= Cliente OpenAI =========
client = OpenAI(api_key=os.getenv("SP_API_KEY_VOZ"))

# ========= Utilidades =========
def ensure_wav(path: Path) -> Path:
    """
    Convierte a WAV si es necesario (mejora compatibilidad).
    Retorna la ruta del archivo WAV (original o convertido temporalmente).
    """
    if path.suffix.lower() == ".wav":
        return path
    audio = AudioSegment.from_file(str(path))
    wav_path = path.with_suffix(".wav")
    audio.export(str(wav_path), format="wav")
    return wav_path

def transcribe_audio(audio_path: Path) -> str:
    """
    Transcribe usando la API de Audio de OpenAI.
    """
    wav_path = ensure_wav(audio_path)
    with open(wav_path, "rb") as f:
        # API oficial de transcripciones (modelos gpt-4o-*-transcribe o whisper-1)
        # https://platform.openai.com/docs/api-reference/audio
        resp = client.audio.transcriptions.create(
            model=STT_MODEL,
            file=f,
            # Si quieres timestamped words/paragraphs cuando esté disponible:
            # response_format="json"
        )
    # En SDKs recientes, el texto viene en .text
    return resp.text if hasattr(resp, "text") else str(resp)

def extract_structured(transcript: str) -> Dict[str, Any]:
    """
    Extrae campos clave del texto con un LLM (funciona bien para audio ruidoso con jerga).
    """
    system = (
        "Eres un ingeniero de confiabilidad en minería subterránea. "
        "Devuelve un JSON con los campos solicitados. Si faltan datos, deja ''. "
        "Severidad ∈ {baja, media, alta, crítica}. Riesgo_seguridad ∈ {bajo, medio, alto}. "
        "Modo de falla usa terminología de mantenimiento (p.ej., fuga, sobrecalentamiento, "
        "vibración excesiva, falla hidráulica, corte eléctrico, desgaste de zapatas...)."
    )
    user = f"""
Transcripción de una incidencia reportada por un supervisor en un socavón:

\"\"\"{transcript}\"\"\"

Devuelve un JSON con estas claves:
- equipo                (ej.: scooptram, jumbo, winche, ventilador, compresor)
- id_equipo             (placa o tag del equipo, si se menciona)
- componente            (ej.: motor, bomba hidráulica, transmisión, frenos, batería)
- modo_falla            (texto corto)
- sintomas              (lista breve separada por punto y coma)
- severidad             (baja|media|alta|crítica)
- riesgo_seguridad      (bajo|medio|alto)
- acciones_sugeridas    (lista breve separada por punto y coma)
- tiempo_fuera_servicio_est (en horas o rango si se menciona)
- ubicacion             (nivel/galería/socavón)
- resumen               (1-2 frases ejecutivas)
"""
    resp = client.chat.completions.create(
        model=NLP_MODEL,
        response_format={"type": "json_object"},  # fuerza JSON
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        temperature=0.2,
    )
    import json
    content = resp.choices[0].message.content
    try:
        data = json.loads(content)
    except Exception:
        data = {"resumen": content}
    return data

def write_csv(rows: List[Dict[str, Any]]):
    fieldnames = [
        "archivo", "fecha_procesado", "equipo", "id_equipo", "componente", "modo_falla",
        "sintomas", "severidad", "riesgo_seguridad", "acciones_sugeridas",
        "tiempo_fuera_servicio_est", "ubicacion", "resumen", "transcripcion"
    ]
    write_header = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)

def process_folder():
    audio_files = [p for p in INPUT_DIR.glob("*") if p.suffix.lower() in AUDIO_EXTS]
    if not audio_files:
        print(f"No se encontraron audios en {INPUT_DIR.resolve()}")
        return
    rows = []
    for path in tqdm(audio_files, desc="Procesando audios"):
        try:
            transcript = transcribe_audio(path)
            info = extract_structured(transcript)
            rows.append({
                "archivo": path.name,
                "fecha_procesado": datetime.now().isoformat(timespec="seconds"),
                "equipo": info.get("equipo", ""),
                "id_equipo": info.get("id_equipo", ""),
                "componente": info.get("componente", ""),
                "modo_falla": info.get("modo_falla", ""),
                "sintomas": info.get("sintomas", ""),
                "severidad": info.get("severidad", ""),
                "riesgo_seguridad": info.get("riesgo_seguridad", ""),
                "acciones_sugeridas": info.get("acciones_sugeridas", ""),
                "tiempo_fuera_servicio_est": info.get("tiempo_fuera_servicio_est", ""),
                "ubicacion": info.get("ubicacion", ""),
                "resumen": info.get("resumen", ""),
                "transcripcion": transcript
            })
        except Exception as e:
            print(f"[WARN] Error procesando {path.name}: {e}")
    if rows:
        write_csv(rows)
        print(f"✅ Listo. Registros añadidos en: {CSV_PATH.resolve()}")

if __name__ == "__main__":
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    process_folder()
