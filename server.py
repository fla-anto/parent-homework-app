import os
import base64
import json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import anthropic
import uvicorn

app = FastAPI(title="Guida API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="."), name="static")


@app.get("/")
def root():
    return FileResponse("index.html")


def get_client():
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise HTTPException(
            status_code=500,
            detail="ANTHROPIC_API_KEY non configurata sul server."
        )

    cleaned = key.strip()

    if not cleaned:
        raise HTTPException(
            status_code=500,
            detail="ANTHROPIC_API_KEY vuota o non valida."
        )

    return anthropic.Anthropic(api_key=cleaned)


ANALYZE_SYSTEM_PROMPT = """Sei Guida, un assistente per genitori italiani che devono aiutare i figli della scuola primaria con i compiti.

Analizzi l'immagine di un compito e rispondi SOLO al genitore, aiutandolo a capire rapidamente:
- di cosa parla il compito
- come spiegarlo al figlio in modo semplice
- quali errori evitare
- come verificare che il figlio abbia capito davvero

STILE:
- Linguaggio molto semplice e naturale
- Frasi brevi
- Tono pratico, chiaro, utile
- Niente tecnicismi inutili
- Niente spiegazioni troppo lunghe
- Più precisione, più coesione, più sintesi

REGOLE:
- NON dare mai la risposta finale del compito
- NON risolvere l'esercizio
- Spiega il metodo al genitore, non la soluzione al bambino
- Usa esempi pratici e quotidiani
- Adatta il linguaggio alla classe del bambino
- Usa il nome del bambino se disponibile

Rispondi SOLO con JSON valido, senza markdown, senza testo fuori dal JSON.

Formato esatto:
{
  "topic": "matematica|italiano|storia|geografia|scienze|lingue|altro",
  "subject": "Argomento specifico breve",
  "what": "Spiegazione introduttiva molto semplice e sintetica in massimo 2 frasi",
  "how": [
    "Primo passaggio concreto: cosa può dire il genitore",
    "Secondo passaggio concreto: esempio semplice e quotidiano",
    "Terzo passaggio concreto: mini esercizio o prova guidata"
  ],
  "watch": [
    "Errore comune numero 1",
    "Errore comune numero 2"
  ],
  "check": "Una domanda semplice che il genitore può fare per verificare se il bambino ha capito davvero"
}
"""

CHAT_SYSTEM_PROMPT = """Sei Guida, un assistente per genitori italiani con figli della scuola primaria.

Il genitore ti fa domande di follow-up su un compito già analizzato.
Tu devi aiutare il genitore a spiegare meglio il concetto al figlio.

STILE:
- Linguaggio semplice, chiaro, concreto
- Frasi brevi
- Tono calmo e pratico
- Niente teoria lunga
- Risposte coese e facili da capire

REGOLE:
- Parla SEMPRE al genitore
- NON dare mai la risposta finale dell'esercizio
- NON svolgere il compito
- Aiuta il genitore a spiegare meglio, fare esempi, verificare la comprensione
- Se il genitore chiede direttamente la risposta, rifiuta gentilmente e riporta il focus sul metodo
- Se utile, suggerisci una frase precisa che il genitore può dire al figlio

Rispondi SOLO con JSON valido nel seguente formato:
{
  "reply": "Risposta utile, semplice e pratica per il genitore"
}
"""


def extract_text_from_response(response):
    parts = []
    for block in response.content:
        if hasattr(block, "text"):
            parts.append(block.text)
    return "".join(parts).strip()


@app.post("/analyze")
async def analyze(
    image: UploadFile = File(...),
    child_name: str = Form(...),
    child_class: str = Form(...),
):
    image_bytes = await image.read()
    image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
    media_type = image.content_type or "image/jpeg"

    user_message = (
        f"Il compito è di {child_name}, che frequenta la {child_class}. "
        "Analizza l'immagine e rispondi con il JSON strutturato richiesto."
    )

    client = get_client()

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            system=ANALYZE_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": user_message,
                        },
                    ],
                }
            ],
        )
    except anthropic.APIError as e:
        raise HTTPException(status_code=502, detail=f"Errore Anthropic: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore interno: {str(e)}")

    raw = extract_text_from_response(response)
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=422,
            detail="Risposta non strutturata dal modello. Riprova con un'immagine più chiara."
        )

    return parsed


@app.post("/chat")
async def chat(
    parent_message: str = Form(...),
    child_name: str = Form(...),
    child_class: str = Form(...),
    subject: str = Form(""),
    what: str = Form(""),
    how: str = Form(""),
    watch: str = Form(""),
    check: str = Form(""),
):
    client = get_client()

    context = f"""
Nome bambino: {child_name}
Classe: {child_class}
Argomento: {subject}

Spiegazione introduttiva:
{what}

Come spiegarlo:
{how}

Errori comuni:
{watch}

Verifica comprensione:
{check}

Domanda del genitore:
{parent_message}
"""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=700,
            system=CHAT_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": context,
                }
            ],
        )
    except anthropic.APIError as e:
        raise HTTPException(status_code=502, detail=f"Errore Anthropic: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore interno: {str(e)}")

    raw = extract_text_from_response(response)
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"reply": raw}

    return parsed


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port)
