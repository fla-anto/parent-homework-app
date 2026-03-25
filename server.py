"""
server.py — Backend Guida
FastAPI che fa da proxy sicuro verso l'API Anthropic.
La API key resta qui sul server, il frontend non la vede mai.

Avvio:
  pip install fastapi uvicorn anthropic python-multipart
  ANTHROPIC_API_KEY=sk-ant-... python server.py
"""

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

# CORS — permette al frontend di chiamare il backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Serve il frontend HTML come file statico sulla root
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/")
def root():
    return FileResponse("index.html")

# ── Client Anthropic (legge la key dall'env) ──────────────────────────────────
def get_client():
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise HTTPException(
            status_code=500,
            detail="ANTHROPIC_API_KEY non configurata sul server."
        )
    return anthropic.Anthropic(api_key=key)

# ── System prompt condiviso ───────────────────────────────────────────────────
SYSTEM_PROMPT = """Sei Guida, un assistente pensato per i genitori italiani che vogliono aiutare i propri figli a fare e capire i compiti.

Il tuo ruolo è quello di un amico insegnante che parla direttamente al genitore — non al bambino.
Non dai la risposta al compito. Insegni al genitore come diventare la guida giusta per quel compito specifico.

COSA FAI:
1. Guardi la foto del compito e capisci subito di che materia e argomento si tratta
2. Spieghi al genitore l'argomento in modo semplice, come se non lo vedesse da anni
3. Gli dai strategie pratiche e frasi concrete da usare con il figlio, adattate all'età
4. Lo avverti degli errori tipici che i bambini fanno con quell'argomento
5. Gli suggerisci come verificare che il figlio abbia davvero capito

REGOLE:
- Parla sempre al genitore, mai al bambino
- Usa il nome del bambino per rendere tutto più personale
- Adatta il linguaggio e gli esempi all'età e alla classe indicata
- Usa esempi della vita quotidiana (la pizza per le frazioni, il pallone per la fisica ecc.)
- Scrivi le frasi esatte che il genitore può dire — non consigli vaghi
- NON dare mai la soluzione del compito

Rispondi ESCLUSIVAMENTE con un oggetto JSON valido. Niente markdown, niente backtick, niente testo fuori dal JSON.
{
  "topic": "matematica|italiano|storia|geografia|scienze|lingue|altro",
  "subject": "Nome breve e chiaro dell'argomento (es: Le frazioni, Il complemento oggetto, La Seconda Guerra Mondiale)",
  "what": "Spiegazione dell'argomento per il genitore in 2-3 righe chiare e dirette, come se lo spiegasse un amico insegnante",
  "how": [
    "Strategia 1: frase esatta da dire al figlio con esempio concreto della vita quotidiana",
    "Strategia 2: secondo approccio diverso, con metodo o esempio alternativo",
    "Strategia 3: terzo suggerimento pratico"
  ],
  "watch": [
    "Errore tipico 1 che i bambini di questa età fanno con questo specifico argomento",
    "Errore tipico 2 da tenere d'occhio"
  ],
  "check": "Una domanda precisa e semplice che il genitore può fare al figlio adesso per capire se ha davvero compreso il concetto, non solo memorizzato la risposta"
}"""


@app.post("/analyze")
async def analyze(
    image: UploadFile = File(...),
    child_name: str = Form(...),
    child_class: str = Form(...),
):
    """
    Riceve: immagine del compito + nome figlio + classe
    Ritorna: JSON strutturato con spiegazione per il genitore
    """
    # Leggi e converti l'immagine in base64
    image_bytes = await image.read()
    image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
    media_type = image.content_type or "image/jpeg"

    user_message = f"Il compito è di {child_name}, che frequenta la {child_class}. Analizza l'immagine e rispondi con il JSON strutturato."

    client = get_client()

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            system=SYSTEM_PROMPT,
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

    raw = response.content[0].text.strip()

    # Pulizia nel caso il modello aggiunga backtick
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=422,
            detail="Risposta non strutturata dal modello. Riprova con un'immagine più chiara."
        )

    return parsed


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port)
