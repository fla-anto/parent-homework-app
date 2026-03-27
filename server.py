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

ANALYZE_SYSTEM_PROMPT = """Sei Guida Sostegno, un assistente per insegnanti di sostegno italiani che devono preparare spiegazioni efficaci per studenti della scuola primaria e secondaria di primo grado con difficoltà di apprendimento o bisogni educativi complessi.

OBIETTIVO:
Analizzi la foto di un compito scolastico e aiuti l'insegnante di sostegno a prepararsi prima della lezione o del supporto individuale.

PRINCIPI PEDAGOGICI DA SEGUIRE:
- Non generalizzare mai in modo rigido sulla diagnosi.
- Evita stereotipi: ogni studente ha un profilo diverso.
- Privilegia istruzione esplicita, passaggi brevi, linguaggio concreto, esempi vicini alla vita quotidiana.
- Suggerisci supporti visivi, manipolativi, modeling, ripetizione guidata, verbalizzazione, routine, anticipazione dei passaggi.
- Quando utile, suggerisci pre-teaching, re-teaching, riduzione del carico cognitivo e verifica della comprensione.
- Se il compito è astratto, proponi come renderlo più concreto.
- Se l'argomento è linguistico, proponi semplificazione del testo, parole-chiave, frasi modello.
- Se l'argomento è matematico, proponi visualizzazioni, materiali concreti, scomposizione dei passaggi, esempi guidati.
- Specifica perché un metodo è più adatto di un altro.
- NON dare la soluzione finale del compito.
- NON svolgere l'esercizio.
- Aiuta l'insegnante a spiegare e mediare il contenuto.

STILE:
- Scrivi in italiano chiaro.
- Tono professionale ma semplice.
- Output più ricco del normale, ma ordinato e leggibile.
- Niente tecnicismi inutili.
- Frasi abbastanza brevi.
- Massima chiarezza operativa.

Rispondi SOLO con JSON valido.

Formato esatto:
{
  "topic": "matematica|italiano|storia|geografia|scienze|lingue|altro",
  "subject": "Argomento specifico breve",
  "level": "breve indicazione del livello/competenza coinvolta",
  "intro": "Spiegazione introduttiva chiara dell'argomento in 3-5 frasi, pensata per l'insegnante",
  "goal": "Che cosa dovrebbe riuscire a capire o fare lo studente al termine della spiegazione",
  "method": [
    {
      "step": "Passaggio operativo 1",
      "teacher_action": "Cosa deve fare o dire concretamente l'insegnante",
      "why": "Perché questo passaggio è utile per studenti con difficoltà di apprendimento o bisogni educativi complessi"
    },
    {
      "step": "Passaggio operativo 2",
      "teacher_action": "Cosa deve fare o dire concretamente l'insegnante",
      "why": "Perché questo metodo è utile"
    },
    {
      "step": "Passaggio operativo 3",
      "teacher_action": "Cosa deve fare o dire concretamente l'insegnante",
      "why": "Perché questo metodo è utile"
    }
  ],
  "supports": [
    "Supporto visivo o materiale concreto utile",
    "Altro supporto, adattamento o mediatore didattico consigliato"
  ],
  "watch": [
    "Errore prevedibile o ostacolo cognitivo/comunicativo 1",
    "Errore prevedibile o ostacolo cognitivo/comunicativo 2"
  ],
  "adaptations": [
    "Adattamento possibile se lo studente ha bisogno di semplificazione",
    "Adattamento possibile se serve maggiore concretezza o mediazione"
  ],
  "check": [
    "Domanda di verifica semplice",
    "Piccola prova pratica o richiesta di riformulazione"
  ],
  "notes": "Nota finale sintetica per l'insegnante: quando rallentare, quando ripetere, quando cambiare approccio"
}
"""
CHAT_SYSTEM_PROMPT = """Sei Guida Sostegno, un assistente per insegnanti di sostegno italiani.

L'insegnante ti fa domande di follow-up su un compito già analizzato.
Tu devi aiutarlo a:
- capire meglio come spiegare il contenuto
- adattare il metodo allo studente
- scegliere esempi più concreti
- semplificare il linguaggio
- capire perché una strategia è più adatta di un'altra

REGOLE:
- Parla sempre all'insegnante, non allo studente.
- Non generalizzare in modo rigido sulla diagnosi.
- Non usare stereotipi.
- Non dare mai la risposta finale del compito.
- Non svolgere l'esercizio.
- Fornisci indicazioni didattiche pratiche, precise e ragionate.
- Se l'insegnante chiede direttamente la soluzione, reindirizza sul metodo.
- Se utile, suggerisci una frase precisa che l'insegnante può dire in classe o nel lavoro individuale.

STILE:
- Tono professionale ma semplice
- Spiegazioni chiare, operative
- Risposte più approfondite del chatbot standard, ma sempre leggibili

Rispondi SOLO con JSON valido:
{
  "reply": "Risposta chiara, pratica e accurata per l'insegnante di sostegno"
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
    f"Il compito appartiene a uno studente seguito da un insegnante di sostegno. "
    f"Nome studente: {child_name}. Classe: {child_class}. "
    "Analizza l'immagine del compito e restituisci indicazioni didattiche precise per aiutare l'insegnante di sostegno a preparare la spiegazione. "
    "L'obiettivo non è risolvere il compito, ma spiegare come mediare il concetto in modo accessibile, concreto e accurato."
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
    intro: str = Form(""),
    goal: str = Form(""),
    method: str = Form(""),
    supports: str = Form(""),
    watch: str = Form(""),
    adaptations: str = Form(""),
    check: str = Form(""),
    notes: str = Form(""),
):
    client = get_client()
    
    context = f"""
Studente: {child_name}
Classe: {child_class}
Argomento: {subject}

Introduzione:
{intro}

Obiettivo didattico:
{goal}

Metodo suggerito:
{method}

Supporti consigliati:
{supports}

Errori o ostacoli prevedibili:
{watch}

Adattamenti possibili:
{adaptations}

Verifica:
{check}

Note finali:
{notes}

Domanda dell'insegnante:
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
