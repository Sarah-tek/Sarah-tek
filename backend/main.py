from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
import pytesseract
from PIL import Image
import io

app = FastAPI()

# CORS for local frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model from Hugging Face
ner_model = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)

@app.post("/verify/")
async def verify_prescription(file: UploadFile = File(...)):
    contents = await file.read()

    try:
        # Convert image bytes to text
        image = Image.open(io.BytesIO(contents))
        extracted_text = pytesseract.image_to_string(image)

        # Run NER
        results = ner_model(extracted_text)
        medicines = [ent['word'] for ent in results if ent['entity_group'] == 'MISC']

        return {"status": "success", "medicines": medicines, "raw_text": extracted_text}

    except Exception as e:
        return {"status": "error", "message": str(e)}