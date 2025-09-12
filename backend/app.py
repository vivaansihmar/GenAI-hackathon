from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import fitz  # PyMuPDF

import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for dev, allow all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_text_from_pdf(file_bytes):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

@app.post("/summarize")
async def summarize(file: UploadFile):
    content = await file.read()
    text = extract_text_from_pdf(content)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"Summarize this legal document in plain language:\n\n{text[:3000]}"}]
    )
    return {"summary": response.choices[0].message.content}

@app.post("/analyze_risks")
async def analyze_risks(file: UploadFile):
    content = await file.read()
    text = extract_text_from_pdf(content)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"Find risky clauses and explain simply:\n\n{text[:3000]}"}]
    )
    return {"risks": response.choices[0].message.content}
