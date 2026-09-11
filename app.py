from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re
import gc
import os
import logging
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# =====================================================================
# CONFIGURATION & API KEY (Set your manual keys here if needed)
# =====================================================================
HF_TOKEN = ""  # Enter your Hugging Face API key/token here (optional)

if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

app = FastAPI(title="Legal AI Assistant", description="Legal problem resolution using fine-tuned T5", version="1.0")

MODEL_PATH = os.getenv("MODEL_PATH", "./saved_legal_model")

# If local model not trained yet, fallback to base model
if not os.path.exists(os.path.join(MODEL_PATH, "config.json")):
    print(f"Warning: Local model at '{MODEL_PATH}' not found. Using 't5-small' base model for now.")
    MODEL_PATH = "t5-small"

torch.set_grad_enabled(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
model.to(device)
gc.collect()

# Setup templates and static directory if present
os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

class LegalProblemInput(BaseModel):
    problem: str

def clean_data(text: str) -> str:
    text = re.sub(r"\r\n", " ", str(text))
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    return text.strip()

def solve_legal_problem(problem: str) -> str:
    cleaned = clean_data(problem)
    input_text = "legal problem: " + cleaned

    inputs = tokenizer(
        input_text,
        padding="max_length",
        max_length=128,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    targets = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=384,
        num_beams=4,
        early_stopping=True
    )

    output = tokenizer.decode(targets[0].cpu(), skip_special_tokens=True)
    return output

# API endpoints
@app.post("/solve/")
async def solve(data: LegalProblemInput):
    solution = solve_legal_problem(data.problem)
    return {"problem": data.problem, "output": solution}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    index_file = os.path.join("templates", "index.html")
    if os.path.exists(index_file):
        return templates.TemplateResponse(request=request, name="index.html", context={})
    return HTMLResponse("<h2>Legal AI Assistant API is running! Visit /docs for the API UI.</h2>")
