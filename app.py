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
# CONFIGURATION
# =====================================================================
HF_TOKEN = os.getenv("HF_TOKEN", "")

app = FastAPI(
    title="Legal AI Assistant",
    description="Legal problem resolution using fine-tuned T5",
    version="1.0"
)

MODEL_PATH = os.getenv("MODEL_PATH", "./saved_legal_model")

def check_model_weights(path: str) -> bool:
    """Check if model weights exist locally or if it's a Hugging Face repo ID."""
    if not os.path.exists(path):
        return False
    if os.path.isdir(path):
        has_weights = any(
            os.path.isfile(os.path.join(path, f))
            for f in ["model.safetensors", "pytorch_model.bin"]
        )
        return has_weights
    return True

# Check if local model has actual trained weights
if not check_model_weights(MODEL_PATH):
    print("=" * 70)
    print(f"[WARNING] Local directory '{MODEL_PATH}' does not contain model weights!")
    print("  Found config files, but 'model.safetensors' or 'pytorch_model.bin' is missing.")
    print("  (GitHub web upload rejects files >25MB, so weights must be copied directly).")
    print("  Falling back to 't5-small' so the server can run without crashing.")
    print("=" * 70)
    MODEL_PATH = "t5-small"

torch.set_grad_enabled(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load Tokenizer with graceful fallback
try:
    tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
except Exception:
    print("Local tokenizer not found, loading 't5-small' tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Load Model
print(f"Loading model from '{MODEL_PATH}'...")
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()
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
    text = re.sub(r"^(legal\s+)?problem:\s*", "", text, flags=re.IGNORECASE)
    return text.strip()

def solve_legal_problem(problem: str) -> str:
    cleaned = clean_data(problem)

    inputs = tokenizer(
        cleaned,
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
        no_repeat_ngram_size=3,
        repetition_penalty=1.2,
        early_stopping=True
    )

    output = tokenizer.decode(targets[0].cpu(), skip_special_tokens=True)
    return output

# API endpoints
@app.post("/solve/")
async def solve(data: LegalProblemInput):
    try:
        if not data.problem.strip():
            return {"problem": data.problem, "output": "Please enter a valid legal problem description."}
        solution = solve_legal_problem(data.problem)
        return {"problem": data.problem, "solution": solution, "output": solution}
    except Exception as e:
        return {"problem": data.problem, "error": str(e), "solution": f"Error during generation: {e}", "output": f"Error during generation: {e}"}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    index_file = os.path.join("templates", "index.html")
    if os.path.exists(index_file):
        return templates.TemplateResponse(request=request, name="index.html", context={})
    return HTMLResponse("<h2>Legal AI Assistant API is running! Visit /docs for the API UI.</h2>")

@app.get("/status/")
async def status():
    return {
        "status": "ready",
        "model_loaded": MODEL_PATH,
        "device": str(device),
        "has_weights": check_model_weights("./saved_legal_model")
    }
