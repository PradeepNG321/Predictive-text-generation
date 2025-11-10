# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from backend.predict_baseline import NGramModel
from backend.predict_transformer import transformer_predict


app = FastAPI(title="Predictive Text - Baseline API")

# allow local dev origin(s)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class PredictRequest(BaseModel):
    text: str
    k: int = 5

class PredictResponse(BaseModel):
    suggestions: list[str]

MODEL = NGramModel(n=3)

@app.on_event("startup")
def load_model():
    data_path = Path("data/corpus.txt")
    if not data_path.exists():
        sample = [
            "hello how are you",
            "hello how is your project",
            "i am fine thank you",
            "how are you doing today",
            "what is your name",
            "this project is about predictive text",
            "predictive text suggestions help typing faster",
        ]
        data_path.parent.mkdir(parents=True, exist_ok=True)
        data_path.write_text("\n".join(sample), encoding="utf8")
        print("Created sample data at", data_path)
    with data_path.open("r", encoding="utf8") as f:
        lines = [l.strip() for l in f if l.strip()]
    MODEL.train_from_lines(lines)
    print(f"Trained n-gram model on {len(lines)} lines.")

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    suggestions = MODEL.predict(req.text, k=req.k)
    return {"suggestions": suggestions}

@app.post("/predict_transformer")
def predict_transformer(req: PredictRequest):
    suggestions = transformer_predict(req.text, k=req.k)
    return {"suggestions": suggestions}


@app.get("/")
def root():
    return {"msg": "Predictive Text Baseline API is running. POST /predict with JSON {'text':..., 'k':...}"}

# Note: keep the __main__ block (optional) — not required for uvicorn, but harmless.
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
