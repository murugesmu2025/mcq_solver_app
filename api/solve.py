import os
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# Enable Cross-Origin Resource Sharing (CORS) so your frontend can call your backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration — Enter your Hugging Face credentials here
# Best Practice: Set HF_TOKEN as an Environment Variable in Vercel settings!
HF_TOKEN = os.environ.get("HF_TOKEN", "your_huggingface_read_token_here")
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

MODEL_URLS = [
    "https://huggingface.co",
    "https://huggingface.co",
    "https://huggingface.co"
]

class MCQRequest(BaseModel):
    question: str
    option_a: str
    option_b: str
    option_c: str
    option_d: str
    option_e: str

def query_hf_api(url, payload):
    try:
        response = requests.post(url, headers=HEADERS, json=payload, timeout=15)
        # Handle cases where Hugging Face is downloading or spinning up the model
        if response.status_code == 503:
            raise HTTPException(status_code=503, detail="Hugging Face model is currently loading. Please retry in 20 seconds.")
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"HF Hub API Error: {response.text}")
        return response.json()
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Hugging Face connection timed out.")

@app.post("/")
def solve_mcq(data: MCQRequest):
    options = [data.option_a, data.option_b, data.option_c, data.option_d, data.option_e]
    
    # Payload format parsed by Hugging Face's pipeline for Multiple Choice / Text Classification
    payload = {
        "inputs": f"Question: {data.question} Options: A) {data.option_a} B) {data.option_b} C) {data.option_c} D) {data.option_d} E) {data.option_e}"
    }

    all_probabilities = []
    
    for url in MODEL_URLS:
        result = query_hf_api(url, payload)
        
        # Parse output pipeline (Assuming typical classification distribution list output)
        try:
            # Sort scores numerically to maintain predictable mapping index order
            if isinstance(result, list) and isinstance(result[0], list):
                result = result[0] # Unwrap nested arrays if present
            scores = [item['score'] for item in sorted(result, key=lambda x: x['label'])]
            
            # Fallback if your model returns fewer than 4 option class classifications
            while len(scores) < 5:
                scores.append(0.0)
            all_probabilities.append(scores[:5])
        except Exception:
            # Fallback uniform distribution array to prevent total application breaking on single api faults
            all_probabilities.append([0.20, 0.20, 0.20, 0.20, 0.20])

    # Element-wise array averaging via base python zip operations
    num_models = len(all_probabilities)
    final_scores = [sum(x) / num_models for x in zip(*all_probabilities)]
    
    # Track the index containing the highest averaged probability score
    best_idx = final_scores.index(max(final_scores))
    letters = ["A", "B", "C", "D", "E"]

    return {
        "predicted_letter": letters[best_idx],
        "predicted_text": options[best_idx],
        "confidence_breakdown": {letters[i]: f"{final_scores[i]*100:.2f}%" for i in range(5)}
    }
