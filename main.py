from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time
from function_approximator.func_approx import *

##
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class FuncApproxItem(BaseModel):
    expression: str
    xMin: int
    xMax: int
    points: int
    neuronsPerLayer: list[int]
    epochs: int
    activation: str
    seed: int
    learningRate: float


@app.post("/api/funcapproximate/")
async def approximate(data: FuncApproxItem):
    start = time.perf_counter()
    X, Y_pred, avg_loss = train_and_predict(dict(data))
    end = time.perf_counter()
    time_ms = (end - start) * 1000
    return {
        "predicted": {
            'x': X.tolist(),
            'y': Y_pred.tolist(),
            'Avg_Train_loss': round(avg_loss, 7),
            'trainTimeMs': round(time_ms, 3),
        }
    }
