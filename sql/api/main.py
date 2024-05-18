from typing import List
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from bornrule.sql import BornClassifierSQL
from .connect import engine


app = FastAPI(
    title="Zoo Animal Classification SQL",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

model = BornClassifierSQL(id="zoo", engine=engine)


@app.get("/")
def read_root():
    return f"Why don't you try some predictions? For instance: " \
           f"/predict/?fins=1 or " \
           f"/predict/?legs=2 or " \
           f"/predict/?legs=4&eggs=0"


@app.post("/predict/")
def predict(animals: List[dict]) -> List[str]:
    y_pred = model.predict(animals)
    return list(y_pred)


@app.get("/predict/")
def predict_single(
    hair: int = -1,
    feathers: int = -1,
    eggs: int = -1,
    milk: int = -1,
    airborne: int = -1,
    aquatic: int = -1,
    predator: int = -1,
    toothed: int = -1,
    backbone: int = -1,
    breathes: int = -1,
    venomous: int = -1,
    fins: int = -1,
    legs: int = -1,
    tail: int = -1,
    domestic: int = -1,
    catsize: int = -1
) -> str:
    animal = {
        f"hair={hair}": 1,
        f"feathers={feathers}": 1,
        f"eggs={eggs}": 1,
        f"milk={milk}": 1,
        f"airborne={airborne}": 1,
        f"aquatic={aquatic}": 1,
        f"predator={predator}": 1,
        f"toothed={toothed}": 1,
        f"backbone={backbone}": 1,
        f"breathes={breathes}": 1,
        f"venomous={venomous}": 1,
        f"fins={fins}": 1,
        f"legs={legs}": 1,
        f"tail={tail}": 1,
        f"domestic={domestic}": 1,
        f"catsize={catsize}": 1,
    }
    y_pred = model.predict([animal])
    return y_pred[0]
