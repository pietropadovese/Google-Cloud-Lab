import pickle
import pandas as pd
from typing import List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import HTMLResponse



app = FastAPI(
    title="Zoo Animal CLassification",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

model = pickle.load(
    open('model.pkl', 'rb')
)


@app.get("/", response_class = HTMLResponse)
def home():
    return """
    <html>
    <head>
        <title>Animal Predictor</title>
    </head>
    <body>
        <h1>Animal Predictor</h1>
        <form action="/predict/" method="post">
            <label for="name1">Animal Name 1:</label><br>
            <input type="text" id="name1" name="animals[0].name"><br>
            <label for="age1">Animal Age 1:</label><br>
            <input type="number" id="age1" name="animals[0].age"><br><br>
            <label for="name2">Animal Name 2:</label><br>
            <input type="text" id="name2" name="animals[1].name"><br>
            <label for="age2">Animal Age 2:</label><br>
            <input type="number" id="age2" name="animals[1].age"><br><br>
            <input type="submit" value="Predict">
        </form>
    </body>
    </html>
    """


class Animal(BaseModel):
    hair: int
    feathers: int
    eggs: int
    milk: int
    airborne: int
    aquatic: int
    predator: int
    toothed: int
    backbone: int
    breathes: int
    venomous: int
    fins: int
    legs: int
    tail: int
    domestic: int
    catsize: int


@app.post("/predict/")
def predict(animals: List[Animal]) -> List[str]:
    X = pd.DataFrame([dict(animal) for animal in animals])
    y_pred = model.predict(X)
    return list(y_pred)
