import tensorflow as tf

from fastapi import FastAPI
from pydantic import BaseModel

MODEL = tf.keras.models.load_model('model/')

app = FastAPI()


class UserInput(BaseModel):
    user_input: float


@app.get('/')
async def index():
    return {"Message": "This is Index"}


@app.post('/predict/')
async def predict(user_input: UserInput):
    prediction = MODEL.predict([UserInput.user_input])
    return {"prediction": float(prediction)}
