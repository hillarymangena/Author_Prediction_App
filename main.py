from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = FastAPI()
templates = Jinja2Templates(directory="templates")

simple_lstm_model = tf.keras.models.load_model('simple_lstm_model.h5')
bidirectional_lstm_model = tf.keras.models.load_model('bidirectional_lstm_model.h5')

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('label_encoder.pickle', 'rb') as handle:
    le = pickle.load(handle)

def preprocess_text(text):
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=200, padding='post', truncating='post')
    return padded

# Define the author mapping based on your list (0-based index)
author_mapping = {
    0: "Charles Dickens",
    1: "William Makepeace Thackeray",
    2: "Anthony Trollope",
    3: "Wilkie Collins",
    4: "Elizabeth Gaskell",
    5: "George Eliot",
    6: "Charlotte Brontë",
    7: "Emily Brontë",
    8: "Anne Brontë",
    9: "Charles Kingsley",
    10: "Thomas Hardy",
    11: "Mary Elizabeth Braddon",
    12: "Margaret Oliphant",
    13: "George Meredith",
    14: "R. D. Blackmore",
    15: "Lewis Carroll",
    16: "Charles Reade",
    17: "George Gissing",
    18: "Samuel Butler",
    19: "Edward Bulwer-Lytton",
    20: "Alfred Tennyson",
    21: "Robert Browning",
    22: "Elizabeth Barrett Browning",
    23: "Christina Rossetti",
    24: "Dante Gabriel Rossetti",
    25: "Gerard Manley Hopkins",
    26: "Coventry Patmore",
    27: "Arthur Hugh Clough",
    28: "Matthew Arnold",
    29: "A. C. Swinburne",
    30: "Thomas Carlyle",
    31: "John Ruskin",
    32: "Walter Pater",
    33: "Thomas Babington Macaulay",
    34: "Herbert Spencer",
    35: "John Stuart Mill",
    36: "James Anthony Froude",
    37: "Leslie Stephen",
    38: "William Morris",
    39: "Oscar Wilde",
    40: "Bram Stoker",
    41: "Sheridan Le Fanu",
    42: "H. Rider Haggard",
    43: "Andrew Lang",
    44: "Juliana Horatia Ewing",
    45: "Dinah Craik",
    46: "Augusta Webster",
    47: "George MacDonald",
    48: "Samuel Smiles",
    49: "George Meredith"
}

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(text_snippet: str = Form(...)):
    processed_text = preprocess_text(text_snippet)
    simple_pred = simple_lstm_model.predict(processed_text)
    bidirectional_pred = bidirectional_lstm_model.predict(processed_text)
    simple_pred_class = int(np.argmax(simple_pred, axis=1)[0])
    bidirectional_pred_class = int(np.argmax(bidirectional_pred, axis=1)[0])
    simple_confidence = float(np.max(simple_pred) * 100)
    bidirectional_confidence = float(np.max(bidirectional_pred) * 100)
    simple_author = author_mapping.get(simple_pred_class, "Unknown")
    bidirectional_author = author_mapping.get(bidirectional_pred_class, "Unknown")
    return {
        "simple_lstm": f"{simple_author} ({simple_confidence:.2f}%)",
        "bidirectional_lstm": f"{bidirectional_author} ({bidirectional_confidence:.2f}%)"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)