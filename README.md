# Car Sales Forecast

A small web app that predicts next month's car sales from the previous 6 months of sales.
It uses an XGBoost regression model, a FastAPI server and a plain HTML form.

## Folder layout

```
app.py                   FastAPI server (serves the page and the /predict endpoint)
train.py                 trains the model and saves model/model.json
requirements.txt         packages needed to run the app
model/model.json         the trained model
frontend/index.html      the form
data/monthly-car-sales.csv   the dataset (Jan 1960 to Dec 1968)
```

## Run it on your computer

```
pip install -r requirements.txt
uvicorn app:app --reload
```

Then open http://127.0.0.1:8000 in your browser.

To train the model again, run `python train.py`. It prints how the model does on the last 12 months and saves a new `model/model.json`.

## Deploy on Render

1. Push this folder to a GitHub repository (make sure `model/model.json` is included).
2. On render.com, create an account and choose New, then Web Service, and connect the repository.
3. Build Command: `pip install -r requirements.txt`
4. Start Command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
5. Choose the free instance type and deploy.
6. Open the Render URL. The free tier can take a minute to wake up.

## How the API works

`POST /predict` with a body like `{"sales": [18024, 16722, 14385, 21342, 17180, 14577]}`.
The six numbers are the last six months, oldest first. It returns `{"prediction": 12508.73}`.
