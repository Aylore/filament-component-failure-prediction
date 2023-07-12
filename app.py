from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from utils.data_models import SamplePostRequest

app = FastAPI()

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("button.html", {"request": request})


@app.post("/button")
def handle_button():
    # Perform any desired action here
    return {"message": "Button clicked!"}


# @app.post("/predict")
# def predict(request : SamplePostRequest):


#     return [request.a  + request.b[0], request.c]