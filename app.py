from fastapi import FastAPI
import uvicorn
from score import read_prep_data, load_apply_model

app=FastAPI()

# @app.get("/{name}")
# def hello(name):
#     return {f"Hello {name} and welcome to this API"}


@app.post("/predict")
def predict():
    features = read_prep_data (filename)
    output = load_apply_model(features, model_file)
    print(output)

if __name__=="__main__":
    uvicorn.run(app)