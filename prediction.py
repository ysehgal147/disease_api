import numpy as np
import pandas as pd
import pickle
from fastapi import FastAPI
from model import SymptomModel
import uvicorn

app = FastAPI()

df=pd.read_csv("Testing.csv")
X = df.iloc[:,:-1]
filename = "model_pickle_1"
load_model = pickle.load(open(filename, "rb+"))
symptoms_dict = {}

@app.get('/')
def index():
    return{'message': 'This is Disease Prediction API'}

@app.post('/predict')
def predict(data:SymptomModel):
    data = data.dict()
    arr=[]
    input_vector={}

    for index, symptom in enumerate(X):
        symptoms_dict[symptom] = index
        no_symptons = int(len(data['symptoms']))                      
        input_vector = np.zeros(len(symptoms_dict))  

    for i in range(no_symptons):             
        x = data['symptoms'][i]                       
        arr.append(symptoms_dict[x])        

    for i in arr:               
        input_vector[i] = 1    

    pred_result = load_model.predict([input_vector])
    return {'prediction': pred_result[0]}

if __name__ == '__main__':
    uvicorn.run(app,host='127.0.0.1',port=8000)
