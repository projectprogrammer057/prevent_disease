import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('clinic.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == "POST":
        pregnence = request.form['num_preg']
        glucose = request.form['glucose_conc']
        pressure = request.form['diastolic_bp']
        thick = request.form['thickness']
        ins = request.form['insulin']
        metab = request.form['bmi']
        pred = request.form['diab_pred']
        year = request.form['age']
        mus = request.form['skin']
        features_value = [np.array([pregnence,glucose,pressure,thick,ins,metab,pred,year,mus])]
    
        features_name = [ "num_preg","glucose_conc","diastolic_bp","thickness","insulin","bmi","diab_pred","age","skin"]
    
        df= pd.DataFrame(features_value, columns=features_name)
        output = model.predict(df)
        
        if output == 1:
            res_val = "***Diabaties and Please Consult to the Doctor(Physician)***"
        else:
            res_val = "***No Diabaties and Enjoy your life***"
        

    return render_template('index.html', prediction_text='Patient has {}'.format(res_val))

if __name__ == "__main__":
    app.run()
