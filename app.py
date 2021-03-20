from flask import Flask,render_template,url_for,request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
def load_heart():
    with open('./Heart_Attack/model.sav', 'rb') as file:
        heart = pickle.load(file)
    return heart


def scaler_heart():
    
    with open('./Heart_Attack/Scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    
    return scaler

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/tumor')
def tumor():
    return render_template('first.html')

@app.route('/heart_attack',methods=['POST','GET'])
def heart_attack():
    if request.method == 'POST':   
        age= int(request.form['Age'])
        totchol= int(request.form['TotChol'])
        sysbp= int(request.form['SysBP'])
        diabp= int(request.form['DiaBP'])
        heart_rate=int(request.form['Heart_rate'])
        bmi=float(request.form['BMI'])
        glucose=int(request.form['Glucose'])
        male=request.form['gender']
        if male =='Male':
            male = 1
        else:
            male = 0
        test_data =[[age,totchol,sysbp,diabp,bmi,heart_rate,glucose,male]]
        test = pd.DataFrame(test_data,columns=["age", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose", "male"])
        heart = load_heart()
        scaler = scaler_heart()
        scaler_input = scaler.transform(test)
        
        pred = heart.predict(scaler_input)
        print(pred)
        if pred == 0:
            pred = "SAFE"
        else:
            pred = "Patient May have heart Attack"
        # return render_template('second.html',result = pred )
        return str(pred)
    else:
        return render_template('second.html')

@app.route('/diabetes')
def diabetes():
    return render_template('third.html')

@app.route('/diabetes')
def diabetes():
    return render_template('third.html')
if __name__ =="__main__":
    app.run(debug=True)