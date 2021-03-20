from flask import Flask,render_template,request
from flask_sqlalchemy import SQLAlchemy
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from collections import Counter

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
db = SQLAlchemy(app)

def load_heart():
    with open('model.sav', 'rb') as file:
        heart = pickle.load(file)
    return heart


def scaler_heart():
    
    with open('Scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    
    return scaler
    

class Todo(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.String(200), nullable=False)

    def __repr__(self):
        return '<Task %r>' % self.id 


@app.route('/',methods=['POST','GET'])
def index():
    render_template
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
        return str(pred)




    else:
        return render_template("index.html")

if __name__=="__main__":    app.run(debug=True)