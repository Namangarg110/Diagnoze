from flask import Flask,render_template,url_for,request
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from keras.models import load_model
import os
from werkzeug.utils import secure_filename
from keras.models import load_model
import cv2
from tqdm import tqdm
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
# from flask_uploads import UploadSet, configure_uploads,IMAGES
app = Flask(__name__)
# @app.route('/upload')
# def upload_file():
#    return render_template('upload.html')
	

		

#heart attack
def load_heart():
    with open('./Heart_Attack/model.sav', 'rb') as file:
        heart = pickle.load(file)
    return heart


def scaler_heart():
    
    with open('./Heart_Attack/Scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    return scaler

@app.route('/heart_attack.html',methods=['POST','GET'])
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
        return render_template('result.html',report= str(pred))
    else:
        return render_template('heart_attack.html')

#diabetes
def load_diab():
    model = load_model('./Diabetes/Saved_model/NN.h5')
    model.load_weights('./Diabetes/Saved_model/NN_Weight.h5')
    return model


def scaler_diab():
    
    with open('./Diabetes/Saved_model/scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    return scaler

@app.route('/diabetes.html',methods=['POST','GET'])
def diabetes():
    if request.method =='POST':
        age= int(request.form['Age'])
        preg= int(request.form['preg'])
        gluc= int(request.form['gluc'])
        bp= int(request.form['bp'])
        st= int(request.form['st'])
        ins=int(request.form['ins'])
        bmi=float(request.form['bmi'])
        DiaPedi=int(request.form['DiaPedi'])
        test_data =[[preg,gluc,bp,st,ins,bmi,DiaPedi,age]]
        test = pd.DataFrame(test_data,columns=["preg", "gluc", "bp", "st", "ins", "bmi", "DiaPedi", "age"])
        diab= load_diab()
        scaler = scaler_diab()
        scaler_input = scaler.transform(test)
        res= np.round(diab.predict(scaler_input))
        if res == 1:
            res ="Patient is Diabetic"
        else:
            res ="Patient is not Diabetic"
        return render_template('result.html',report= res)
    else:
        return render_template('diabetes.html')


        

#tumor

def load_tumor():
    model = load_model('./Tumor/Saved_Model/tumor_CNN.h5')
    model.load_weights('./Tumor/Saved_Model/CNN_weights.h5')
    return model



def uploader():
        path = 'static/uploads/'
        uploads = sorted(os.listdir(path), key=lambda x: os.path.getctime(path+x))        # Sorting as per image upload date and time
        print(uploads)
        #uploads = os.listdir('static/uploads')
        uploads = ['uploads/' + file for file in uploads]
        uploads.reverse()
        return render_template("index.html",uploads=uploads)            # Pass filenames to front end for display in 'uploads' variable

app.config['UPLOAD_PATH'] = 'static/uploads'             # Storage path    
@app.route("/tumor.html",methods=['GET','POST'])
def upload_file():                                       # This method is used to upload files 
    if request.method == 'POST':
        f = request.files['fileUpload']
        print(f.filename)
        #f.save(secure_filename(f.filename))
        model = load_tumor()
        filename = secure_filename(f.filename)
        path=os.path.join(app.config['UPLOAD_PATH'], filename)
        f.save(os.path.join(app.config['UPLOAD_PATH'], filename))
        predictions = ['No Tumor', 'Pituitary Tumor', 'Meningioma Tumor', 'Glioma Tumor']
        img = cv2.imread(path)
        img = cv2.resize(img,(70,70))
        nor =255
        img = img/nor
        res = predictions[np.argmax(model.predict(img.reshape(1,70,70,3)))]
        return render_template('result.html',report=res)
                  
    
    else:
        return render_template('tumor1.html')



# photos=UploadSet('photos',IMAGES)
# app.config['UPLOAD_PHOTOS_DEST'] = 'static/img'
# configure_uploads(app,photos)

# @app.route('/upload',method=['GET',['POST']])
# def upload():
#     if request.method == 'POST' and 'photo' in request.files:
#         filename = photo.save(request.file['photo'])
#         return filename
#     return render_template('upload.html')



# @app.route('/uploader', methods = ['GET', 'POST'])
# def upload_file():
#     import cv2
#     if request.method == 'POST':
#         f = request.files['file']
#         # f.save(secure_filename(f.filename))
#         model= load_tumor()
#         predictions = ['no_tumor', 'pituitary_tumor', 'meningioma_tumor', 'glioma_tumor']
#         img = asarray(f)
#         # img = cv2.imread(img)
#         img = cv2.resize(img,(70,70))
#         nor = 255
#         img = img/nor
#         print('Predictions',predictions[np.argmax(model.predict(img.reshape(1,70,70,3)))])
#         import matplotlib.pyplot as plt
#         plt.imshow(img)
#         plt.show()
#         return 'file uploaded successfully'
#     else:
#         return render_template('upload.html')

# @app.route('/uploader')  
# def upload():  
#     return render_template("upload.html")  
 
# @app.route('/success', methods = ['POST'])  
# def success():  
#     if request.method == 'POST':  
#         f = request.files['file']  
#         f.save(f.filename)  
#         return render_template("success.html", name = f.filename)  
  

#corona

def prepare_img(file):
    img_path = ''
    img = image.load_img(img_path+file, target_size=(224,224))
    img_array = image.img_to_array(img)
    img_array_extended_dims = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet.preprocess_input(img_array_extended_dims)

def load_corona():
    json_file = open('./Covid-19/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("./Covid-19/model.h5")
    return loaded_model
@app.route('/corona.html',methods=["POST","GET"])
def corona():
    if request.method == 'POST':
        f = request.files['fileUpload']
        model = load_corona()
        filename = secure_filename(f.filename)
        f.save(os.path.join(app.config['UPLOAD_PATH'], filename))
        path=os.path.join(app.config['UPLOAD_PATH'], filename)
        data=prepare_img(path)
        pred=model.predict(data)
        pred = pred.argmax(axis=1)
        if pred == 0:
            pred = "Covid-19 Positive"
        else:
            pred = "Covid-19 Negative"
        return render_template('result.html',report=pred)
    else:
        return render_template('corona1.html')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/index.html')
def index_html():
    return render_template('index.html')

@app.route('/about.html')
def aboutus():
    return render_template('about.html')

if __name__ =="__main__":
    app.run(debug=True)