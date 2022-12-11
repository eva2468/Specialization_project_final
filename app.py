from flask import Flask,request, url_for, redirect, render_template,flash
import numpy as np
import joblib
import keras.utils as image
from keras.models import load_model
import cv2


app = Flask(__name__,static_url_path='/static')

dic = {0 : 'dense', 1 : 'normal'}

model_img =load_model('image_processing.h5')
model=joblib.load('forestfiremodel.pkl')

@app.route('/')
def dash():
    return render_template("dash.html")
@app.route('/predict')
def dash_1  ():
    return render_template("index.html")
@app.route('/image_dash')
def img ():
    return render_template("image_dash.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[float(x) for x in request.form.values()]
    final=[np.array(int_features)]
    print(int_features)
    print(final)
    out=model.predict(final)
    if out==1:
        return render_template('index.html Your Forest is in Danger')
    else:
        return render_template('index.html Your Forest is safe')
   
def predict_label(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    new_arr = cv2.resize(img, (60, 60))
    new_arr = np.array(new_arr)
    new_arr = new_arr.reshape(-1, 60, 60, 1)
    return new_arr
   
    
    
@app.route('/image_dash',methods=['POST','GET'])
def get_image():
    if request.method == 'POST':
        img = request.files['my_image']
        img_path="static/"+img.filename
        img.save(img_path)
        prediction =model_img.predict(predict_label(img_path))
    return render_template("image_dash.html",prediction = dic[prediction.argmax()], img_path = img_path)
 

if __name__ == '__main__':
    app.run(debug=True)
