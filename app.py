from flask import Flask,request, url_for, redirect, render_template,flash
import numpy as np
import joblib
import keras.utils as image
from newsapi import NewsApiClient
from keras.models import load_model
import cv2



app = Flask(__name__,static_url_path='/static')
newsapi = NewsApiClient(api_key='70fdb9ba81ba40b6bda148e672898bd9')

dic = {0 : 'dense', 1 : 'normal'}

model_img =load_model('image_processing.h5')
model=joblib.load('forestfiremodel.pkl')

@app.route('/dash')
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
    print("hai",out)
    if out==1:
        return render_template('index.html',pred='1')
    else:
        return render_template('index.html',pred='0')
		
	

	   	

   
def predict_label(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    new_arr = cv2.resize(img, (60, 60))
    new_arr = np.array(new_arr)
    new_arr = new_arr.reshape(-1, 60, 60, 1)
    return new_arr 
# helper function
def get_sources_and_domains():
	all_sources = newsapi.get_sources()['sources']
	sources = []
	domains = []
	for e in all_sources:
		id = e['id']
		domain = e['url'].replace("http://", "")
		domain = domain.replace("https://", "")
		domain = domain.replace("www.", "")
		slash = domain.find('/')
		if slash != -1:
			domain = domain[:slash]
		sources.append(id)
		domains.append(domain)
	sources = ", ".join(sources)
	domains = ", ".join(domains)
	return sources, domains

    
@app.route('/image_dash',methods=['POST','GET'])
def get_image():
    if request.method == 'POST':
        img = request.files['my_image']
        img_path="static/"+img.filename
        img.save(img_path)
        prediction =model_img.predict(predict_label(img_path))
    return render_template("image_dash.html",prediction = dic[prediction.argmax()], img_path = img_path)

@app.route("/", methods=['GET', 'POST'])
def home():
	if request.method == "POST":
		sources, domains = get_sources_and_domains()
		keyword = request.form["keyword"]
		related_news = newsapi.get_everything(q=keyword,
									sources=sources,
									domains=domains,
									language='en',
									sort_by='relevancy')
		no_of_articles = related_news['totalResults']
		if no_of_articles > 100:
			no_of_articles = 100
		all_articles = newsapi.get_everything(q=keyword,
									sources=sources,
									domains=domains,
									language='en',
									sort_by='relevancy',
									page_size = no_of_articles)['articles']
		return render_template("dash.html", all_articles = all_articles,
							keyword=keyword)
	else:
		top_headlines = newsapi.get_top_headlines(country="in", language="en")
		total_results = top_headlines['totalResults']
		if total_results > 100:
			total_results = 100
		all_headlines = newsapi.get_top_headlines(country="in",
													language="en",
													page_size=total_results)['articles']
		return render_template("dash.html", all_headlines = all_headlines)
	return render_template("dash.html")




if __name__ == '__main__':
    app.run(debug=True)
