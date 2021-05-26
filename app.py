from flask import Flask,render_template,url_for,request,jsonify
from flask_cors import cross_origin
import pandas as pd
import numpy as np
import datetime
import pickle



app = Flask(__name__, template_folder="template")
model = pickle.load(open("RandomforestModelFinal1.pkl", "rb"))
print("Model Loaded")

@app.route("/",methods=['GET'])
@cross_origin()
def home():
	return render_template("index.html")

@app.route("/dashboard",methods=['GET'])
@cross_origin()
def dashboard():
	return render_template("dashboard.html")

@app.route("/predict",methods=['GET', 'POST'])
@cross_origin()
def predict():
	if request.method == "POST":

		# MinTemp
		MinTemp = float(request.form['mintemp'])
		# MaxTemp
		MaxTemp = float(request.form['maxtemp'])
		# Rainfall
		Rainfall = float(request.form['rainfall'])
		# Evaporation
		Evaporation = float(request.form['evaporation'])
		# Sunshine
		Sunshine = float(request.form['sunshine'])
		# Wind Gust Speed
		WindGustSpeed = float(request.form['windgustspeed'])
		# Wind Speed 9am
		WindSpeed9am = float(request.form['windspeed9am'])
		# Wind Speed 3pm
		WindSpeed3pm = float(request.form['windspeed3pm'])
		# Humidity 3pm
		Humidity3pm = float(request.form['humidity3pm'])
		# Pressure 9am
		Pressure9am = float(request.form['pressure9am'])
		# Pressure 3pm
		Pressure3pm = float(request.form['pressure3pm'])
		# Cloud 9am
		Cloud9am = float(request.form['cloud9am'])
		# Cloud 3pm
		Cloud3pm = float(request.form['cloud3pm'])
		# Cloud 3pm
		Location = float(request.form['location'])
		# Wind Dir 9am
		WindDir9am = float(request.form['winddir9am'])
		# Wind Dir 3pm
		WindDir3pm = float(request.form['winddir3pm'])
		# Wind Gust Dir
		WindGustDir = float(request.form['windgustdir'])
		# Rain Today
		RainToday = float(request.form['raintoday'])


		input_lst = np.array([Location, MinTemp, MaxTemp, Rainfall,	Evaporation, Sunshine, WindGustDir,	WindGustSpeed, WindDir9am, WindDir3pm, WindSpeed9am, WindSpeed3pm, Humidity3pm,	
		Pressure9am, Pressure3pm, Cloud9am,	Cloud3pm, RainToday])

		input_lst = input_lst.reshape(-1, 1) 
		pred = model.predict(input_lst)
		output = pred

		if(pred.mean()>0.5):
			return render_template("rainy.html")

		else:
			return render_template("sunny.html")
			
			
	return render_template("predictor.html")

if __name__=='__main__':
	app.run(debug=True)
