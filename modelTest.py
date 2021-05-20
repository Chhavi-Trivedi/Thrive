import pandas as pd
import numpy as np
import datetime
import pickle



model = pickle.load(open("RandomforestModelFinal1.pkl", "rb"))
print("Model Loaded")

Location = float(2)
MinTemp = float(13.4)
MaxTemp = float(13.4)
Rainfall = float(0.6)
Evaporation = float(5.468232)
Sunshine = float(7.611178)
WindGustDir = float(13)
WindGustSpeed = float(44.0)
WindDir9am = float(13)
WindDir3pm = float(14)
WindSpeed9am = float(20.0)
WindSpeed3pm = float(24.0)
Humidity3pm = float(22.0)
Pressure9am = float(1007.7)
Pressure3pm = float(1007.1)
Cloud9am= float(8.000000)
Cloud3pm = float(4.50993)
RainToday = float(0.0)


input_lst = np.array([Location, MinTemp,	MaxTemp,	Rainfall,	Evaporation,	Sunshine,	WindGustDir,	WindGustSpeed,	
WindDir9am,	WindDir3pm,	WindSpeed9am,	WindSpeed3pm,	Humidity3pm,	Pressure9am,	Pressure3pm,	
Cloud9am,	Cloud3pm,	RainToday])

input_lst = input_lst.reshape(-1, 1) 

pred = model.predict(input_lst)

if(pred.mean()>0.5):
	print("Will rain")

else:
	print("will not rain")




