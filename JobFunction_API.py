
from flask import Flask, jsonify, request #import objects from the Flask model4
import sklearn
from sklearn.externals import joblib
import numpy as np
import joblib

app = Flask(__name__) #define app using Flask





loaded_model = joblib.load('jobFunction_prediction_model.sav')

Sales_Keywords = np.load('Sales_keywords.npy')
Marketing_Keywords = np.load('Marekting_keywords.npy')
Software_Keywords =  np.load('Software_keywords.npy')



@app.route('/job/<string:job>', methods = ['GET'])
def JobFunction(job):
	Software_exist = False
	Sales_exist = False
	Marketing_exist = False

	words = job.split()
	for word in words:
		word = word.lower()
		
		if word in Sales_Keywords:
			Sales_exist = True
			break
		if word in Marketing_Keywords:
			Marketing_exist =True
			break
		if word in Software_Keywords:
			Software_exist =True
			break 
	if Software_exist:    
		job= "Software Developer"
	if Sales_exist:
		job= "Sales Consultant"
	if Marketing_exist:
		job= "Marketing Specialist"


	y_pred = loaded_model.predict([job])

	return y_pred[0]


if __name__ == '__main__':
	app.run(debug=True) #run app on port 8080 in debug mode