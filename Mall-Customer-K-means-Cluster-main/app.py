# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 19:18:14 2022

@author: 91931
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# Set the styles to Seaborn
sns.set()
# Import the KMeans module so we can perform k-means clustering with sklearn
from sklearn.cluster import KMeans
import pickle

app = Flask(__name__)
# Load the pickled model
km_model = pickle.load(open('mallcustomers.pkl','rb')) 


# Importing the dataset
data = pd.read_csv('Mall_Customers (1).csv')
data= data.filter(["Annual Income (k$)","Spending Score (1-100)"], axis=1)

KMeans(n_clusters=5 , random_state = 36)
km_model.fit(data)
km_model.cluster_centers_

cluster_map= pd.DataFrame()
cluster_map['data_index']=data.index.values
cluster_map['cluster']=km_model.labels_
cluster_map= cluster_map[cluster_map.cluster==1]

@app.route('/')
def home():
  
    return render_template("index.html")
  
@app.route('/predict',methods=['GET'])
def predict():
    
    '''
    For rendering results on HTML GUI
    '''

    income = float(request.args.get('income'))
    spend  = float(request.args.get('spend'))
    
    prediction = km_model.predict([[income, spend]])

    if prediction==[0]:
      result="Customer is Sensible as custmor have High Annual Income with Low Spending Score"
    elif prediction==[1]:
      result="Customer is Careful as custmor have Low Annual Income with Low Spending Score"
    elif prediction==[2]:
      result="Custmor is Standard as customer have Average Annual Income with Average Spending Score"
    elif prediction==[3]:
      result="Custmor is Target as customer have High Annual Income with High Spending Score"
    else:
      result="Customer is Careless as custmor have Low Annual Income with High Spending Score"


    return render_template('index.html', prediction_text='Mall Customer Analysis Model  has predicted Whether Customer is  Target or Not : {}'.format(result))


if __name__ == "__main__":
    app.run(debug=True)
