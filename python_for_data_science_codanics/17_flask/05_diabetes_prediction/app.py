from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

app = Flask(__name__, template_folder='templates')

svm_model = pickle.load(open('./model/svm_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')