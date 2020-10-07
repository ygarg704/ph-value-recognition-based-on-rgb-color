from flask import Flask, render_template, request
import pandas as pd
import re
import os
import joblib
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/second', methods=['POST'])
def second():
    if request.method == 'POST':
        comment1 = request.form['blue']
        comment2 = request.form['green']
        comment3 = request.form['red']

        comment = [comment1, comment2, comment3]

        ds = pd.read_csv('dataset/ph-data.csv')

        X = ds.iloc[:, :-1]
        X.loc[len(X)] = comment
        X = X.values

        sc = StandardScaler()
        X = sc.fit_transform(X)

        model = joblib.load('notebook/model.pkl')
        pred = model.predict(X)[-1]

        print(pred)
        return render_template("result.html", prediction=pred, comments1=comment1, comments2=comment2, comments3=comment3)


if __name__ == '__main__':
    app.run()
