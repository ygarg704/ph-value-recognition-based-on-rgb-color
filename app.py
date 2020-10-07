from flask import Flask, render_template, request
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def button():
    if request.method == 'POST':
        comment = request.json['data']
        comment = comment.split(",")
        comment = [int(i) for i in comment]
        print(comment)

        ds = pd.read_csv('dataset/ph-data.csv')

        X = ds.iloc[:, :-1]
        X.loc[len(X)] = comment
        X = X.values
        print(X[-1])

        sc = StandardScaler()
        X = sc.fit_transform(X)

        model = joblib.load('notebook/model.pkl')
        pred = model.predict(X)[-1].tolist()
        print(X[-1])


        preds = {
            "Prediction": pred
        }

        print(preds)
        return preds


if __name__ == '__main__':
    app.run()


