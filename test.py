import requests

data = {'data': "36,27,231"}
r = requests.post("http://127.0.0.1:5000/predict", json=data)
print('response from server:', r)
