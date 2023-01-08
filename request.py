import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'toss_win':0, 'venue':'Dubai International Cricket Stadium', 'team1':'Deccan Chargers','team2':'Deccan Chargers','toss_decision':'bat','umpire1':'A Deshmukh','umpire2':'A Deshmukh'})

print(r.json())