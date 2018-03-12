import requests
from shape_maker import Shape
import json

# Set server address
addr = 'http://localhost:5000'
test_url = addr + '/api/test'
# Set post header
content_type = 'application/json'
headers = {'content-type': content_type}
# Create 2 shapes
creator = Shape()
circle = creator.make('circle')
retrangle = creator.make('retrangle')
# Transform numpy array to list
circle = circle.tolist()
retrangle = retrangle.tolist()
# wrap them into json
json_f1 = json.dumps({'data': retrangle})
json_f2 = json.dumps({'data': circle})
# post request
response1 = requests.post(test_url, json=json_f1, headers=headers)
print(response1.text)
response2 = requests.post(test_url, json=json_f2, headers=headers)
print(response2.text)
