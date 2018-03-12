from flask import Flask, request, Response
from utilities import array2tensor
from model import model
import torch
from torch.autograd import Variable
import json
import numpy as np

app = Flask(__name__)


@app.route('/api/test', methods=['POST'])
def test():
    net = model.ShapeDetectNetwork()
    net.load_state_dict(torch.load('./model/shapeDetect', map_location=lambda storage, loc: storage))
    r = request.json
    r_json = json.loads(r)
    data = r_json['data']
    numpy_data = np.asarray(data)
    o = net(Variable(array2tensor(numpy_data).unsqueeze(0)))
    classes = ['circle', 'retrangle']
    _, idx = torch.max(o.data, 1)
    shape = classes[idx[0]]
    # response
    response = {
        'message': 'The shape is {}'.format(shape)
    }
    # encode response using jsonpickle
    response_pickled = json.dumps(response)
    return Response(response=response_pickled, status=200, mimetype="application/json")


# start flask app
app.run(host="0.0.0.0", port=5000)
