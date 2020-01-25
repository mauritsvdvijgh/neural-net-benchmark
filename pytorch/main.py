from falcon.media.validators import jsonschema
from collections import OrderedDict
import numpy as np
import falcon
import torch
import torch.nn as nn
import torch.optim as optim


# model configuration
hidden_layer_sizes = (30,30,30)
classes = [0,1,2,3,4,5,6,7]
input_dim = 4
activation = 'relu'


def get_activation(activation):
    if 'relu':
        return nn.ReLU()
    else:
        raise NotImplementedError("Activation function '{}' not implemented".format(activation))


model_dict = OrderedDict()
prev_size = input_dim
for i, size in enumerate(hidden_layer_sizes):
    model_dict[str(i*2)] = nn.Linear(prev_size, size)
    model_dict[str(i*2+1)] = get_activation(activation)
    prev_size = size
model_dict[str(i*2+2)] = nn.Linear(prev_size, len(classes))
model_dict[str(i*2+3)] = nn.Softmax(dim=1)

model = nn.Sequential(model_dict)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


class ModelResource(object):
    def on_get(self, req, resp):
        resp.status = falcon.HTTP_200
        if req.media:
            resp.media = {'class':str(torch.max(model(torch.FloatTensor((req.media['samples']))), 2).indices)}
        else:
            resp.media = {'model':str(model)}


    @jsonschema.validate({'sample': {'type':'array'}, 'class':{'type':'int'}})
    def on_post(self, req, resp):
        resp.status = falcon.HTTP_200
        optimizer.zero_grad()
        outputs = model(torch.FloatTensor(req.media['samples'][0]))
        loss = criterion(outputs, torch.LongTensor(req.media['classes']))
        loss.backward()
        optimizer.step()


app = falcon.API()
model_resource = ModelResource()
app.add_route('/', model_resource)

