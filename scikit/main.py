import numpy as np
import falcon
import time
from model import ScikitModel


class ModelResource(object):
    def __init__(self, model):
        self.model = model

    def on_get(self, req, resp):
        try:
            samples = np.array(req.media['samples'])
            start = time.time()
            predictions = self.model.predict(samples)
            end = time.time()
            resp.media = {'classes':predictions.tolist(), 'time': end-start}
        except KeyError:
            resp.media ={}

    def on_post(self, req, resp):
        samples = np.array(req.media['samples'][0])
        labels = np.array(req.media['classes'])
        start = time.time()
        self.model.train(samples, labels)
        end = time.time()
        resp.media = {'time': end - start}


hidden_layer_sizes = (100,)
classes = [0,1,2,3,4,5,6,7,8,9]

app = falcon.API()
app.add_route(
    '/',
    ModelResource(
        ScikitModel(
            hidden_layer_sizes,
            classes
        )
    )
)

