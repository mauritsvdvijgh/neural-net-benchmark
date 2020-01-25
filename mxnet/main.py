from keras.models import Sequential
from keras.layers import Dense
import numpy
from falcon.media.validators import jsonschema
import falcon


# model configuration
hidden_layer_sizes = (30,30,30)
classes = [0,1,2,3,4,5,6,7]
input_dim = 4
activation = 'relu'


#  model config
model = Sequential()
prev_size = input_dim
for size in hidden_layer_sizes:
    model.add(Dense(size, input_dim=prev_size, activation=activation))
    prev_size = size
model.add(Dense(len(classes), activation='sigmoid'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# http endpoint definition
class ModelResource(object):
    @jsonschema.validate({'samples': 'array'})
    def on_get(self, req, resp):
        resp.status = falcon.HTTP_200
        if req.media:
            resp.media = {'class':model.predict_classes(req.media['samples']).tolist()}
        else:
            resp.media = {'model':str(model)}

    @jsonschema.validate({'samples': 'array', 'classes':'array'})
    def on_post(self, req, resp):
        resp.status = falcon.HTTP_200
        model.fit(req.media['samples'], req.media['classes'])


# set up the wsgi webapp
app = falcon.API()
model_resource = ModelResource()
app.add_route('/', model_resource)

