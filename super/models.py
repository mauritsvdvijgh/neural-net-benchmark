class Scikit():
    def __init__(self, hidden_layer_sizes, classes, input_dim, activation='relu', solver='adam', batch_size='auto'):
        from sklearn.neural_network import MLPClassifier
        self.classes = classes
        self.model = MLPClassifier(
            batch_size=batch_size,
            hidden_layer_sizes=hidden_layer_sizes
        )

    def predict(self, samples):
        predictions = self.model.predict(samples)
        return predictions

    def train(self, samples, labels):
        self.model.partial_fit(samples, labels, classes=self.classes)


class Keras():
    def __init__(self, hidden_layer_sizes, classes, input_dim, activation='relu', solver='adam', batch_size=0):
        from keras.models import Sequential
        from keras.layers import Dense
        self.model = Sequential()
        prev_size = input_dim
        for size in hidden_layer_sizes:
            self.model.add(Dense(size, input_dim=prev_size, activation=activation))
            prev_size = size
        self.model.add(Dense(len(classes), activation='softmax'))
        self.model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=solver,
            metrics=[
                'accuracy'
            ]
        )
        self.epochs = 0


    def predict(self, samples):
        predictions = self.model.predict_classes(samples)
        return predictions

    def train(self, samples, labels):
        self.model.fit(samples, labels, batch_size=len(samples), epochs = self.epochs+1, initial_epoch=self.epochs, validation_split=0.1, verbose=0)
        self.epochs += 1


class PyTorch(object):
    def __init__(self, hidden_layer_sizes, classes, input_dim, activation='relu', solver='adam', batch_size=0):
        from collections import OrderedDict
        import torch
        import torch.nn as nn
        import torch.optim as optim
        model_dict = OrderedDict()
        prev_size = input_dim
        for i, size in enumerate(hidden_layer_sizes):
            model_dict[str(i*2)] = nn.Linear(prev_size, size)
            model_dict[str(i*2+1)] = nn.ReLU()
            prev_size = size
        model_dict[str(i*2+2)] = nn.Linear(prev_size, len(classes))
        model_dict[str(i*2+3)] = nn.LogSoftmax(dim=1)

        self.model = nn.Sequential(model_dict)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters())

    def get_activation(self, activation):
        import torch.nn as nn
        if 'relu':
            return nn.ReLU()
        else:
            raise NotImplementedError("Activation function '{}' not implemented".format(activation))

    def predict(self, samples):
        import torch
        predictions = torch.max(self.model(torch.FloatTensor(samples)), 1).indices
        return predictions

    def train(self, samples, labels):
        import torch
        self.optimizer.zero_grad()
        outputs = self.model(torch.FloatTensor(samples))
        loss = self.criterion(outputs, torch.LongTensor(labels))
        loss.backward()
        self.optimizer.step()

