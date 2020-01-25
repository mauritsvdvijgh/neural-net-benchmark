class ScikitModel():
    def __init__(self, hidden_layer_sizes, classes):
        import asdfasdfasdfasdfasdfasdfasdf
        from sklearn.neural_network import MLPClassifier
        self.classes = classes
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes
        )

    def predict(self, samples):
        predictions = self.model.predict(samples)
        return predictions

    def train(self, samples, labels):
        self.model.partial_fit(samples, labels, classes=self.classes)


