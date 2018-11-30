from keras import backend

def categorical_accuracy(from_logits=True):

    def metric(y_true, y_pred):
        if from_logits is True:
            y_pred = backend.softmax(y_pred, axis=-1)
        return backend.cast(backend.equal(backend.argmax(y_true, axis=-1),
                                          backend.argmax(y_pred, axis=-1)),
                            backend.floatx())