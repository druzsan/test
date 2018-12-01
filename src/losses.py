import math
from keras import backend


def arc_loss(s=64., m=.5):
    """
    Wrapper for ArcFace Loss function because of extra parameters
    :param s: scaling of embeddings, or radius of the hypersphere [float]
    :param m: inter-class angle margin on the hypersphere [float]
    :return: ArcFace Loss function
    """
    cos_m = math.cos(m)
    sin_m = math.sin(m)
    mm = sin_m * m
    threshold = math.cos(math.pi - m)

    def loss_function(y_true, y_pred):
        """
        ArcFace Loss function.
        y_pred argument should be logit classification from the last nn-layer, NOT conditional probabilities. So do not
        apply Softmax activation after the last nn-layer. Softmax will be applied at the end of the ArcFace Loss
        function (categorical_crossentropy with argument from_logits=True).
        Furthermore, the length of the embeddings (inputs to the last nn-layer) AND the length of the corresponding
        weight vector should be forced to be 1, and the biases in the the last nn-layer should be set equally to 0.
        :param y_true: true labels in one-hot notation [tensor of shape (batch_size, classes)]
        :param y_pred: logits [tensor of shape (batch_size, classes)]
        :return: ArcFace Loss
        """
        cos_t = y_pred
        cos_t2 = backend.square(cos_t)
        sin_t2 = 1. - cos_t2
        sin_t = backend.sqrt(sin_t2)
        cos_mt = s * (cos_t * cos_m - sin_t * sin_m)

        cond_v = cos_t - threshold
        cond = backend.relu(cond_v)

        keep_val = s * (cos_t - mm)
        cos_mt_temp = backend.switch(cond, cos_mt, keep_val)

        mask = y_true
        inv_mask = 1. - mask

        s_cos_t = s * cos_t

        prediction_with_margin = s_cos_t * inv_mask + cos_mt_temp * mask

        loss = backend.categorical_crossentropy(y_true, prediction_with_margin, from_logits=True)
        return loss

    return loss_function


def cos_loss(s=64., m=.4):

    def loss_function(y_true, y_pred):
        # Implementation according to https://github.com/auroua/InsightFace_TF
        cos_t = y_pred
        cos_t_m = cos_t - m

        mask = y_true
        inv_mask = 1. - mask

        prediction_with_margin = s * cos_t * inv_mask + s * cos_t_m * mask

        loss = backend.categorical_crossentropy(y_true, prediction_with_margin, from_logits=True)
        return loss

    return loss_function


def cos_loss_v2(s=64., m=.4):
    sm = s * m

    def loss_function(y_true, y_pred):
        # My own implementation
        loss = backend.categorical_crossentropy(y_true, s * y_pred - sm * y_true, from_logits=True)
        return loss

    return loss_function
