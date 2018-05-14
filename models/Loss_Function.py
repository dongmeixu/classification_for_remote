import keras.backend as K
import numpy as np


# 定义marco f1 score的相反数作为loss
def score_loss(y_true, y_pred):
    loss = 0
    for i in np.eye(4):
        y_true_ = K.constant([list(i)]) * y_true
        y_pred_ = K.constant([list[i]]) * y_pred
        loss += 0.5 * K.sum(y_true_ * y_pred_) \
                / K.sum(y_true_ + y_pred_ + K.epsilon())
        return -K.log(loss + K.epsilon())


# 定义marco f1 score的计算公式
def score_metric(y_true, y_pred):
    y_true = K.argmax(y_true)
    y_pred = K.argmax(y_pred)
    score = 0.
    for i in range(4):
        y_true_ = K.cast(K.equal(y_true, i), 'float32')
        y_pred_ = K.cast(K.equal(y_pred, i), 'float32')
        score += 0.5 * K.sum(y_true_ * y_pred_) \
                 / K.sum(y_true_ + y_pred_ + K.epsilon())
    return score

