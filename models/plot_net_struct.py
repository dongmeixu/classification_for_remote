from keras import Input, Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout


def get_net(input_shape):
    input = Input(shape=input_shape)
    x = Conv2D(filters=32, kernel_size=(3, 3), padding="valid", activation="relu")(input)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(filters=64, kernel_size=(3, 3), padding="valid", activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(filters=128, kernel_size=(3, 3), padding="valid", activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(10, activation="sigmoid")(x)

    model = Model(input, x)

    return model

model = get_net((256, 256, 3))
from keras.utils.vis_utils import plot_model
plot_model(model, "simple_cnn.png")