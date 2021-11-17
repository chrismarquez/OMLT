from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

import pandas as pd
import numpy as np
import json


tf.keras.backend.set_floatx('float64')

def generate_dataset(n_samples):

    w = 5
    x = np.linspace(-2,2,n_samples)    
    df = pd.DataFrame(x, columns=['x'])
    df['y'] = np.sin(w*x) + x**2

    # Scaling Parameters for the NN
    scale = {'mean': {}, 'std': {}}

    mean_x = df.mean(axis=0)
    std_x = df.std(axis=0)

    # Bounds Parameters for the NN
    bounds = {'min': {}, 'max': {}}

    min_x = df.min(axis=0)
    max_x = df.max(axis=0)

    for col in df.columns:
        scale['mean'][col] = mean_x[col]
        scale['std'][col] = std_x[col]
        bounds['min'][col] = min_x[col]
        bounds['max'][col] = max_x[col]
        df[col + "_scaled"] = (df[col] - scale['mean'][col]) / (scale['std'][col])

    with open("keras_model_sin_wave/scale.json", 'w') as f:
        json.dump(scale, f)
        f.close()

    with open("keras_model_sin_wave/bounds.json", 'w') as f:
        json.dump(bounds, f)
        f.close()

    return df

#train a simple 3 layer neural network
def create_dense_model(nodes_per_layer,activation = 'relu'):
    model = Sequential(name='sin_wave')

    model.add(Input(1))
    model.add(Dense(nodes_per_layer, activation=activation))
    model.add(Dense(nodes_per_layer, activation=activation))
    model.add(Dense(nodes_per_layer, activation=activation))
    model.add(Dense(1))
    model.compile(optimizer=Adam(), loss='mse')

    return model

def train_nn(model, inputs, outputs, savepath, n_epochs):
    print("Training NN")

    #early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(x=inputs, y=outputs,verbose=1, epochs=n_epochs)#, callbacks=[early_stop])
    model.save(savepath)

    return model

def predict_nn(nn, inputs):
    with open("keras_model_sin_wave/scale.json") as f:
        scales = json.load(f)
        f.close()

    # Scale inputs
    scaled_inputs = (inputs - scales['mean']['x'])/ (scales['std']["x"])

    # Get predictions and convert to dataframe
    predictions = nn.predict(x=scaled_inputs)
    df = pd.DataFrame(predictions, columns=["y_predict"])*(scales['std']["y"]) + scales['mean']["y"]

    return df


def load_nn(path):
    if os.path.isdir(path):
        print("Loading NN")
        return tf.keras.models.load_model(path)


if __name__ == "__main__":
    df = generate_dataset(10000)
    df.to_csv("sinwave_dataset.csv")

    NN_r1_path = "keras_model_sin_wave"
    nn = create_dense_model(100)
    nn = train_nn(nn,df['x_scaled'], df['y_scaled'],NN_r1_path,100)

    predictions_df = predict_nn(nn,df["x"])
    nn.save(NN_r1_path)