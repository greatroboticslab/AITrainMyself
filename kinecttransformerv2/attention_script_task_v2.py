import argparse
import os
import glob
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
from kinect_learning import *

DATA_DIR = 'data'


def train_model(data_file_name, epochs, noise, num_classes=3):
    # 1. Load Data
    file_path = os.path.join(DATA_DIR, data_file_name)
    data_collection = joints_collection(data_file_name.rstrip('.csv'))
    data = load_data_multiple_dimension(file_path, data_collection, noise)
    (train_x, train_y), (val_x, val_y) = create_datasets(data['positions'], data['labels'])

    query_input = keras.Input(shape=train_x[0].shape, dtype='float64')
    value_input = keras.Input(shape=train_x[0].shape, dtype='float64')

    # Convolution layers
    convolution_layer = keras.layers.DepthwiseConv1D(kernel_size=4, padding='same')
    convolved_query = convolution_layer(query_input)
    convolved_value = convolution_layer(value_input)

    # Batch normalization
    convolved_query = keras.layers.BatchNormalization()(convolved_query)
    convolved_value = keras.layers.BatchNormalization()(convolved_value)

    # Second convolution to extract more features
    convolved_query = keras.layers.DepthwiseConv1D(kernel_size=4, padding='same', activation='relu')(convolved_query)
    convolved_value = keras.layers.DepthwiseConv1D(kernel_size=4, padding='same', activation='relu')(convolved_value)

    # Attention layer
    attention_output = keras.layers.Attention()([convolved_query, convolved_value])

    # Flatten and Dense layers
    flatten_layer = keras.layers.Flatten()(attention_output)

    # Dropout layer to avoid overfitting
    dropout_layer_1 = keras.layers.Dropout(0.5)(flatten_layer)

    # Dense layer with L2 regularization and 'relu' activation
    dense_layer_1 = keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(
        dropout_layer_1)

    # Dropout to avoid overfitting
    dropout_layer_2 = keras.layers.Dropout(0.3)(dense_layer_1)

    # Final Dense layer with softmax activation for classification
    dense_layer_2 = keras.layers.Dense(num_classes, activation='softmax')(dropout_layer_2)

    # Build the model
    model = keras.Model(inputs=[query_input, value_input], outputs=[dense_layer_2])

    # Compile the model
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(),
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy']
    )

    # Callbacks
    lr_schedule = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

    # 3. Train Model
    history = model.fit(
        (train_x, train_x), train_y,
        validation_data=((val_x, val_x), val_y),
        epochs=epochs,
        batch_size=64,  # You can tune this
        callbacks=[lr_schedule, early_stopping]
    )

    return model, history


def create_datasets(x, y, test_size=0.4):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)

    # Shuffle
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    x = np.take(x, indices, axis=0)
    y = np.take(y, indices)

    # Train/Validation split
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=test_size)
    return (x_train, y_train), (x_valid, y_valid)


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('--noise', default=False, action=argparse.BooleanOptionalAction)
    FILE_NAMES = list(map(os.path.basename, glob.glob('./data/*.csv')))
    print('Training on files: {}'.format(FILE_NAMES))
    ARGS = PARSER.parse_args()
    print(f'Noise: {ARGS.noise}')

    TRAINING_ATTEMPTS = 10
    EPOCHS = 100
    RESULT = {
        file_name: [train_model(file_name, epochs=EPOCHS, noise=ARGS.noise) for _ in range(TRAINING_ATTEMPTS)]
        for file_name in FILE_NAMES
    }
    BEST_RESULTS = {
        file_name: max(trained_models, key=lambda model: model[1].history['val_accuracy'])
        for file_name, trained_models in RESULT.items()
    }

    for DATA_FILE, BEST_RESULT in BEST_RESULTS.items():
        print(DATA_FILE, BEST_RESULT[1].history['val_accuracy'][-1])
