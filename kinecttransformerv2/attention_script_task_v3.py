import argparse
import os
import glob
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
from kinect_learning import *
import time

from transformers import AutoModelForCausalLM

DATA_DIR = 'data'

# Training function
def train_model(data_file_name, epochs, noise, num_classes=3):
    # 1. Load Data
    file_path = os.path.join(DATA_DIR, data_file_name)
    data_collection = joints_collection(data_file_name.rstrip('.csv'))
    data = load_data_multiple_dimension(file_path, data_collection, noise)
    (train_x, train_y), (val_x, val_y) = create_datasets(data['positions'], data['labels'])

    # Input layers
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

    # Dropout to avoid overfitting
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

    # Train the model
    history = model.fit(
        (train_x, train_x), train_y,
        validation_data=((val_x, val_x), val_y),
        epochs=epochs,
        batch_size=64,
        callbacks=[lr_schedule, early_stopping]
    )

    # Save the trained model to disk for later inference
    model.save(f'models/model_{data_file_name.rstrip(".csv")}.h5')

    return model, history


# Create datasets from data
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


# Inference function with quantization
def run_inference(data_file_name, test_data):
    # Load the trained model
    model_path = f'models/model_{data_file_name.rstrip(".csv")}.h5'
    model = tf.keras.models.load_model(model_path)

    # Quantize the model using EETQ
    #quantized_model = eetq.quantize(model)

    quantization_config = EetqConfig("int8")
    modelq = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", quantization_config=quantization_config)

    # Preprocess test data
    test_data = np.asarray(test_data, dtype=np.float64)

    # Time the inference step
    start_time = time.time()

    # Run inference with the quantized model
    predictions = modelq.predict([test_data, test_data])

    # Measure time taken for inference
    end_time = time.time()
    inference_time = end_time - start_time

    print(f"Inference completed in {inference_time:.4f} seconds")
    print(f"Predicted class probabilities: {predictions}")

    # Convert probabilities to predicted classes
    predicted_classes = np.argmax(predictions, axis=-1)
    print(f"Predicted classes: {predicted_classes}")

    return predicted_classes, inference_time


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('--noise', default=False, action=argparse.BooleanOptionalAction)
    FILE_NAMES = list(map(os.path.basename, glob.glob('./data/*.csv')))
    print('Training on files: {}'.format(FILE_NAMES))
    ARGS = PARSER.parse_args()
    print(f'Noise: {ARGS.noise}')

    TRAINING_ATTEMPTS = 5
    EPOCHS = 10
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

    # Example: running inference on a test sample
    test_data = np.random.rand(1, 14, 3)  # Random test data, adjust shape as needed
    for file_name in FILE_NAMES:
        predicted_classes, inference_time = run_inference(file_name, test_data)
