import tensorflow as tf
from Tools import return_file_paths
from tensorflow.keras import layers, models
from DLStep1_DataGenerators import return_generator
import os


def return_model() -> models.Model:
    inputs = x = layers.Input(shape=[None, None, None, 1])
    conv = layers.Conv3D
    conv_kernel = (3, 3, 3)
    max_pool = layers.MaxPool3D
    pool_kernel = (2, 2, 2)
    up_sampling = layers.UpSampling3D

    encoding_list = []
    filters_list = []
    initial_conv = 16
    max_filters = 32
    number_of_layers = 2
    conv_per_layer = 2
    for _ in range(number_of_layers):
        filters_list.append(initial_conv)
        for __ in range(conv_per_layer):
            x = conv(initial_conv, conv_kernel, padding="same", name=f"Encoding_Conv_{_}_{__}")(x)
            x = layers.BatchNormalization(name=f"Encoding_BN_{_}_{__}")(x)
            x = layers.Activation("elu", name=f"Encoding_Activation_{_}_{__}")(x)
            if __ == conv_per_layer - 1:
                encoding_list.append(x)
        initial_conv = int(initial_conv * 2)
        if initial_conv > max_filters:
            initial_conv = max_filters
        x = max_pool(pool_kernel)(x)
    """
    Make a bottle neck
    """
    for __ in range(conv_per_layer):
        x = conv(initial_conv, conv_kernel, padding="same", name=f"Bottom_Conv_{_}_{__}")(x)
        x = layers.BatchNormalization(name=f"Bottom_BN_{_}_{__}")(x)
        x = layers.Activation("elu", name=f"Bottom_Activation_{_}_{__}")(x)
    for _ in range(conv_per_layer):
        x = up_sampling(pool_kernel)(x)
        prev_side = encoding_list.pop()
        x = layers.Concatenate()([x, prev_side])
        for __ in range(conv_per_layer):
            x = conv(initial_conv, conv_kernel, padding="same", name=f"Decoding_Conv_{_}_{__}")(x)
            x = layers.BatchNormalization(name=f"Decoding_BN_{_}_{__}")(x)
            x = layers.Activation("elu", name=f"Decoding_Activation_{_}_{__}")(x)

    x = conv(2, conv_kernel, padding="same", activation='softmax', name="EndConv")(x)
    model = tf.keras.Model(inputs, x)
    return model


def train():
    _, tf_records_path, tensorboard_path = return_file_paths()
    session_num = 1
    model = return_model()

    records_path = [tf_records_path]
    train_generator = return_generator(records_path, batch=5, out_shape=(32, 128, 128))

    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_path, profile_batch=0,
                                                 write_graph=True)
    checkpoint_path = os.path.join(tensorboard_path, "checkpoint", f"Session_{session_num}/cp.weights.h5")

    # Create a callback that saves the model's weights
    monitor_save = "loss"  # "val_loss"

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     save_best_only=False,
                                                     monitor=monitor_save,
                                                     verbose=1)

    callbacks = [tensorboard, cp_callback]  # reduce_on_plateau

    learning_rate = 1e-3
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    loss = tf.keras.losses.CategoricalCrossentropy()

    metrics = [tf.keras.metrics.CategoricalCrossentropy()]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    model.fit(train_generator.data_set, epochs=100, callbacks=callbacks, steps_per_epoch=len(train_generator))


if __name__ == '__main__':
    train()
