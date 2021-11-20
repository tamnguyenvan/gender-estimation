"""Train age and gender estimation model."""
import os
import argparse
import sys
from sys import argv

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import plot_model

import utils
import augmentor
import models


def parse_args(argv):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Path to data directory.')
    parser.add_argument('--dataset', type=str, default='imdb,wiki,utk',
                        help='Datasets would be used to train.')
    parser.add_argument('--image_size', type=int, default=224,
                        help='The input image size.')
    parser.add_argument('--model', type=str, default='resnet18_v2',
                        choices=['resnet18_v2', 'mobilenetv3'],
                        help='The model used for training.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='The training batch size.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='The number of epochs would be trained.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='The initial learning rate.')
    parser.add_argument('--save_dir', type=str, default='saved_models',
                        help='The directory for saving checkpoints.')
    parser.add_argument('--save_history', action='store_true', default=True,
                        help='The flag indicates visualization.')
    return parser.parse_args(argv)


@tf.function
def parse_fn(path, age, gender, image_size, augmentation=True):
    """Parse image function."""
    image = tf.io.decode_jpeg(tf.io.read_file(path, 'rb'), channels=3)
    image = tf.image.resize(image, (image_size, image_size))
    image = tf.image.convert_image_dtype(image, tf.float32)

    # Data augmentation
    if augmentation:
        image = augmentor.random_flip_left_right(image)
        image = augmentor.random_rotate(image, angle=5, radian=False)
        image = augmentor.random_shift(image,
                                       width_shift_range=0.1,
                                       height_shift_range=0.1)
        image = augmentor.zoom_random(image, percentage_area=0.9)
        image = augmentor.random_contrast(image,
                                          min_factor=0.8,
                                          max_factor=1.2)
        image = augmentor.random_brightness(image,
                                            min_factor=0.8,
                                            max_factor=1.2)
        image = augmentor.random_erase(image, rectangle_area=0.15)

    image = image / 255.
    return image, gender


def main(args):
    # Fixed parameters
    num_classes = 1

    data_dir = args.data_dir
    dataset = args.dataset
    model_type = args.model
    image_size = args.image_size
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr

    # Data preparation
    ((train_paths, train_ages, train_genders),
     (val_paths, val_ages, val_genders)) = utils.load_data(data_dir, dataset)

    num_train = len(train_paths)
    num_val = len(val_paths)
    print(f'Model type: {model_type}')
    print(f'Number of training examples: {len(train_paths)}')
    print(f'Number of validation examples: {len(val_paths)}')

    train_data = tf.data.Dataset.from_tensor_slices(
        (train_paths, train_ages, train_genders))
    train_data = train_data.shuffle(1000) \
        .map(lambda x, y, z: parse_fn(x, y, z, image_size),
             num_parallel_calls=tf.data.experimental.AUTOTUNE) \
        .batch(batch_size) \
        .repeat() \
        .prefetch(tf.data.experimental.AUTOTUNE)

    val_data = tf.data.Dataset.from_tensor_slices(
        (val_paths, val_ages, val_genders))
    val_data = val_data.map(
        lambda x, y, z: parse_fn(x, y, z, image_size, False),
        num_parallel_calls=tf.data.experimental.AUTOTUNE) \
        .batch(batch_size) \
        .prefetch(tf.data.experimental.AUTOTUNE)

    # Build the model
    input_shape = (image_size, image_size, 3)
    # base_model_fn = getattr(sys.modules['models'], model_type)
    base_model = tf.keras.applications.MobileNetV3Small(
        input_shape=input_shape,
        include_top=False,
        include_preprocessing=False,
    )
    x = base_model.output
    x = tf.keras.layers.Flatten()(x)
    output = layers.Dense(units=num_classes,
                              activation='sigmoid',
                              name='gender')(x)
    model = Model(base_model.input, output)

    opt = Adam(learning_rate=lr)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics='accuracy')
    model.summary()

    # Prepare model saving directory.
    save_dir = os.path.join(os.getcwd(), args.save_dir)
    model_name = 'gender_%s.{epoch:03d}.{val_loss:.4f}.h5' % model_type
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)

    # Prepare callbacks for saving the model and learning rate schedule
    checkpoint = ModelCheckpoint(filepath=filepath,
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True)

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=3,
                                   verbose=1,
                                   min_lr=0.5e-6)
    early_stopping = EarlyStopping(monitor='val_loss',
                                   mode='auto',
                                   patience=5,
                                   verbose=1,
                                   restore_best_weights=True)
    callbacks = [checkpoint, lr_reducer, early_stopping]

    # Train the model
    hist = model.fit(
        train_data,
        steps_per_epoch=num_train // batch_size,
        validation_data=val_data,
        validation_steps=num_val // batch_size,
        epochs=epochs,
        callbacks=callbacks)

    if args.save_history:
        history_path = os.path.join(save_dir, 'history.npy')
        np.save(history_path, hist.history)


if __name__ == '__main__':
    main(parse_args(argv[1:]))