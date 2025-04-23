import os
import argparse
import yaml
import tensorflow as tf
from models.unet import Unet
from models.fpn import FPN
from models.linknet import LinkNet
from models.pspnet import PSPNet
from data.augmentation import Augmentation
from utils.losses import Losses
from utils.metrics import Metrics
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_image(image_path, config):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, config['image_size'])
    image = tf.cast(image, tf.float32) / 255.0
    return image

def load_mask(mask_path, config):
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=4)
    mask = tf.image.resize(mask, config['image_size'], method='nearest')
    mask = tf.cast(mask, tf.float32) / 255.0
    return mask

def preprocess(image, mask, preprocessing_fn):
    image = preprocessing_fn(image)
    return image, mask

def augmentation(image, mask, augmentation_fn):
    sample = augmentation_fn(image=image, mask=mask)
    return tf.convert_to_tensor(sample['image']), tf.convert_to_tensor(sample['mask'])

def process_path(image_path, mask_path, config, augmentation_fn=None, preprocessing_fn=None):
    image = load_image(image_path, config)
    mask = load_mask(mask_path, config)

    if augmentation_fn:
        image, mask = tf.numpy_function(
            lambda img, msk: augmentation(img, msk, augmentation_fn),
            [image, mask],
            [tf.float32, tf.float32]
        )
        image.set_shape((*config['image_size'], 3))
        mask.set_shape((*config['image_size'], 4))

    if preprocessing_fn:
        image, mask = tf.numpy_function(
            lambda img, msk: preprocess(img, msk, preprocessing_fn),
            [image, mask],
            [tf.float32, tf.float32]
        )
    return image, mask

def load_data(images_dir, masks_dir, config, augmentation_fn=None, preprocessing_fn=None):
    image_paths = tf.data.Dataset.list_files(images_dir + '/*.jpg', shuffle=False)
    mask_paths = tf.data.Dataset.list_files(masks_dir + '/*.png', shuffle=False)
    dataset = tf.data.Dataset.zip((image_paths, mask_paths))
    dataset = dataset.map(
        lambda img, msk: process_path(img, msk, config, augmentation_fn, preprocessing_fn),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    return dataset

def prepare_dataset(dataset, batch_size, shuffle=False, train=True):
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    if train:
        dataset = dataset.repeat()
    return dataset

def main(args):
    config = load_config(args.config)
    aug_config = args.aug_config

    # Setup augmentations
    augmentation_fn = Augmentation(aug_config, image_size=(*config['image_size'], 3)).get_training_augmentation()

    # Load datasets
    train_dataset = load_data(
        images_dir=config['paths']['x_train_dir'],
        masks_dir=config['paths']['y_train_dir'],
        config=config,
        augmentation_fn=augmentation_fn,
        preprocessing_fn=None
    )

    valid_dataset = load_data(
        images_dir=config['paths']['x_valid_dir'],
        masks_dir=config['paths']['y_valid_dir'],
        config=config,
        augmentation_fn=None,
        preprocessing_fn=None
    )

    train_dataset = prepare_dataset(train_dataset, config['BATCH_SIZE'], shuffle=True, train=True)
    valid_dataset = prepare_dataset(valid_dataset, config['BATCH_SIZE'], shuffle=False, train=False)

    # Losses and metrics
    losses = Losses()
    combined_loss = losses.combine_losses(losses.categorical_focal_loss, losses.dice_loss)

    metrics = Metrics()
    metric_list = [
        metrics.iou_score,
        metrics.f1_score,
        metrics.dice,
        metrics.precision,
        metrics.recall
    ]

    ARCHITECTURE_MAP = {
        'unet': Unet,
        'fpn': FPN,
        'linknet': LinkNet,
        'pspnet': PSPNet
    }

    arch_name = config['ARCHITECTURE'].lower()
    if arch_name not in ARCHITECTURE_MAP:
        raise ValueError(f"Unsupported architecture: {arch_name}")

    ModelClass = ARCHITECTURE_MAP[arch_name]
    model = ModelClass(
        config['BACKBONE'],
        input_shape=(*config['image_size'], 3),
        classes=config['n_classes'],
        activation='sigmoid'
    )

    model.compile(optimizer=tf.keras.optimizers.Adam(config['LR']), loss=combined_loss, metrics=metric_list)

    # Callbacks
    log_csv = CSVLogger('train_logs.csv', separator=',', append=False)
    checkpoint = ModelCheckpoint(
        filepath=f"{config['BACKBONE']}_{arch_name}.keras",
        monitor='val_iou_score',
        verbose=1,
        save_best_only=True,
        mode='max'
    )

    callbacks = [checkpoint, ReduceLROnPlateau(), log_csv]

    steps_per_epoch = len(os.listdir(config['paths']['x_train_dir'])) // config['BATCH_SIZE']

    model.fit(
        train_dataset,
        steps_per_epoch=steps_per_epoch,
        epochs=config['EPOCHS'],
        validation_data=valid_dataset,
        callbacks=callbacks
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a segmentation model.")
    parser.add_argument('--config', required=True, help="Path to training config file (YAML).")
    parser.add_argument('--aug_config', required=True, help="Path to augmentation config file (YAML).")
    args = parser.parse_args()
    main(args)

