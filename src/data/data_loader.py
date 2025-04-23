import tensorflow as tf
import glob
import numpy as np

class DataLoader:
    def __init__(self, cfg,augmentation_fn,dataset_type):
        self.image_size = cfg["model"]["image_size"]
        self.dataset_type = dataset_type
        if self.dataset_type == 'train':
            self.path = cfg['data']["train_data"]
            self.batch_size = cfg['data']['train_batch_size']
        elif self.dataset_type == 'val':
            self.path = cfg['data']["val_data"]
            self.batch_size = cfg['data']['val_batch_size']
        self.height, self.width, self.channel = self.image_size
        self.image_path = self.path + '/' + 'images'
        self.mask_path = self.path + '/' + 'masks'
        self.augmentation_fn = augmentation_fn
        self.dataset_type = dataset_type

    def load_image(self,img):
        image = tf.io.read_file(img)
        image = tf.image.decode_jpeg(image, channels=self.channel)
        image = tf.image.resize(image, [self.height, self.width])
        image = tf.cast(image, tf.float32) / 255.0
        return image

    def load_mask(self,msk):
        mask = tf.io.read_file(msk)
        mask = tf.image.decode_png(mask, channels=1)
        mask = tf.image.resize(mask, [self.height, self.width], method='nearest')
        mask = tf.cast(mask, tf.float32) / 255.0
        return mask

    def augmentation(self, image, mask):
        def _augment(img, msk):
            img, msk = img.numpy(), msk.numpy()  # Convert tensors to NumPy
            sample = self.augmentation_fn(image=img, mask=msk)
            return sample['image'].astype(np.float32), sample['mask'].astype(np.float32)

        image, mask = tf.py_function(_augment, [image, mask], [tf.float32, tf.float32])
        return image, mask


    def process_path(self, img, msk):
        image = self.load_image(img)
        mask = self.load_mask(msk)
        return image, mask

    def load_data(self):
        image_paths = sorted(glob.glob(self.image_path + '/*.jpg') + glob.glob(self.image_path + '/*.png'))
        mask_paths = sorted(glob.glob(self.mask_path + '/*.jpg') + glob.glob(self.mask_path + '/*.png'))
        image_paths = tf.data.Dataset.from_tensor_slices(image_paths)
        mask_paths = tf.data.Dataset.from_tensor_slices(mask_paths)
        dataset = tf.data.Dataset.zip((image_paths, mask_paths))
        dataset = dataset.map(lambda img_path, msk_path: self.process_path(img_path, msk_path),
                              num_parallel_calls=tf.data.AUTOTUNE)
        return dataset

    def prepare_dataset(self, dataset):
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        #dataset = dataset.repeat()
        return dataset

    def get_dataset(self):
        dataset = self.load_data()
        dataset = self.prepare_dataset(dataset)
        return dataset




