import yaml
import albumentations as A

class Augmentation:
    def __init__(self, yaml_path, image_size=(512, 512,3)):
        self.image_size = image_size
        self.config = self._load_config(yaml_path)

    def _load_config(self, yaml_path):
        with open(yaml_path, "r") as file:
            return yaml.safe_load(file)

    def get_training_augmentation(self):
        height, width,_ = self.image_size
        aug_list = []
        aug_cfg = self.config.get("augmentation", {})

        if aug_cfg.get("horizontal_flip", {}).get("enabled", False):
            aug_list.append(A.HorizontalFlip(p=aug_cfg["horizontal_flip"]["probability"]))

        if aug_cfg.get("shift_scale_rotate", {}).get("enabled", False):
            aug_list.append(A.ShiftScaleRotate(
                scale_limit=aug_cfg["shift_scale_rotate"]["scale_limit"],
                rotate_limit=aug_cfg["shift_scale_rotate"]["rotate_limit"],
                shift_limit=aug_cfg["shift_scale_rotate"]["shift_limit"],
                p=aug_cfg["shift_scale_rotate"]["probability"],
                border_mode=aug_cfg["shift_scale_rotate"]["border_mode"]

            ))

        if aug_cfg.get("pad_if_needed", {}).get("enabled", False):
            aug_list.append(A.PadIfNeeded(
                min_height=height,
                min_width=width,
                border_mode=aug_cfg["pad_if_needed"]["border_mode"],
		value=0
            ))

        if aug_cfg.get("perspective", {}).get("enabled", False):
            aug_list.append(A.Perspective(p=aug_cfg["perspective"]["probability"]))

        if aug_cfg.get("brightness_contrast_gamma", {}).get("enabled", False):
            aug_list.append(A.OneOf([
                A.RandomBrightnessContrast(p=aug_cfg["brightness_contrast_gamma"]["RandomBrightnessContrast_prob"]),
                A.RandomGamma(p=aug_cfg["brightness_contrast_gamma"]["RandomGamma_prob"]),
            ], p=aug_cfg["brightness_contrast_gamma"]["probability"]))

        if aug_cfg.get("random_crop", {}).get("enabled", False):
            aug_list.append(A.RandomCrop(
                height=aug_cfg["random_crop"]["height"],
                width=aug_cfg["random_crop"]["width"],
                p=aug_cfg["random_crop"]["probability"]
            ))
            aug_list.append(A.PadIfNeeded(min_height=height, min_width=width, border_mode=0,value=0, p=1.0))

        if aug_cfg.get("motion_blur", {}).get("enabled", False):
            aug_list.append(A.OneOf([
                A.MotionBlur(p=1),
                A.MedianBlur(blur_limit=3, p=1),
                A.Blur(blur_limit=3, p=1),
            ], p=aug_cfg["motion_blur"]["probability"]))

        if aug_cfg.get("distortion", {}).get("enabled", False):
            aug_list.append(A.OneOf([
                A.OpticalDistortion(p=aug_cfg["distortion"]["OpticalDistortion_prob"]),
                A.GridDistortion(p=aug_cfg["distortion"]["GridDistortion_prob"]),
                A.PiecewiseAffine(p=aug_cfg["distortion"]["PiecewiseAffine_prob"]),
            ], p=aug_cfg["distortion"]["probability"]))

        return A.Compose(aug_list)

    def get_validation_augmentation(self):
        height, width,_ = self.image_size
        return A.Compose([
            A.PadIfNeeded(height, width, border_mode=0)
        ])