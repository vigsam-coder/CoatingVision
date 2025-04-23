import tensorflow as tf

# Jaccard Loss
class JaccardLoss(tf.keras.losses.Loss):
    def __init__(self, smooth=1e-6, name="jaccard_loss"):
        super().__init__(name=name)
        self.smooth = smooth

    def call(self, y_true, y_pred):
        intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
        union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3]) - intersection
        jaccard = (intersection + self.smooth) / (union + self.smooth)
        return 1 - jaccard

# Dice Loss
class DiceLoss(tf.keras.losses.Loss):
    def __init__(self, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE, name="dice_loss"):
        super().__init__(name=name, reduction=reduction)

    def call(self, y_true, y_pred):
        numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
        denominator = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
        return 1 - (numerator / (denominator + 1e-6))

class Losses:
    def __init__(self):
        self.binary_focal_loss = tf.keras.losses.BinaryFocalCrossentropy(
            apply_class_balancing=False,
            alpha=0.25,
            gamma=2.0,
            from_logits=False,
            label_smoothing=0.0,
            axis=-1,
            reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
            name='binary_focal_crossentropy'
        )

        self.categorical_focal_loss = tf.keras.losses.CategoricalFocalCrossentropy(
            alpha=0.25,
            gamma=2.0,
            from_logits=False,
            label_smoothing=0.0,
            axis=-1,
            reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
            name='categorical_focal_crossentropy'
        )

        self.binary_crossentropy = tf.keras.losses.BinaryCrossentropy(
            from_logits=False,
            label_smoothing=0.0,
            axis=-1,
            reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
            name='binary_crossentropy'
        )

        self.categorical_crossentropy = tf.keras.losses.CategoricalCrossentropy(
            from_logits=False,
            label_smoothing=0.0,
            axis=-1,
            reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
            name='categorical_crossentropy'
        )

        self.jaccard_loss = JaccardLoss(smooth=1e-6, name="jaccard_loss")
        self.dice_loss = DiceLoss(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE, name="dice_loss")

    def get_loss(self, loss_name):
        loss_dict = {
            'binary_focal_loss': self.binary_focal_loss,
            'categorical_focal_loss': self.categorical_focal_loss,
            'binary_crossentropy': self.binary_crossentropy,
            'categorical_crossentropy': self.categorical_crossentropy,
            'jaccard_loss': self.jaccard_loss,
            'dice_loss': self.dice_loss
        }
        return loss_dict.get(loss_name, None)

    def combine_losses(self, loss1, loss2):
        def loss_fn(y_true, y_pred):
            return loss1(y_true, y_pred) + loss2(y_true, y_pred)
        return loss_fn


# Instantiate Losses class
#losses = Losses()

# Example combined losses
#bce_dice_loss = losses.combine_losses(losses.binary_crossentropy, losses.dice_loss)
#bce_jaccard_loss = losses.combine_losses(losses.binary_crossentropy, losses.jaccard_loss)
#cce_dice_loss = losses.combine_losses(losses.categorical_crossentropy, losses.dice_loss)
#cce_jaccard_loss = losses.combine_losses(losses.categorical_crossentropy, losses.jaccard_loss)
#binary_focal_dice_loss = losses.combine_losses(losses.binary_focal_loss, losses.dice_loss)
#binary_focal_jaccard_loss = losses.combine_losses(losses.binary_focal_loss, losses.jaccard_loss)
#categorical_focal_dice_loss = losses.combine_losses(losses.categorical_focal_loss, losses.dice_loss)
#categorical_focal_jaccard_loss = losses.combine_losses(losses.categorical_focal_loss, losses.jaccard_loss)
