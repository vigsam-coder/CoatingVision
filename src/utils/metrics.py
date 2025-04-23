import tensorflow as tf

class DiceCoefficient(tf.keras.metrics.Metric):
    def __init__(self, name="dice_coefficient", **kwargs):
        super(DiceCoefficient, self).__init__(name=name, **kwargs)
        self.intersection = self.add_weight(name="intersection", initializer="zeros")
        self.union = self.add_weight(name="union", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred > 0.5, tf.float32)
        y_true = tf.cast(y_true, tf.float32)

        intersection = tf.reduce_sum(y_true * y_pred)
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)

        self.intersection.assign_add(intersection)
        self.union.assign_add(union)

    def result(self):
        return (2.0 * self.intersection + 1e-7) / (self.union + 1e-7)

    def reset_state(self):
        self.intersection.assign(0.0)
        self.union.assign(0.0)


class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name="f1_score", threshold=0.5, **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.threshold = threshold
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred > self.threshold, tf.float32)
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        precision = self.precision.result()
        recall = self.recall.result()
        return (2 * precision * recall) / (precision + recall + tf.keras.backend.epsilon())

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()


class Metrics:
    def __init__(self):
        self.iou_score = tf.keras.metrics.BinaryIoU(
            target_class_ids=[1],
            name="iou_score",
            dtype=tf.float32,
        )

        self.f1_score = F1Score(
            threshold=0.5,
            name="f1_score",
            dtype=tf.float32
        )

        self.dice = DiceCoefficient(name="dice_coefficient")

        self.precision = tf.keras.metrics.Precision(
            thresholds=0.5,
            name="precision",
            dtype=tf.float32
        )

        self.recall = tf.keras.metrics.Recall(
            thresholds=0.5,
            name="recall",
            dtype=tf.float32
        )

        self.accuracy = tf.keras.metrics.Accuracy(
            name="accuracy",
            dtype=tf.float32
        )

        self.binary_accuracy = tf.keras.metrics.BinaryAccuracy(
            name="binary_accuracy",
            dtype=tf.float32
        )

    def get_metric(self, metric_name):
        metric_dict = {
            "iou_score": self.iou_score,
            "f1_score": self.f1_score,
            "dice_coefficient": self.dice,
            "precision": self.precision,
            "recall": self.recall,
            "accuracy": self.accuracy,
            "binary_accuracy": self.binary_accuracy
        }
        return metric_dict.get(metric_name, None)

    def combine_metrics(self, metric1, metric2):
        return lambda y_true, y_pred: metric1(y_true, y_pred) + metric2(y_true, y_pred)


# Instantiate Metrics class
#metrics = Metrics()

# Example usage
#iou_score = metrics.iou_score
#f1_score = metrics.f1_score
#dice = metrics.dice
#precision = metrics.precision
#recall = metrics.recall
#accuracy = metrics.accuracy
#binary_accuracy = metrics.binary_accuracy
