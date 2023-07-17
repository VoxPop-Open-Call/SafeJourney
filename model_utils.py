import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import os
import numpy as np
import pandas as pd
from tensorflow.keras import layers
import yaml
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import RocCurveDisplay
from itertools import cycle
from sklearn.preprocessing import LabelBinarizer
import random

sess = tf.compat.v1.Session()

SEED = 42


def reset_seeds():
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)


def plot_roc(
    train_to_eval,
    labels_to_eval,
    prob_list_to_eval,
    task_classes,
    task_to_eval,
    nbr_classes,
):
    label_binarizer = LabelBinarizer().fit(train_to_eval)
    y_onehot_test = label_binarizer.transform(labels_to_eval)
    y_a_preds = prob_list_to_eval

    COLORS = []
    for color in mcolors.TABLEAU_COLORS.items():
        COLORS.append(color[1])

    colors = cycle(COLORS)
    fig, ax = plt.subplots(figsize=(8, 8))
    for i, color in zip(range(nbr_classes), colors):
        class_of_interest = task_classes[task_to_eval][i]
        class_id = np.flatnonzero(label_binarizer.classes_ == class_of_interest)[0]
        # class_id

        RocCurveDisplay.from_predictions(
            y_onehot_test[:, class_id],
            y_a_preds[:, class_id],
            name=f"{class_of_interest} vs the rest",
            color=color,
            ax=ax,
        )

    plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"One-vs-Rest ROC curves:\n{task_to_eval}")
    plt.legend()
    plt.show()

    return


def plot_metrics(history, train, val, metric):
    plt.plot(history[train])
    plt.plot(history[val])
    plt.title(train)
    plt.ylabel(metric)
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.show()


def focal_loss(gamma=2.0, alpha=4.0):
    gamma = float(gamma)
    alpha = float(alpha)

    def focal_loss_fixed(y_true, y_pred):
        """Focal loss for multi-classification
        FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
        Notice: y_pred is probability after softmax
        gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
        d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
        Focal Loss for Dense Object Detection
        https://arxiv.org/abs/1708.02002
        Arguments:
            y_true {tensor} -- ground truth labels, shape of [batch_size, num_cls]
            y_pred {tensor} -- model's output, shape of [batch_size, num_cls]
        Keyword Arguments:
            gamma {float} -- (default: {2.0})
            alpha {float} -- (default: {4.0})
        Returns:
            [tensor] -- loss.
        """
        epsilon = 1.0e-9
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)

        model_out = tf.math.add(y_pred, epsilon)
        ce = tf.math.multiply(y_true, -tf.math.log(model_out))
        weight = tf.math.multiply(
            y_true, tf.math.pow(tf.subtract(1.0, model_out), gamma)
        )
        fl = tf.math.multiply(alpha, tf.math.multiply(weight, ce))
        reduced_fl = tf.reduce_max(fl, axis=1)
        return tf.reduce_mean(reduced_fl)

    return focal_loss_fixed


def get_config_yaml(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def build_model(output_shapes: dict, input_shape=(224, 224, 3), dropout_rate=0.4):
    # inputs
    input_image = layers.Input(shape=input_shape)
    ipt_image = layers.Lambda(lambda x: tf.image.resize(x, (448, 448)))(input_image)

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(448, 448, 3), include_top=False, weights="imagenet", pooling="max"
    )
    base_model.trainable = False

    x = base_model(ipt_image)
    # Add dense layers after max pooling
    x = layers.Dense(200, activation="relu")(x)
    shareable_output = layers.Dropout(dropout_rate)(x)

    # define task layers
    outputs = []
    for task, shape in output_shapes.items():
        print(task)
        x = layers.Dense(50, activation="relu")(shareable_output)
        x = layers.Dropout(dropout_rate)(x)
        outputs.append(layers.Dense(shape, activation="softmax", name=task)(x))

    model = tf.keras.Model(input_image, outputs, name="Mobile-Net")

    return model


class DataGenDepth(tf.keras.utils.Sequence):
    def __init__(
        self,
        dataframe,
        task_classes,
        relative_path="",
        batch_size=16,
        image_path_column="path",
        shuffle=True,
        seed=1,
        augmentation=False,
    ):
        self.dataframe = dataframe
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.task_classes = task_classes
        self.image_path_column = image_path_column
        self.relative_path = relative_path

        self.le = {k: LabelEncoder().fit(v) for k, v in self.task_classes.items()}
        self.augmentation = augmentation

        self.n = len(self.dataframe)

    def on_epoch_end(self):
        if self.shuffle:
            self.dataframe = self.dataframe.sample(
                frac=1, random_state=self.seed
            ).reset_index(drop=True)

    def __getitem__(self, index):
        batches = self.dataframe[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        ret = self.__get_data(batches)
        return ret

    def __len__(self):
        return self.n // self.batch_size

    def __get_input(self, path):
        image = tf.keras.preprocessing.image.load_img(
            os.path.join(self.relative_path, path)
        )

        image = tf.keras.preprocessing.image.img_to_array(image)
        image = tf.image.resize(image, (448, 448))

        image_arr = tf.keras.applications.mobilenet_v2.preprocess_input(image)

        if self.augmentation:
            data_augmentation = tf.keras.Sequential([layers.RandomFlip("horizontal")])

            image_arr = data_augmentation(image_arr)

            # image_arr = tf.keras.preprocessing.image.random_rotation(
            #    image_arr, 20, fill_mode="nearest",
            #    row_axis=0, col_axis=1, channel_axis=2
            # )
            # image_arr = tf.keras.preprocessing.image.random_shift(
            #    image_arr, 0.2, 0.2, fill_mode="nearest",
            #    row_axis=0, col_axis=1, channel_axis=2
            # )
            # print(type(image_arr))
        if type(image_arr).__module__ != np.__name__:
            # with sess.as_default():

            #     return image_arr.eval()
            return image_arr.numpy()
        else:
            return image_arr
        # return image_arr

    def __get_output(self, label, task):
        label = self.le[task].transform([label])
        return tf.keras.utils.to_categorical(
            label, len(self.task_classes[task])
        ).ravel()

    def __get_data(self, batches):
        image_paths = batches[self.image_path_column]

        X_batch = np.asarray([self.__get_input(x) for x in image_paths])

        y_batch = {}

        for key in self.task_classes.keys():
            y_batch[key] = np.asarray([self.__get_output(x, key) for x in batches[key]])

        # print("DEBUG: ",y_batch)

        # y_batch = {k: np.asarray([self.__get_output(x, k) for x in batches[k]])}

        return X_batch, y_batch
