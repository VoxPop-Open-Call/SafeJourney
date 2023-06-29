import os
import pickle
import argparse
import tensorflow as tf
import pandas as pd
from datetime import datetime
import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from utils import choose_gpu
from tqdm.keras import TqdmCallback

from model_utils import (
    build_model,
    DataGenDepth,
    get_config_yaml,
    focal_loss,
    reset_seeds,
)

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# tf.compat.v1.disable_eager_execution()
tf.__version__
choose_gpu()

PARSER = argparse.ArgumentParser(
    description="Script to train image classification model"
)

PARSER.add_argument(
    "--config-path",
    metavar="c",
    type=str,
    nargs="?",
    help="Path to the yaml config file",
)

ARGS = PARSER.parse_args()

reset_seeds()

config = get_config_yaml(ARGS.config_path)

model_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# === GET DATA FROM CSV ====================================================

data = pd.read_csv(config["data_file_path"])

train_size = config["training_parameters"]["train_test_split"][0]
test_size = config["training_parameters"]["train_test_split"][1]

gss = GroupShuffleSplit(
    n_splits=1, test_size=test_size, train_size=train_size, random_state=42
)

for train_idx, test_idx in gss.split(data, data, data["img_name"]):
    train = data.iloc[train_idx].reset_index(drop=True)
    test = data.iloc[test_idx].reset_index(drop=True)

train.to_csv(f"{config['train_val_output_folder']}train_{model_timestamp}.csv")
test.to_csv(f"{config['train_val_output_folder']}val_{model_timestamp}.csv")
# === BUILD THE MODEL ======================================================

training_parameters = config["training_parameters"]
output_shapes = {k: len(v) for k, v in config["categories"].items()}
model = build_model(output_shapes, input_shape=(640, 640, 3))
task_classes = config["categories"]


task_losses = {k: "categorical_crossentropy" for k in output_shapes.keys()}
print(task_losses)

ROC = tf.keras.metrics.AUC(curve="ROC")

PR_ROC = tf.keras.metrics.AUC(curve="PR")

Recall = tf.keras.metrics.Recall()
Precision = tf.keras.metrics.Precision()

model.compile(
    optimizer=Adam(learning_rate=training_parameters["learning_rate"]),
    loss=task_losses,
    metrics=["accuracy", ROC, PR_ROC, Recall, Precision],
)


# === TRAIN AND SAVE THE MODEL ======================================================

image_path = ""
train_datagen = DataGenDepth(
    train,
    relative_path=image_path,
    task_classes=task_classes,
    batch_size=training_parameters["batch_size"],
    image_path_column="path",
    augmentation=True,
)
val_datagen = DataGenDepth(
    test,
    relative_path=image_path,
    task_classes=task_classes,
    batch_size=training_parameters["batch_size"],
    image_path_column="path",
    shuffle=True,
)

model_output = config["model_output "]  # data

earlyStopping = EarlyStopping(
    monitor="val_loss",
    patience=config["training_parameters"]["patience"],
    verbose=0,
    mode="min",
    restore_best_weights=True,
)
history = model.fit(
    train_datagen,
    epochs=config["training_parameters"]["max_epochs"],
    verbose=config["training_parameters"]["verbose"],
    validation_data=val_datagen,
    callbacks=[earlyStopping],
)
model.save(os.path.join(model_output, f"model_{model_timestamp}.h5"))

# === SAVE THE MODEL HISTORY ======================================================

history_path = os.path.join(model_output, f"model_{model_timestamp}_history")

print(history_path)
with open(history_path, "wb") as file_pi:
    pickle.dump(history.history, file_pi)


print("======================= DONE ==============================")
