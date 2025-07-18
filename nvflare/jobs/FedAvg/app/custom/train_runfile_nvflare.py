"""
Training runfile

Date: September 2023
Author: Wout Decrop (based on code from Ignacio Heredia)
Email: wout.decrop@VLIZ.be
Github: lifewatch

Description:
This file contains the commands for training a convolutional net for image classification for phytoplankton.

Additional notes:
* On the training routine: Preliminary tests show that using a custom lr multiplier for the lower layers yield to better
results than freezing them at the beginning and unfreezing them after a few epochs like it is suggested in the Keras
tutorials.
"""

import json
import logging
import os
import time
from datetime import datetime

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

# (1) import nvflare client API
import nvflare.client as flare

from planktonclas import config, model_utils, paths, utils
from planktonclas.data_utils import (compute_classweights, compute_meanRGB,
                                     create_data_splits, data_sequence,
                                     json_friendly, k_crop_data_sequence,
                                     load_aphia_ids, load_class_names,
                                     load_data_splits)
from planktonclas.optimizers import customAdam
from nvflare.client.tracking import MLflowWriter

# TODO: Add additional metrics for test time in addition to accuracy






# from planktonclas.api import load_inference_model

# Set Tensorflow verbosity logs
tf.get_logger().setLevel(logging.ERROR)


# Allow GPU memory growth
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def main(TIMESTAMP, CONF):
    # (2) initialize nvflare
    flare.init()
    mlflow_writer = MLflowWriter()
    mlflow_writer.log_params(CONF)
    paths.timestamp = TIMESTAMP
    paths.CONF = CONF

    utils.create_dir_tree()
    utils.backup_splits()

    client_id = flare.get_site_name()

    

    
    

    if f'{client_id}_train.txt' not in os.listdir(paths.get_ts_splits_dir()):
        if not (CONF["dataset"]["split_ratios"]):
            if (CONF["training"]["use_validation"]) & (CONF["training"]["use_test"]):
                split_ratios = [0.8, 0.1, 0.1]
            elif (CONF["training"]["use_validation"]) & (~CONF["training"]["use_test"]):
                split_ratios = [0.9, 0.1, 0]
            else:
                split_ratios = [1, 0, 0]
        else:
            split_ratios = CONF["dataset"]["split_ratios"]
        create_data_splits(
            splits_dir=paths.get_ts_splits_dir(),
            im_dir=paths.get_images_dir(),
            split_ratios=split_ratios,
        )
    # Load the training data
    X_train, y_train = load_data_splits(
        splits_dir=paths.get_ts_splits_dir(),
        im_dir=paths.get_images_dir(),
        split_name="train",
    )
    
    # Load the validation data
    if (CONF["training"]["use_validation"]) and (
        f'{client_id}_val.txt' in os.listdir(paths.get_ts_splits_dir())
    ):
        X_val, y_val = load_data_splits(
            splits_dir=paths.get_ts_splits_dir(),
            im_dir=paths.get_images_dir(),
            split_name="val",
        )
    else:
        print("No validation data.")
        X_val, y_val = None, None
        CONF["training"]["use_validation"] = False

    # Load the class names
    class_names = load_class_names(splits_dir=paths.get_ts_splits_dir())
    aphia_ids = load_aphia_ids(splits_dir=paths.get_ts_splits_dir())
    # Update the configuration
    CONF["model"]["preprocess_mode"] = model_utils.model_modes[
        CONF["model"]["modelname"]
    ]
    CONF["training"]["batch_size"] = min(CONF["training"]["batch_size"], len(X_train))

    if CONF["model"]["num_classes"] is None:
        CONF["model"]["num_classes"] = len(class_names)

    # Compute the class weights
    if CONF["training"]["use_class_weights"]:
        class_weights = compute_classweights(
            y_train, max_dim=CONF["model"]["num_classes"]
        )
    else:
        class_weights = None

    # Compute the mean and std RGB values
    if CONF["dataset"]["mean_RGB"] is None:
        CONF["dataset"]["mean_RGB"], CONF["dataset"]["std_RGB"] = compute_meanRGB(
            X_train
        )

    # Create data generator for train and val sets
    train_gen = data_sequence(
        X_train,
        y_train,
        batch_size=CONF["training"]["batch_size"],
        num_classes=CONF["model"]["num_classes"],
        im_size=CONF["model"]["image_size"],
        mean_RGB=CONF["dataset"]["mean_RGB"],
        std_RGB=CONF["dataset"]["std_RGB"],
        preprocess_mode=CONF["model"]["preprocess_mode"],
        aug_params=CONF["augmentation"]["train_mode"],
    )
    train_steps = int(np.ceil(len(X_train) / CONF["training"]["batch_size"]))

    if CONF["training"]["use_validation"]:
        val_gen = data_sequence(
            X_val,
            y_val,
            batch_size=CONF["training"]["batch_size"],
            num_classes=CONF["model"]["num_classes"],
            im_size=CONF["model"]["image_size"],
            mean_RGB=CONF["dataset"]["mean_RGB"],
            std_RGB=CONF["dataset"]["std_RGB"],
            preprocess_mode=CONF["model"]["preprocess_mode"],
            aug_params=CONF["augmentation"]["val_mode"],
        )
        val_steps = int(np.ceil(len(X_val) / CONF["training"]["batch_size"]))
    else:
        val_gen = None
        val_steps = None

    # Launch the training
    t0 = time.time()

    # Create the model and compile it
    model, base_model = model_utils.create_model(CONF)
    print('the training conf', CONF)
    # Get a list of the top layer variables that should not be applied a
    # lr_multiplier
    base_vars = [var.name for var in base_model.trainable_variables]
    all_vars = [var.name for var in model.trainable_variables]
    top_vars = set(all_vars) - set(base_vars)
    top_vars = list(top_vars)

    # Set trainable layers
    if CONF["training"]["mode"] == "fast":
        for layer in base_model.layers:
            layer.trainable = False

    model.compile(
        optimizer=customAdam(
            learning_rate=CONF["training"]["initial_lr"],
            amsgrad=True,
            lr_mult=0.1,
            excluded_vars=top_vars,
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    

    @flare.train
    def train(input_model=None):

        # (5) loads model from NVFlare
        for k, v in input_model.params.items():
            #print(f"Loading model weights for layer {k}")
            model.get_layer(k).set_weights(v)


        history = model.fit(
            x=train_gen,
            steps_per_epoch=train_steps,
            epochs=CONF["training"]["epochs"],
            class_weight=class_weights,
            validation_data=val_gen,
            validation_steps=val_steps,
            callbacks=utils.get_callbacks(CONF),
            verbose=1,
            initial_epoch=0,
        )

        if 'val_accuracy' in history.history.keys():
             mlflow_writer.log_metrics(
           metrics={'accuracy':history.history['accuracy'][-1], 'val_accuracy':history.history['val_accuracy'][-1]} , step=input_model.current_round
        )
        else:
            mlflow_writer.log_metrics(
           metrics={'accuracy':history.history['accuracy'][-1]} , step=input_model.current_round
        )

        # Saving everything
        print("Saving data to {} folder.".format(paths.get_timestamped_dir()))
        print("Saving training stats ...")
        stats = {
            "epoch": history.epoch,
            "training time (s)": round(time.time() - t0, 2),
            "timestamp": TIMESTAMP,
        }
        stats.update(history.history)
        stats = json_friendly(stats)
        stats_dir = paths.get_stats_dir()
        with open(os.path.join(stats_dir, "stats.json"), "w") as outfile:
            json.dump(stats, outfile, sort_keys=True, indent=4)

        print("Saving the configuration ...")
        model_utils.save_conf(CONF)

        print("Saving the model to h5...")
        fpath = os.path.join(paths.get_checkpoints_dir(), "final_model.h5")
        #model.save(fpath, include_optimizer=False)

    
        print('local model summary')
      #  print(model.summary())
        print("Finished training")

        if CONF["training"]["use_test"]:
            print("Start testing")
            X_test, y_test = load_data_splits(
                splits_dir=paths.get_ts_splits_dir(),
                im_dir=paths.get_images_dir(),
                split_name="test",
            )
            crop_num = 10
            filemode = "local"
            test_gen = k_crop_data_sequence(
                inputs=X_test,
                im_size=CONF["model"]["image_size"],
                mean_RGB=CONF["dataset"]["mean_RGB"],
                std_RGB=CONF["dataset"]["std_RGB"],
                preprocess_mode=CONF["model"]["preprocess_mode"],
                aug_params=CONF["augmentation"]["val_mode"],
                crop_mode="random",
                crop_number=crop_num,
                filemode=filemode,
            )
            top_K = 5

            output = model.predict(
                test_gen,
                verbose=1,
              #  max_queue_size=10,
               # workers=4,
             #   use_multiprocessing=CONF["training"]["use_multiprocessing"],
            )

            # reshape to (N, crop_number, num_classes)
            output = output.reshape(len(X_test), -1, output.shape[-1])
            output = np.mean(output, axis=1)  # take the mean across the crops

            # sort labels in descending prob
            lab = np.argsort(output, axis=1)[:, ::-1]
            lab = lab[:, :top_K]  # keep only top_K labels
            prob = output[
                np.repeat(np.arange(len(lab)), lab.shape[1]), lab.flatten()
            ].reshape(
                lab.shape
            )  # retrieve corresponding probabilities

            pred_lab, pred_prob = lab, prob

            if aphia_ids is not None:
                pred_aphia_ids = [aphia_ids[i] for i in pred_lab]
                pred_aphia_ids = [aphia_id.tolist() for aphia_id in pred_aphia_ids]
            else:
                pred_aphia_ids = aphia_ids

            class_index_map = {
                index: class_name for index, class_name in enumerate(class_names)
            }

            # Convert arrays of strings to lists of integers
            pred_lab_names = [
                [class_index_map[label] for label in labels] for labels in pred_lab
            ]
            y_test_names = [class_index_map.get(index) for index in y_test]

            # Save the predictions
            pred_dict = {
                "filenames": list(X_test),
                "pred_lab": pred_lab.tolist(),
                "pred_prob": pred_prob.tolist(),
                "pred_lab_names": pred_lab_names,
                "aphia_ids": pred_aphia_ids,
            }
            if y_test is not None:
                pred_dict["true_lab"] = y_test.tolist()
                pred_dict["true_lab_names"] = y_test_names

            pred_path = os.path.join(
                paths.get_predictions_dir(),
                "{}+{}+top{}.json".format("final_model.h5", "DS_split", top_K),
            )
            with open(pred_path, "w") as outfile:
                json.dump(pred_dict, outfile, sort_keys=True)
            print(f"Predictions file saved in: {pred_path}")
        print("finished testing")
        
        # (3) send back the model to nvflare server
        output_model = flare.FLModel(
            params={layer.name: layer.get_weights() for layer in model.layers},
            current_round=input_model.current_round,
           
        )

        return output_model
    
    # (4) gets FLModel from NVFlare
    while flare.is_running():
        input_model = flare.receive()
        current_round = input_model.current_round
        total_rounds = input_model.total_rounds

        print(f"Currently in round {current_round} out of a total of {total_rounds} rounds")

        # (optional) print system info
        system_info = flare.system_info()
        print(f"NVFlare system info: {system_info}")
       

    
        
        train(input_model=input_model)

        
        sys_info = flare.system_info()
        # LOG SOME INFO ABOUT THE CLIENT IN THE MLFLOW AS TAGS 
        #mlflow_writer.set_tags(sys_info)


if __name__ == "__main__":

    CONF = config.get_conf_dict()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    main(TIMESTAMP=timestamp, CONF=CONF)
