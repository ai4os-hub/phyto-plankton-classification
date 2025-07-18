"""
API for the image classification package

Date: September 2018
Author: Ignacio Heredia
Email: iheredia@ifca.unican.es
Github: ignacioheredia

Notes: Based on https://github.com/indigo-dc/plant-classification-theano/blob/package/plant_classification/api.py

Descriptions:
The API will use the model files inside ../models/api. If not found it will use the model files of the last trained model.
If several checkpoints are found inside ../models/api/ckpts we will use the last checkpoint.

Warnings:
There is an issue of using Flask with Keras: https://github.com/jrosebr1/simple-keras-rest-api/issues/1
The fix done (using tf.get_default_graph()) will probably not be valid for standalone wsgi container e.g. gunicorn,
gevent, uwsgi.
"""

import builtins
import glob
import json
import os
import re
import tempfile
import warnings
import zipfile
from collections import OrderedDict
from datetime import datetime
from functools import wraps
from aiohttp.web import HTTPException

import numpy as np
import pkg_resources
import requests
import tensorflow as tf
from deepaas.model.v2.wrapper import UploadedFile
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from webargs import fields
import logging

import os
from webargs import fields


from planktonclas import config, paths, test_utils, utils
from planktonclas.data_utils import (
    load_aphia_ids,
    load_class_info,
    load_class_names,
)
from planktonclas.train_runfile import train_fn



logger = logging.getLogger(__name__)
ENV_LOG_LEVEL = os.getenv("API_LOG_LEVEL", default="INFO")
LOG_LEVEL = getattr(logging, ENV_LOG_LEVEL.upper())
logger.setLevel(LOG_LEVEL)


NOW = str("{:%Y_%m_%d_%H_%M_%S}".format(datetime.now()))

import os
from marshmallow import fields, ValidationError
from pathlib import Path
from marshmallow import ValidationError

loaded_ts, loaded_ckpt = None, None
graph, model, conf, class_names, class_info, aphia_ids = (
    None,
    None,
    None,
    None,
    None,
    None,
)

# Additional parameters
allowed_extensions = set(
    ["png", "jpg", "jpeg", "PNG", "JPG", "JPEG"]
)  # allow only certain file extensions
top_K = 5  # number of top classes predictions to return


def load_inference_model(timestamp=None, ckpt_name=None):
    """
    Load a model for prediction.

    Parameters
    ----------
    * timestamp: str
        Name of the timestamp to use. The default is the last timestamp in `./models`.
    * ckpt_name: str
        Name of the checkpoint to use. The default is the last checkpoint in `./models/[timestamp]/ckpts`.
    """
    global loaded_ts, loaded_ckpt
    global graph, model, conf, class_names, class_info, aphia_ids

    # Set the timestamp
    timestamp_list = next(os.walk(paths.get_models_dir()))[1]
    timestamp_list = sorted(timestamp_list)
    if not timestamp_list:
        raise Exception(
            "You have no models in your `./models` folder to be used for inference. "
            "Therefore the API can only be used for training."
        )
    elif timestamp is None:
        timestamp = timestamp_list[-1]
    elif timestamp not in timestamp_list:
        raise ValueError(
            "Invalid timestamp name: {}. Available timestamp names are: {}".format(
                timestamp, timestamp_list
            )
        )
    paths.timestamp = timestamp
    print("Using TIMESTAMP={}".format(timestamp))

    # Set the checkpoint model to use to make the prediction
    ckpt_list = os.listdir(paths.get_checkpoints_dir())
    ckpt_list = sorted(
        [name for name in ckpt_list if name.endswith(".h5")]
    )
    if not ckpt_list:
        raise Exception(
            "You have no checkpoints in your `./models/{}/ckpts` folder to be used for inference. ".format(
                timestamp
            )
            + "Therefore the API can only be used for training."
        )
    elif ckpt_name is None:
        ckpt_name = ckpt_list[-1]
    elif ckpt_name not in ckpt_list:
        raise ValueError(
            "Invalid checkpoint name: {}. Available checkpoint names are: {}".format(
                ckpt_name, ckpt_list
            )
        )
    print("Using CKPT_NAME={}".format(ckpt_name))

    # Clear the previous loaded model
    tf.keras.backend.clear_session()
    # Load the class names and info
    splits_dir = paths.get_ts_splits_dir()
    class_names = load_class_names(splits_dir=splits_dir)
    aphia_ids = load_aphia_ids(splits_dir)
    class_info = None
    if "info.txt" in os.listdir(splits_dir):
        class_info = load_class_info(splits_dir=splits_dir)
        if len(class_info) != len(class_names):
            warnings.warn(
                """The 'classes.txt' file has a different length than the 'info.txt' file.
            If a class has no information whatsoever you should leave that classes row empty or put a '-' symbol.
            The API will run with no info until this is solved."""
            )
            class_info = None
    if class_info is None:
        class_info = ["" for _ in range(len(class_names))]

    # Load training configuration
    conf_path = os.path.join(paths.get_conf_dir(), "conf.json")
    with open(conf_path) as f:
        conf = json.load(f)
        update_with_saved_conf(conf)

    # Load the model
    model = load_model(
        os.path.join(paths.get_checkpoints_dir(), ckpt_name),
        custom_objects=utils.get_custom_objects(),
        compile=False,
        # Disable deserialization of training config
    )

    loaded_ts = timestamp
    loaded_ckpt = ckpt_name


def update_with_saved_conf(saved_conf):
    """
    Update the default YAML configuration with the configuration saved from training
    """
    # Update the default conf with the user input
    CONF = config.CONF
    for group, val in sorted(CONF.items()):
        if group in saved_conf.keys():
            for g_key, g_val in sorted(val.items()):
                if g_key in saved_conf[group].keys():
                    g_val["value"] = saved_conf[group][g_key]

    # Check and save the configuration
    config.check_conf(conf=CONF)
    config.conf_dict = config.get_conf_dict(conf=CONF)


def update_with_query_conf(user_args):
    """
    Update the default YAML configuration with the user's input args from the API query
    """
    # Update the default conf with the user input
    CONF = config.CONF
    for group, val in sorted(CONF.items()):
        for g_key, g_val in sorted(val.items()):
            if g_key in user_args:
                raw_value = user_args[g_key]
                if not raw_value:
                    continue  # skip if the value is empty
                try:
                    # Try parsing as JSON
                    g_val["value"] = json.loads(raw_value)
                except json.JSONDecodeError:
                    # Fall back to treating it as a plain string
                    g_val["value"] = raw_value
    # Check and save the configuration
    config.check_conf(conf=CONF)
    config.conf_dict = config.get_conf_dict(conf=CONF)


def catch_error(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        try:
            pred = f(*args, **kwargs)
            return {"status": "OK", "predictions": pred}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    return wrap


def catch_localfile_error(file_list):
    # Error catch: Empty query
    if not file_list:
        raise ValueError("Empty query")

    # Error catch: Image format error
    for f in file_list:
        extension = os.path.basename(f.content_type).split("/")[-1]
        # extension = mimetypes.guess_extension(f.content_type)
        if extension not in allowed_extensions:
            raise ValueError(
                "Local image format error: "
                "At least one file is not in a standard image format ({}).".format(
                    allowed_extensions
                )
            )


def warm():
    try:
        load_inference_model()
    except Exception as e:
        print(e)


def prepare_files(directory):
    """
    Prepare a list of dictionaries with attributes mimicking UploadedFile from image files in the directory.

    :param directory: The directory to search for image files.
    :return: A list of dictionaries with attributes similar to UploadedFile.
    """
    # Get all image files from the directory with given extensions
    extensions = ["*.jpg", "*.png", "*.jpeg"]
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(directory, ext)))

    # Create a list of dictionaries with attributes similar to UploadedFile
    uploaded_files = []
    for file_path in files:
        # Extract the filename from the path
        file_name = os.path.basename(file_path)
        uploaded_files.append(
            UploadedFile(
                name="data",
                filename=file_path,
                content_type="image/jpeg",  # Adjust if necessary based on file type
                original_filename=file_name,
            )
        )
    return uploaded_files


@catch_error
def predict(**args):
    logger.debug("Predict with args: %s", args)
    try:
        if not any([args["image"], args["zip"]]):
            raise Exception(
                "You must provide either 'image' or 'file_location' or 'zip' in the payload"
            )

        if args["zip"]:
            # Check if zip file is provided
            zip_file = args["zip"]

            # Create a temporary directory to extract the files
            with tempfile.TemporaryDirectory() as temp_dir:
                # Extract the zip file
                with zipfile.ZipFile(
                    zip_file.filename, "r"
                ) as zip_ref:
                    zip_ref.extractall(temp_dir)

                # Get the list of files extracted from the zip
                folder_files = os.listdir(temp_dir)

                # Assign the list of files to args["files"]

                uploaded_files = []
                for file in folder_files:
                    file_path = os.path.join(temp_dir, file)
                    uploaded_files.append(
                        UploadedFile(
                            name="data",
                            filename=file_path,
                            content_type="image/jpeg",
                            original_filename=file,
                        )
                    )

                # Assign the list of files to args["files"]
                args["files"] = uploaded_files

                return predict_data(args)
        elif args["image"]:
            args["files"] = [
                args["image"]
            ]  # patch until list is available
            # raise RuntimeError("args files ", args["files"])
            print(args["files"])
            return predict_data(args)

    except Exception as err:
        raise HTTPException(reason=err) from err


def predict_data(args):
    """
    Function to predict an image in binary format
    """
    # Check user configuration
    logger.debug("Predict with args: %s", args)
    try:

        update_with_query_conf(args)
        conf = config.conf_dict

        merge = False
        catch_localfile_error(args["files"])

        if (
            loaded_ts != conf["testing"]["timestamp"]
            or loaded_ckpt != conf["testing"]["ckpt_name"]
        ):
            load_inference_model(
                timestamp=conf["testing"]["timestamp"],
                ckpt_name=conf["testing"]["ckpt_name"],
            )
            conf = config.conf_dict
        # Create a list with the path to the images
        filenames = [f.filename for f in args["files"]]
        print(filenames)
        original_filenames = [
            f.original_filename for f in args["files"]
        ]

        # with graph.as_default():
        pred_lab, pred_prob = test_utils.predict(
            model=model,
            X=filenames,
            conf=conf,
            top_K=top_K,
            filemode="local",
            merge=merge,
            use_multiprocessing=False,
        )

        if merge:
            pred_lab, pred_prob = np.squeeze(pred_lab), np.squeeze(
                pred_prob
            )
            print(pred_lab.shape, pred_prob.shape)

        return format_prediction(
            pred_lab, pred_prob, original_filenames
        )
    except Exception as err:
        raise HTTPException(reason=err) from err


def get_predictions_dir(CONF):
    file_location = CONF.get("testing", {}).get("file_location", None)
    output_directory = CONF["testing"]["output_directory"]

    if file_location is not None:
        if os.path.exists(file_location):
            os.makedirs(
                os.path.join(
                    os.path.dirname(file_location), "predictions"
                ),
                exist_ok=True,
            )
            return os.path.join(
                os.path.dirname(file_location), "predictions"
            )
    else:
        if output_directory is None:
            # Define your get_timestamped_dir() function accordingly
            return os.path.join(
                paths.get_timestamped_dir(), "predictions"
            )
        else:
            return os.path.join(output_directory)


def format_prediction(labels, probabilities, original_filenames):
    if aphia_ids is not None:
        pred_aphia_ids = [aphia_ids[i] for i in labels]
        pred_aphia_ids = [
            aphia_id.tolist() for aphia_id in pred_aphia_ids
        ]
    else:
        pred_aphia_ids = aphia_ids
    class_index_map = {
        index: class_name
        for index, class_name in enumerate(class_names)
    }
    pred_lab_names = [
        [class_index_map[label] for label in labels]
        for labels in labels
    ]
    # pred_labels=[class_names[i] for i in labels]
    pred_prob = probabilities

    pred_dict = {
        "filenames": list(original_filenames),
        "pred_lab": pred_lab_names,  # Use converted list
        "pred_prob": pred_prob.tolist(),
        "aphia_ids": pred_aphia_ids,
    }
    conf = config.conf_dict
    ckpt_name = conf["testing"]["ckpt_name"]
    split_name = "test"
    pred_path = os.path.join(
        get_predictions_dir(conf),
        "{}+{}+top{}.json".format(ckpt_name, split_name, top_K),
    )
    with open(pred_path, "w") as outfile:
        json.dump(pred_dict, outfile, sort_keys=True)

    return pred_dict


def get_directory_choices(base_path="/srv/data/"):
    # Get a list of all directories in the base_path
    try:
        directories = [
            d
            for d in os.listdir(base_path)
            if os.path.isdir(os.path.join(base_path, d))
        ]
        return directories
    except Exception as e:
        print(f"Error accessing directories: {e}")
        return []


def validate_directory(path):
    # Convert the input to a Path object if it's a string
    if isinstance(path, str):
        path = Path(
            path.strip("'\"")
        )  # Remove any leading/trailing quotes

    # Check if the path is a valid directory
    if not path.is_dir():
        raise ValidationError(f"{path} is not a valid directory")

    return path


from pathlib import Path


def train(**args):
    """
    Train an image classifier
    """
    try:
        logger.info("Training model...")
        logger.debug("Train with args: %s", args)

        update_with_query_conf(user_args=args)
        CONF = config.conf_dict
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        config.print_conf_table(CONF)
        K.clear_session()  # remove the model loaded for prediction

        validate_directory(args["images_directory"])

        train_fn(TIMESTAMP=timestamp, CONF=CONF)

        return {"modelname": timestamp}

    except Exception as err:
        logger.critical(err, exc_info=True)
        raise HTTPException(reason=err) from err


def populate_parser(parser, default_conf):
    """
    Returns a arg-parse like parser.
    """
    print("popu parser")
    for group, val in default_conf.items():
        for g_key, g_val in val.items():
            gg_keys = g_val.keys()

            # Load optional keys
            help = g_val["help"] if ("help" in gg_keys) else ""
            type = (
                getattr(builtins, g_val["type"])
                if ("type" in gg_keys)
                else None
            )
            choices = (
                g_val["choices"] if ("choices" in gg_keys) else None
            )

            # Additional info in help string
            help += (
                "\n"
                + "<font color='#C5576B'> Group name: **{}**".format(
                    str(group)
                )
            )
            if choices:
                help += "\n" + "Choices: {}".format(str(choices))
            if type:
                help += "\n" + "Type: {}".format(g_val["type"])
            help += "</font>"

            # Create arg dict
            opt_args = {
                "load_default": json.dumps(g_val["value"]),
                "metadata": {"description": help},
                "required": False,
            }
            if choices:
                json_choices = [json.dumps(i) for i in choices]
                opt_args["metadata"]["enum"] = json_choices
                opt_args["validate"] = fields.validate.OneOf(
                    json_choices
                )
            parser[g_key] = fields.Str(**opt_args)

    return parser


def get_train_args():
    parser = OrderedDict()
    default_conf = config.CONF
    default_conf = OrderedDict(
        [
            ("general", default_conf["general"]),
            ("model", default_conf["model"]),
            ("training", default_conf["training"]),
            ("monitor", default_conf["monitor"]),
            ("dataset", default_conf["dataset"]),
            ("augmentation", default_conf["augmentation"]),
        ]
    )

    return populate_parser(parser, default_conf)


def get_predict_args():
    parser = OrderedDict()
    default_conf = config.CONF
    default_conf = OrderedDict([("testing", default_conf["testing"])])

    # Add options for modelname
    timestamp = default_conf["testing"]["timestamp"]
    timestamp_list = next(os.walk(paths.get_models_dir()))[1]
    timestamp_list = sorted(timestamp_list)
    if not timestamp_list:
        timestamp["value"] = ""
    else:
        timestamp["value"] = timestamp_list[-1]
        timestamp["choices"] = timestamp_list

    parser["image"] = fields.Field(
        required=False,
        load_default=None,
        # type="file",
        data_key="image",
        #  location="form",
        metadata={
            "description": "Select the image you want to classify.",
            "type": "file",
            "location": "form",
        },
    )

    parser["zip"] = fields.Field(
        required=False,
        load_default=None,
        #  type="file",
        data_key="zip_data",
        # location="form",
        metadata={
            "description": "Select the ZIP file containing images you want to classify.",
            "type": "file",
            "location": "form",
        },
    )

    return populate_parser(parser, default_conf)


def get_metadata(distribution_name="planktonclas"):
    """
    Function to read metadata
    """

    metadata = {
         "name": config.MODEL_METADATA.get("name"),
            "author": config.MODEL_METADATA.get("authors"),
            "author-email": config.MODEL_METADATA.get(
                "author-emails"
            ),
            "description": config.MODEL_METADATA.get("summary"),
            "license": config.MODEL_METADATA.get("license"),
            "version": config.MODEL_METADATA.get("version"),
            
         
        }

    return metadata


schema = {
    "status": fields.Str(),
    "message": fields.Str(),
    "predictions": fields.Field(),
}