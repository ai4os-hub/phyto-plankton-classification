"""
Miscellaneous functions manage data.

Date: April 2025
Author: Ignacio Heredia (Updated for Albumentations 2.0)
Email: iheredia@ifca.unican.es
Github: ignacioheredia
"""

import base64
import os
import queue
import random
import subprocess
import threading
import warnings
from multiprocessing import Pool

import albumentations as A
import cv2
import numpy as np
import requests
from tensorflow.keras.utils import Sequence, to_categorical
from tqdm import tqdm


def create_data_splits(
    splits_dir, im_dir, split_ratios=[0.7, 0.15, 0.15]
):
    train_txt_file = os.path.join(splits_dir, "train.txt")
    test_txt_file = os.path.join(splits_dir, "test.txt")
    val_txt_file = os.path.join(splits_dir, "val.txt")
    class_txt_file = os.path.join(splits_dir, "classes.txt")
    file_paths = []

    for root, _, files in tqdm(
        os.walk(im_dir), desc="Searching files"
    ):
        for file in tqdm(files, desc=f"Processing {root}"):
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, im_dir)
            file_paths.append(relative_path)

    # Get a list of folder names within the "im_dir" directory
    folder_names = next(os.walk(im_dir))[1]

    # Assign numbers based on the location of each folder in the list
    folder_numbers = {
        folder_name: index
        for index, folder_name in enumerate(folder_names)
    }

    # Count the number of files in each subfolder
    folder_counts = {folder_name: 0 for folder_name in folder_names}
    for file_path in file_paths:
        # Assuming UNIX-like path separator
        folder_name = file_path.split("/")[0]
        if folder_name in folder_counts:
            folder_counts[folder_name] += 1

    # Initialize lists to keep track of files added to each split for each
    # folder
    train_files_by_folder = {
        folder_name: [] for folder_name in folder_names
    }
    test_files_by_folder = {
        folder_name: [] for folder_name in folder_names
    }
    val_files_by_folder = {
        folder_name: [] for folder_name in folder_names
    }

    # Split the files into training, testing, and validation sets
    for folder_name in folder_names:
        folder_files = [
            file_path
            for file_path in file_paths
            if file_path.startswith(folder_name + "/")
        ]
        random.shuffle(folder_files)
        num_files = len(folder_files)
        train_cutoff = int(num_files * split_ratios[0])
        test_cutoff = train_cutoff + int(num_files * split_ratios[1])

        train_files_by_folder[folder_name] = folder_files[
            :train_cutoff
        ]
        test_files_by_folder[folder_name] = folder_files[
            train_cutoff:test_cutoff
        ]
        val_files_by_folder[folder_name] = folder_files[test_cutoff:]

    # Combine files from each folder into overall train, test, and validation
    # sets
    train_files = [
        file
        for folder_files in train_files_by_folder.values()
        for file in folder_files
    ]
    test_files = [
        file
        for folder_files in test_files_by_folder.values()
        for file in folder_files
    ]
    val_files = [
        file
        for folder_files in val_files_by_folder.values()
        for file in folder_files
    ]

    # Write the file paths to text files for training, testing, and validation
    write_text_file(train_files, train_txt_file, folder_numbers)
    write_text_file(test_files, test_txt_file, folder_numbers)
    write_text_file(val_files, val_txt_file, folder_numbers)

    # Write the class names to a text file
    with open(class_txt_file, "w") as f_class:
        for label in tqdm(
            folder_numbers, desc="Writing classes file"
        ):
            f_class.write(str(label) + "\n")


def write_text_file(file_list, file_path, folder_numbers):
    with open(file_path, "w") as f:
        for file in tqdm(file_list, desc=f"Writing {file_path}"):
            file = file.replace(
                "\\", "/"
            )  # Assuming UNIX-like path separator
            f.write(
                file
                + " "
                + str(folder_numbers[file.split("/")[0]])
                + "\n"
            )


def load_data_splits(splits_dir, im_dir, split_name="train"):
    """
    Load the data arrays from the [train/val/test].txt files.
    Lines of txt files have the following format:
    'relative_path_to_image' 'image_label_number'

    Parameters
    ----------
    im_dir : str
        Absolute path to the image folder.
    split_name : str
        Name of the data split to load

    Returns
    -------
    X : Numpy array of strs
        First colunm: Contains 'absolute_path_to_file' to images.
    y : Numpy array of int32
        Image label number
    """
    if "{}.txt".format(split_name) not in os.listdir(splits_dir):
        raise ValueError(
            "Invalid value for the split_name parameter: there is no `{}.txt` file in the `{}` "
            "directory.".format(split_name, splits_dir)
        )

    # Loading splits
    print("Loading {} data...".format(split_name))
    split = np.genfromtxt(
        os.path.join(splits_dir, "{}.txt".format(split_name)),
        dtype="str",
        delimiter=" ",
    )
    X = np.array([os.path.join(im_dir, i) for i in split[:, 0]])

    # TODO Check this part of the code
    if len(split.shape) == 2:
        y = split[:, 1].astype(np.int32)
    else:  # maybe test file has not labels
        y = None

    return X, y


def mount_nextcloud(frompath, topath):
    """
    Mount a NextCloud folder in your local machine or viceversa.
    """
    command = ["rclone", "copy", frompath, topath]
    result = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    output, error = result.communicate()
    if error:
        warnings.warn(
            "Error while mounting NextCloud: {}".format(error)
        )
    return output, error


def load_class_names(splits_dir):
    """
    Load list of class names

    Returns
    -------
    Numpy array of shape (N) containing strs with class names
    """
    print("Loading class names...")
    class_names = np.genfromtxt(
        os.path.join(splits_dir, "classes.txt"),
        dtype="str",
        delimiter="/n",
    )
    return class_names


def load_aphia_ids(splits_dir):
    """
    Load list of aphia_ids

    Returns
    -------
    Numpy array of shape (N) containing strs with class names
    """
    print("Loading aphia_ids...")
    try:
        aphia_ids = np.genfromtxt(
            os.path.join(splits_dir, "aphia_ids.txt"),
            dtype="str",
            delimiter="/n",
        )
    except BaseException:
        aphia_ids = None

    return aphia_ids


def load_class_info(splits_dir):
    """
    Load list of class names

    Returns
    -------
    Numpy array of shape (N) containing strs with class names
    """
    print("Loading class info...")
    class_info = np.genfromtxt(
        os.path.join(splits_dir, "info.txt"),
        dtype="str",
        delimiter="/n",
    )
    return class_info


def load_image(filename, filemode="local"):
    """
    Function to load a local image path (or an url) into a numpy array.

    Parameters
    ----------
    filename : str
        Path or url to the image
    filemode : {'local','url'}
        - 'local': filename is absolute path in local disk.
        - 'url': filename is internet url.

    Returns
    -------
    A numpy array
    """
    if filemode == "local":
        image = cv2.imread(filename, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(
                "The local path does not exist or does not correspond to an image: \n {}".format(
                    filename
                )
            )

    elif filemode == "url":
        try:
            if filename.startswith(
                "data:image"
            ):  # base64 encoded string
                data = base64.b64decode(filename.split(";base64,")[1])
            else:  # normal url
                data = requests.get(filename, timeout=10).content
            data = np.frombuffer(data, np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if image is None:
                raise Exception
        except BaseException:
            raise ValueError(
                "Incorrect url path: \n {}".format(filename)
            )

    else:
        raise ValueError("Invalid value for filemode.")

    image = cv2.cvtColor(
        image, cv2.COLOR_BGR2RGB
    )  # change from default BGR OpenCV format to Python's RGB format
    return image


def preprocess_batch(
    batch, mean_RGB, std_RGB, mode="tf", channels_first=False
):
    """
    Standardize batch to feed the net. Adapted from [1] to take replace the default imagenet mean and std.
    [1] https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py

    Parameters
    ----------
    batch : list of numpy arrays
    mean_RGB, std_RGB : list of floats, len=3
        Mean/std RGB values for your dataset.
    channels_first : bool
        Use batch of shape (N, C, H, W) instead of (N, H, W, C)

    Returns
    -------
    Numpy array
    """

    mean_RGB, std_RGB = np.array(mean_RGB), np.array(std_RGB)
    batch = (
        np.array(batch) - mean_RGB[None, None, None, :]
    )  # mean centering

    if mode == "caffe":
        batch = batch[:, :, :, ::-1]  # switch from RGB to BGR
    if mode == "tf":
        batch /= 127.5  # scaling between [1, -1]
    if mode == "torch":
        batch /= std_RGB
    if channels_first:
        batch = batch.transpose(0, 3, 1, 2)  # shape(N, 3, 224, 224)
    return batch.astype(np.float32)


def augment(im, params=None):
    """
    Perform data augmentation on some image using the albumentations package.

    Parameters
    ----------
    im : Numpy array
    params : dict or None
        Contains the data augmentation parameters
        Mandatory keys:
        - h_flip ([0,1] float): probability of performing an horizontal left-right mirroring.
        - v_flip ([0,1] float): probability of performing an vertical up-down mirroring.
        - rot ([0,1] float):  probability of performing a rotation to the image.
        - rot_lim (int):  max degrees of rotation.
        - stretch ([0,1] float):  probability of randomly stretching an image.
        - crop ([0,1] float): randomly take an image crop.
        - zoom ([0,1] float): random zoom applied to crop_size.
            --> Therefore the effective crop size at each iteration will be a
                random number between 1 and crop*(1-zoom). For example:
                  * crop=1, zoom=0: no crop of the image
                  * crop=1, zoom=0.1: random crop of random size between 100% image and 90% of the image
                  * crop=0.9, zoom=0.1: random crop of random size between 90% image and 80% of the image
                  * crop=0.9, zoom=0: random crop of always 90% of the image
                  Image size refers to the size of the shortest side.
        - blur ([0,1] float):  probability of randomly blurring an image.
        - pixel_noise ([0,1] float):  probability of randomly adding pixel noise to an image.
        - pixel_sat ([0,1] float):  probability of randomly using HueSaturationValue in the image.
        - cutout ([0,1] float):  probability of using cutout in the image.

    Returns
    -------
    Numpy array
    """

    # 1) Crop the image
    effective_zoom = np.random.rand() * params["zoom"]
    crop = params["crop"] - effective_zoom

    ly, lx, channels = im.shape
    crop_size = int(crop * min([ly, lx]))
    rand_x = np.random.randint(low=0, high=lx - crop_size + 1)
    rand_y = np.random.randint(low=0, high=ly - crop_size + 1)

    crop_transform = A.Crop(
        x_min=rand_x,
        y_min=rand_y,
        x_max=rand_x + crop_size,
        y_max=rand_y + crop_size,
    )

    im = crop_transform(image=im)["image"]

    # 2) Now add the transformations for augmenting the image pixels
    transform_list = []

    # Add random stretching
    if params["stretch"]:
        transform_list.append(
            A.PerspectiveTransform(
                scale=(0.05, 0.1), p=params["stretch"]
            )
        )

    # Add random rotation
    if params["rot"]:
        transform_list.append(
            A.Rotate(limit=params["rot_lim"], p=params["rot"])
        )

    # Add horizontal flip
    if params["h_flip"]:
        transform_list.append(A.HorizontalFlip(p=params["h_flip"]))

    # Add vertical flip
    if params["v_flip"]:
        transform_list.append(A.VerticalFlip(p=params["v_flip"]))

    # Add some blur to the image
    if params["blur"]:
        transform_list.append(
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=7, p=1.0),
                    A.MedianBlur(blur_limit=7, p=1.0),
                    A.Blur(blur_limit=7, p=1.0),
                ],
                p=params["blur"],
            )
        )

    # Add pixel noise
    if params["pixel_noise"]:
        transform_list.append(
            A.OneOf(
                [
                    A.CLAHE(clip_limit=2, p=1.0),
                    A.Sharpen(p=1.0),
                    A.Emboss(p=1.0),
                    A.RandomBrightnessContrast(
                        contrast_limit=0, p=1.0
                    ),
                    A.RandomBrightnessContrast(
                        brightness_limit=0, p=1.0
                    ),
                    A.RGBShift(p=1.0),
                    A.RandomGamma(p=1.0),
                ],
                p=params["pixel_noise"],
            )
        )

    # Add pixel saturation
    if params["pixel_sat"]:
        transform_list.append(
            A.HueSaturationValue(p=params["pixel_sat"])
        )

    # Remove randomly remove some regions from the image
    if params["cutout"]:
        ly, lx, channels = im.shape
        scale_low, scale_high = (
            0.05,
            0.25,
        )  # min and max size of the squares wrt the full image
        scale = np.random.uniform(scale_low, scale_high)
        transform_list.append(
            A.CoarseDropout(
                num_holes_range=(8, 8), 
                hole_height_range=(int(scale * ly), int(scale * ly)), 
                hole_width_range=(int(scale * lx), int(scale * lx)),  
                fill=0, 
                p=params["cutout"],
            )
        )

    # Compose all image transformations and augment the image
    augmentation_fn = A.Compose(transform_list)
    im = augmentation_fn(image=im)["image"]

    return im


def resize_im(im, height, width):
    resize_fn = A.Resize(height=height, width=width)
    return resize_fn(image=im)["image"]


def data_generator(
    inputs,
    targets,
    batch_size,
    mean_RGB,
    std_RGB,
    preprocess_mode,
    aug_params,
    num_classes,
    im_size=224,
    shuffle=True,
):
    """
    Generator to feed Keras fit function

    Parameters
    ----------
    inputs : Numpy array, shape (N, H, W, C)
    targets : Numpy array, shape (N)
    batch_size : int
    shuffle : bool
    aug_params : dict
    im_size : int
        Final image size to feed the net's input (eg. 224 for Resnet).

    Returns
    -------
    Generator of inputs and labels
    """

    # Create list of indices
    idxs = np.arange(len(inputs))
    if shuffle:
        np.random.shuffle(idxs)

    for start_idx in range(
        0, len(inputs) - batch_size + 1, batch_size
    ):
        excerpt = idxs[start_idx : start_idx + batch_size]
        batch_X = []
        for i in excerpt:
            im = load_image(inputs[i], filemode="local")
            im = augment(im, params=aug_params)
            im = resize_im(im, height=im_size, width=im_size)
            batch_X.append(im)  # shape (N, 224, 224, 3)
        batch_X = preprocess_batch(
            batch=batch_X,
            mean_RGB=mean_RGB,
            std_RGB=std_RGB,
            mode=preprocess_mode,
        )
        batch_y = to_categorical(
            targets[excerpt], num_classes=num_classes
        )

        yield batch_X, batch_y


def buffered_generator(source_gen, buffer_size=10):
    """
    Generator that runs a slow source generator in a separate thread. Beware of the GIL!
    Author: Benanne (github-kaggle/benanne/ndsb)

    Parameters
    ----------
    source_gen : generator
    buffer_size: the maximal number of items to pre-generate (length of the buffer)

    Returns
    -------
    Buffered generator
    """
    if buffer_size < 2:
        raise RuntimeError("Minimal buffer size is 2!")

    buffer = queue.Queue(maxsize=buffer_size - 1)
    # the effective buffer size is one less, because the generation process
    # will generate one extra element and block until there is room in the
    # buffer.

    def _buffered_generation_thread(source_gen, buffer):
        for data in source_gen:
            buffer.put(data, block=True)
        buffer.put(None)  # sentinel: signal the end of the iterator

    thread = threading.Thread(
        target=_buffered_generation_thread, args=(source_gen, buffer)
    )
    thread.daemon = True
    thread.start()

    for data in iter(buffer.get, None):
        yield data


class data_sequence(Sequence):
    """
    Instance of a Keras Sequence that is safer to use with multiprocessing than a standard generator.
    Check https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    """

    def __init__(
        self,
        inputs,
        targets,
        batch_size,
        mean_RGB,
        std_RGB,
        preprocess_mode,
        aug_params,
        num_classes,
        im_size=224,
        shuffle=True,
    ):
        """
        Parameters are the same as in the data_generator function
        """

        self.inputs = inputs
        self.targets = targets
        self.batch_size = batch_size
        self.mean_RGB = mean_RGB
        self.std_RGB = std_RGB
        self.preprocess_mode = preprocess_mode
        self.aug_params = aug_params
        self.num_classes = num_classes
        self.im_size = im_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.inputs) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_idxs = self.indexes[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]
        batch_X = []
        tmp_idxs = []
        for i in batch_idxs:
            try:
                im = load_image(self.inputs[i])
            except Exception as e:
                print(e)
                continue
            if self.aug_params:
                im = augment(im, params=self.aug_params)
            im = resize_im(
                im, height=self.im_size, width=self.im_size
            )
            batch_X.append(im)  # shape (N, 224, 224, 3)
            tmp_idxs.append(i)
        batch_X = preprocess_batch(
            batch=batch_X,
            mean_RGB=self.mean_RGB,
            std_RGB=self.std_RGB,
            mode=self.preprocess_mode,
        )
        batch_y = to_categorical(
            self.targets[tmp_idxs], num_classes=self.num_classes
        )
        return batch_X, batch_y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.inputs))
        if self.shuffle:
            np.random.shuffle(self.indexes)


def standard_tencrop_batch(im, crop_prop=0.9):
    """
    Returns an ordered ten crop batch of images from an original image (corners, center + mirrors).

    Parameters
    ----------
    im : numpy array, type np.uint8
    crop_prop: float, [0, 1]
        Size of the crop with respect to the whole image

    Returns
    -------
    List of 10 numpy arrays
    """
    batch = []

    min_side = np.amin(im.shape[:2])
    # resize to shorter border
    im = resize_im(im, height=min_side, width=min_side)
    h, w = min_side, min_side  # height, width (square)
    crop_size = int(crop_prop * min_side)

    # Crops
    c1 = A.Crop(x_min=0, y_min=0, x_max=crop_size, y_max=crop_size)(
        image=im
    )[
        "image"
    ]  # top-left

    c2 = A.Crop(
        x_min=0, y_min=h - crop_size, x_max=crop_size, y_max=h
    )(image=im)[
        "image"
    ]  # bottom-left

    c3 = A.Crop(
        x_min=w - crop_size, y_min=0, x_max=w, y_max=crop_size
    )(image=im)[
        "image"
    ]  # top-right

    c4 = A.Crop(
        x_min=w - crop_size, y_min=h - crop_size, x_max=w, y_max=h
    )(image=im)[
        "image"
    ]  # bottom-right

    c5 = A.Crop(
        x_min=np.round((w - crop_size) / 2).astype(int),
        y_min=np.round((h - crop_size) / 2).astype(int),
        x_max=np.round((w + crop_size) / 2).astype(int),
        y_max=np.round((h + crop_size) / 2).astype(int),
    )(image=im)[
        "image"
    ]  # center

    # Save crop and its mirror
    lr_aug = A.HorizontalFlip(p=1)
    for image in [c1, c2, c3, c4, c5]:
        batch.append(image)
        batch.append(lr_aug(image=image)["image"])

    return batch


class k_crop_data_sequence(Sequence):
    """
    Data sequence generator for test time to feed to predict_generator.
    Each batch delivered is composed by multiple crops (default=10) of the same image.
    """

    def __init__(
        self,
        inputs,
        mean_RGB,
        std_RGB,
        preprocess_mode,
        aug_params,
        crop_number=10,
        crop_mode="random",
        filemode="local",
        im_size=224,
    ):
        """
        Parameters are the same as in the data_generator function except for:

        Parameters
        ----------
        crop_number : int
            Number of crops of each image to take.
        mode :str, {'random', 'standard'}
            If 'random' data augmentation is performed randomly.
            If 'standard' we take the standard 10 crops (corners +center + mirrors)
        filemode : {'local','url'}
            - 'local': filename is absolute path in local disk.
            - 'url': filename is internet url.
        """
        self.inputs = inputs
        self.mean_RGB = mean_RGB
        self.std_RGB = std_RGB
        self.preprocess_mode = preprocess_mode
        self.aug_params = aug_params
        self.crop_number = crop_number
        self.crop_mode = crop_mode
        self.filemode = filemode
        self.im_size = im_size

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        batch_X = []
        im = load_image(self.inputs[idx], filemode=self.filemode)

        if self.crop_mode == "random":
            for _ in range(self.crop_number):
                if self.aug_params:
                    im_aug = augment(im, params=self.aug_params)
                else:
                    im_aug = np.copy(im)
                im_aug = resize_im(
                    im_aug, height=self.im_size, width=self.im_size
                )
                batch_X.append(im_aug)  # shape (N, 224, 224, 3)

        if self.crop_mode == "standard":
            batch_X = standard_tencrop_batch(im)

        batch_X = preprocess_batch(
            batch=batch_X,
            mean_RGB=self.mean_RGB,
            std_RGB=self.std_RGB,
            mode=self.preprocess_mode,
        )
        return batch_X


def im_stats(filename):
    """
    Helper for function compute_meanRGB
    """
    im = load_image(filename, filemode="local")
    mean = np.mean(im, axis=(0, 1))
    std = np.std(im, axis=(0, 1))
    return mean.tolist(), std.tolist()


def compute_meanRGB(im_list, verbose=False, workers=4):
    """
    Returns the mean and std RGB values for the whole dataset.
    For example in the plantnet dataset we have:
        mean_RGB = np.array([ 107.59348955,  112.1047813 ,   80.9982362 ])
        std_RGB = np.array([ 52.78326119,  50.56163087,  50.86486131])

    Parameters
    ----------
    im_list : array of strings
        Array where the first column is image_path (or image_url). Shape (N,).
    verbose : bool
        Show progress bar
    workers: int
        Numbers of parallel workers to perform the computation with.

    References
    ----------
    https://stackoverflow.com/questions/41920124/multiprocessing-use-tqdm-to-display-a-progress-bar
    """

    with Pool(workers) as p:
        r = list(
            tqdm(
                p.imap(im_stats, im_list),
                total=len(im_list),
                disable=verbose,
            )
        )

    r = np.asarray(r)
    mean, std = r[:, 0], r[:, 1]
    mean, std = np.mean(mean, axis=0), np.mean(std, axis=0)

    print("Mean RGB pixel: {}".format(mean.tolist()))
    print("Standard deviation of RGB pixel: {}".format(std.tolist()))

    return mean.tolist(), std.tolist()


def compute_classweights(labels, max_dim=None, mode="balanced"):
    """
    Compute the class weights  for a set of labels to account for label imbalance.

    Parameters
    ----------
    labels : numpy array, type (ints), shape (N)
    max_dim : int
        Maximum number of classes. Default is the max value in labels.
    mode : str, {'balanced', 'log'}

    Returns
    -------
    Numpy array, type (float32), shape (N)
    """
    if mode is None:
        return None

    try:
        weights = np.bincount(labels)
    except TypeError:
        _, weights = np.unique(labels, return_counts=True)

    weights = np.sum(weights) / weights

    # Fill the count if some high number labels are not present in the sample
    if max_dim is not None:
        diff = max_dim - len(weights)
        if diff != 0:
            weights = np.pad(
                weights,
                pad_width=(0, diff),
                mode="constant",
                constant_values=0,
            )

    # Transform according to different modes
    if mode == "balanced":
        pass
    elif mode == "log":
        # do not use --> produces numerical instabilities at inference when
        # transferring weights trained on GPU to CPU
        weights = np.log(weights)  # + 1
    else:
        raise ValueError(
            '{} is not a valid option for parameter "mode"'.format(
                mode
            )
        )

    return weights.astype(np.float32)


def json_friendly(d):
    """
    Return a json friendly dictionary (mainly remove numpy data types)
    """
    new_d = {}
    for k, v in d.items():
        if isinstance(v, (np.float32, np.float64)):
            v = float(v)
        elif isinstance(v, (np.ndarray, list)):
            if isinstance(v[0], (np.float32, np.float64)):
                v = np.array(v).astype(float).tolist()
            else:
                v = np.array(v).tolist()
        new_d[k] = v
    return new_d
