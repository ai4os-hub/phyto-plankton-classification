import os
import glob
import json
import tarfile
import shutil
import pandas as pd
import numpy as np
from tqdm import tqdm
from skimage.io import imread
from skimage import measure, morphology
from skimage.color import rgb2gray
from tensorflow.keras.models import load_model

from planktonclas import paths, config, plot_utils, utils
from planktonclas.test_utils import predict
from planktonclas.data_utils import load_class_names

# === USER PARAMETERS ===
TIMESTAMP = 'PI10_model'
MODEL_NAME = 'final_model.h5'
TOP_K = 5
PARENT_DIR = r'\\qarchive\data_sensors\plankton-imager-10\script_test'

# === SETUP ===
paths.timestamp = TIMESTAMP
class_names = load_class_names(splits_dir=paths.get_ts_splits_dir())
model = load_model(os.path.join(paths.get_checkpoints_dir(), MODEL_NAME),
                   custom_objects=utils.get_custom_objects())
conf_path = os.path.join(paths.get_conf_dir(), 'conf.json')
with open(conf_path) as f:
    conf = json.load(f)

# === REGION FUNCTIONS ===
def getImageRegionList(filename):
    image = imread(filename)
    if image.ndim == 3:
        image = rgb2gray(image)
    image_threshold = np.where(image > np.mean(image), 0., 1.0)
    image_dilated = morphology.dilation(image_threshold, np.ones((4, 4)))
    label_list = measure.label(image_dilated)
    label_list = (image_threshold * label_list).astype(int)
    return measure.regionprops(label_list)

def getMaxAreaDict(filename):
    regions = getImageRegionList(filename)
    if not regions:
        return {'object_additional_area': 0}

    r = max(regions, key=lambda x: x.area)
    return {
        'object_additional_diameter_equivalent': r.equivalent_diameter,
        'object_additional_length_minor_axis': r.minor_axis_length,
        'object_additional_length_major_axis': r.major_axis_length,
        'object_additional_eccentricity': r.eccentricity,
        'object_additional_area': r.area,
        'object_additional_perimeter': r.perimeter,
        'object_additional_orientation': r.orientation,
        'object_additional_area_convex': r.convex_area,
        'object_additional_area_filled': r.filled_area,
        'object_additional_box_min_row': r.bbox[0],
        'object_additional_box_max_row': r.bbox[2],
        'object_additional_box_min_col': r.bbox[1],
        'object_additional_box_max_col': r.bbox[3],
        'object_additional_ratio_extent': r.extent,
        'object_additional_ratio_solidity': r.solidity,
        'object_additional_inertia_tensor_eigenvalue1': r.inertia_tensor_eigvals[0],
        'object_additional_inertia_tensor_eigenvalue2': r.inertia_tensor_eigvals[1],
        'object_additional_moments_hu1': r.moments_hu[0],
        'object_additional_moments_hu2': r.moments_hu[1],
        'object_additional_moments_hu3': r.moments_hu[2],
        'object_additional_moments_hu4': r.moments_hu[3],
        'object_additional_moments_hu5': r.moments_hu[4],
        'object_additional_moments_hu6': r.moments_hu[5],
        'object_additional_moments_hu7': r.moments_hu[6],
        'object_additional_euler_number': r.euler_number,
        'object_additional_countcoords': len(r.coords)
    }

# === MAIN LOOP ===
tar_files = glob.glob(os.path.join(PARENT_DIR, '*.tar*'))

for tar_path in tqdm(tar_files, desc="Processing TAR files"):
    tar_filename = os.path.basename(tar_path)
    base_name = os.path.splitext(tar_filename)[0]
    json_filename = f"{base_name}_predictions.json"
    csv_filename = f"{base_name}_properties.csv"
    json_path = os.path.join(PARENT_DIR, json_filename)
    csv_path = os.path.join(PARENT_DIR, csv_filename)

    if os.path.exists(json_path) and os.path.exists(csv_path):
        tqdm.write(f"Skipping {tar_filename} (already processed).")
        continue

    untar_dir = os.path.join(PARENT_DIR, base_name)
    if not os.path.exists(untar_dir):
        tqdm.write(f"Extracting {tar_filename}")
        with tarfile.open(tar_path, 'r:*') as tar_ref:
            tar_ref.extractall(untar_dir)
    else:
        tqdm.write(f"Folder already exists for {tar_filename}, skipping extraction.")
        
    FILEPATHS = glob.glob(os.path.join(untar_dir, '**', '*.tif'), recursive=True)
    tqdm.write(f"Found {len(FILEPATHS)} .tif files in {base_name}")

    if not FILEPATHS:
        tqdm.write(f"No .tif files found in {base_name}, skipping.")
        shutil.rmtree(untar_dir)
        continue

    # Predict
    pred_lab, pred_prob = predict(model, FILEPATHS, conf, top_K=TOP_K, filemode='local')

    results_json = []
    results_csv = []

    for i in tqdm(range(len(FILEPATHS)), desc=f"Processing {base_name}", leave=False):
        path = FILEPATHS[i]
        relative_path = os.path.relpath(path, PARENT_DIR)
        labels = [class_names[pred_lab[i, j]] for j in range(TOP_K)]
        probs = [float(pred_prob[i, j]) for j in range(TOP_K)]

        results_json.append({
            "filepath": relative_path,
            "top5_labels": labels,
            "top5_probs": probs
        })

        region_props = getMaxAreaDict(path)
        region_props["filepath"] = relative_path
        results_csv.append(region_props)

    # Save results
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    tqdm.write(f"Saved predictions to {json_path}")

    df = pd.DataFrame(results_csv)
    df.to_csv(csv_path, index=False)
    tqdm.write(f"Saved image properties to {csv_path}")

    shutil.rmtree(untar_dir)
    tqdm.write(f"Cleaned up {untar_dir}\n")
