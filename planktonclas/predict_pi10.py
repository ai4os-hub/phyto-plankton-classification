import os
import glob
import json
import zipfile
import shutil

from tqdm import tqdm
import numpy as np
from tensorflow.keras.models import load_model
from planktonclas import paths, config, plot_utils, utils
from planktonclas.test_utils import predict
from planktonclas.data_utils import load_class_names

# === USER PARAMETERS ===
TIMESTAMP = 'PI10_model'
MODEL_NAME = 'final_model.h5'
TOP_K = 5
PARENT_DIR = rf'\\fs\SHARED\transfert\Wout_er_zijn_geen_vaste_stoelen'

# === SETUP ===
paths.timestamp = TIMESTAMP
class_names = load_class_names(splits_dir=paths.get_ts_splits_dir())

model = load_model(os.path.join(paths.get_checkpoints_dir(), MODEL_NAME),
                   custom_objects=utils.get_custom_objects())

# === LOOP OVER ZIP FILES ===
zip_files = glob.glob(os.path.join(PARENT_DIR, '*.zip'))

for zip_path in tqdm(zip_files, desc="Processing ZIP files"):
    zip_filename = os.path.basename(zip_path)
    base_name = os.path.splitext(zip_filename)[0]
    json_filename = f"{base_name}_predictions.json"
    json_path = os.path.join(paths.get_predictions_dir(), json_filename)

    if os.path.exists(json_path):
        tqdm.write(f"Skipping {zip_filename} (already predicted).")
        continue

    unzip_dir = os.path.join(PARENT_DIR, base_name)
    tqdm.write(f"Unzipping {zip_filename} to {unzip_dir}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(unzip_dir)

    FILEPATHS = glob.glob(os.path.join(unzip_dir, '**', '*.tif'), recursive=True)
    tqdm.write(f"Found {len(FILEPATHS)} .tif files in {base_name}")

    if not FILEPATHS:
        tqdm.write(f"No .tif files found in {base_name}, skipping.")
        shutil.rmtree(unzip_dir)
        continue

    pred_lab, pred_prob = predict(model, FILEPATHS, conf, top_K=TOP_K, filemode='local')

    results = []
    for i in tqdm(range(len(FILEPATHS)), desc=f"Processing {base_name}", leave=False):
        full_path = FILEPATHS[i]
        relative_path = os.path.relpath(full_path, PARENT_DIR)
        labels = [class_names[pred_lab[i, j]] for j in range(TOP_K)]
        probs = [float(pred_prob[i, j]) for j in range(TOP_K)]
        results.append({
            "filepath": relative_path,
            "top5_labels": labels,
            "top5_probs": probs,
        })

    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    tqdm.write(f"Saved predictions to {json_path}")

    shutil.rmtree(unzip_dir)
    tqdm.write(f"Deleted unzipped directory {unzip_dir}\n")
