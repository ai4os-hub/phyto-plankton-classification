# === LIBRARIES ===
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import shutil
from pathlib import Path
import tarfile
import pandas as pd
import subprocess
import time
import random
import json
import numpy as np
from tqdm import tqdm
from skimage.io import imread
from skimage.color import rgb2gray
from skimage import measure, morphology
from tensorflow.keras.models import load_model
from planktonclas import paths as plk_paths, utils
from planktonclas.test_utils import predict
from planktonclas.data_utils import load_class_names
import datetime
import threading
import time
import csv

last_summary_date = None  # will track the last date email was sent
last_afternoon_summary_sent_day = None  # for 15:00 status update

#=== STORE LOGS ===
log_file_path = "/path/PI10/logs/processing_times.csv"

#=== MAILING ===
from dotenv import load_dotenv
load_dotenv(dotenv_path="/path/PI10/.env")
EMAIL_SETTINGS = {
    'smtp_server': os.getenv('SMTP_SERVER'),
    'smtp_port': int(os.getenv('SMTP_PORT')),
    'sender_email': os.getenv('SENDER_EMAIL'),
    'sender_password': os.getenv('SENDER_PASSWORD'),
    'recipients': [email.strip() for email in os.getenv('EMAIL_RECIPIENTS').split(',')]
}
daily_tar_reports = []  # Stores dicts with tar_name, quarantined, quarantine_reason, quarantine_path, status_log

import smtplib
from email.mime.text import MIMEText

def email_scheduler():
    global last_afternoon_summary_sent_day
    while True:
        now = datetime.datetime.now()
        current_date = now.date()

        # Send summary at exactly 15:00 once per day
        if (now.hour == 11 and now.minute == 0
                and last_afternoon_summary_sent_day != current_date):
            send_daily_summary_email(now.strftime('%Y-%m-%d'), daily_tar_reports)
            last_afternoon_summary_sent_day = current_date

        time.sleep(60)  # check every minute

def send_daily_summary_email(summary_date, report_data):
    import math
    global source_dir

    subject = f"[PI10] Daily Summary - {summary_date}"
    all_tar_files = list(source_dir.glob("*.tar"))
    tar_stems = {tar.stem for tar in all_tar_files}

    # === Count current files ===
    preview_root = Path("/path/not_processed/previews")
    current_counts = {
        "tar": len(all_tar_files),
        "gpstag": len(list(source_dir.glob("*_gpstag.csv"))),
        "predictions": len(list(source_dir.glob("*_predictions_relative.json"))),
        "image_props": len(list(source_dir.glob("*_image_properties.csv"))),
        "topspecies": len(list(source_dir.glob("*_topspecies.csv"))),
        "hitsmisses": len(list(source_dir.glob("*_hitsmisses.txt"))),
        "backgrounds": len(list(source_dir.glob("*_Background.tif"))),
        "previews": len(list(preview_root.glob("*_preview"))),
    }

    # === Load yesterday's counts from file ===
    delta_file = Path("daily_tar_count.json")
    yesterday_counts = {k: 0 for k in current_counts}
    if delta_file.exists():
        try:
            with open(delta_file, 'r') as f:
                yesterday_counts.update(json.load(f))
        except Exception as e:
            print(f"⚠️ Could not read yesterday's count: {e}")

    # === Calculate deltas ===
    deltas = {k: current_counts[k] - yesterday_counts.get(k, 0) for k in current_counts}
    tar_delta = deltas["tar"]

    # === Save today’s counts for tomorrow ===
    try:
        with open(delta_file, 'w') as f:
            json.dump(current_counts, f)
    except Exception as e:
        print(f"⚠️ Could not write today's count: {e}")

    # === Calculate to-do (missing output per TAR) ===
    required_outputs = {
        "_gpstag.csv": ("GPS data", 0.5),                   # 30 mins per file
        "_hitsmisses.txt": ("Hits/Misses", 10 / 3600),      # 10 sec per file
        "_Background.tif": ("Background.tif", 10 / 3600),   # 10 sec per file
        "_predictions_relative.json": ("Predictions (JSON)", 3),  # 3 hours per file
        "_image_properties.csv": ("Image Properties (CSV)", 0.5), # 30 mins per file
        "_topspecies.csv": ("Top Species CSV", 2 / 60),     # 2 mins per file
    }

    todo_counts = {}
    raw_time_estimations = {}
    formatted_time_estimations = {}

    for suffix, (label, per_file_hours) in required_outputs.items():
        count = sum(not (source_dir / f"{stem}{suffix}").exists() for stem in tar_stems)
        todo_counts[label] = count
        total_hours = count * per_file_hours
        raw_time_estimations[label] = total_hours

        # Format time
        if total_hours >= 24:
            formatted_time_estimations[label] = f"{round(total_hours / 24, 2)} day(s)"
        else:
            h = int(total_hours)
            m = round((total_hours - h) * 60)
            formatted_time_estimations[label] = f"{h}h {m}min"

    total_time = sum(raw_time_estimations.values())
    if total_time >= 24:
        total_time_str = f"{round(total_time / 24, 2)} day(s)"
    else:
        th = int(total_time)
        tm = round((total_time - th) * 60)
        total_time_str = f"{th}h {tm}min"

    # === Build email body ===
    body_lines = []
    body_lines.append(f"🆕 **{abs(tar_delta)} TARs {'extra' if tar_delta >= 0 else 'less'} compared to yesterday**")
    body_lines.append("=" * 60)
    body_lines.append("")

    body_lines.append(f"**Summary for {summary_date}**")
    body_lines.append(f"TARs entirely processed today: {len(report_data)}")
    body_lines.append("")

    body_lines.append("**Folder Totals vs Yesterday:**")
    for key, label in [
        ("tar", "TAR files"),
        ("gpstag", "GPS data (gpstag.csv)"),
        ("predictions", "Predictions (JSON)"),
        ("image_props", "Image Properties (CSV)"),
        ("topspecies", "Top Species CSV"),
        ("hitsmisses", "Hits/Misses TXT"),
        ("backgrounds", "Background.tif"),
        ("previews", "Preview folders"),
    ]:
        delta = deltas[key]
        sign = "+" if delta >= 0 else "-"
        body_lines.append(f"- {label}: {current_counts[key]} ({sign}{abs(delta)})")

    body_lines.append("")
    body_lines.append("**To-do by output module (missing files):**")
    for label in todo_counts:
        count = todo_counts[label]
        formatted_time = formatted_time_estimations[label]
        body_lines.append(f"- {label}: {count} files missing ({formatted_time})")

    body_lines.append("")
    body_lines.append(f"**Total estimated processing time left:** {total_time_str}")

    # === Send email ===
    msg = MIMEText("\n".join(body_lines))
    msg['Subject'] = subject
    msg['From'] = EMAIL_SETTINGS['sender_email']
    msg['To'] = ", ".join(EMAIL_SETTINGS['recipients'])

    try:
        with smtplib.SMTP(EMAIL_SETTINGS['smtp_server'], EMAIL_SETTINGS['smtp_port']) as server:
            server.starttls()
            server.login(EMAIL_SETTINGS['sender_email'], EMAIL_SETTINGS['sender_password'])
            server.sendmail(msg['From'], EMAIL_SETTINGS['recipients'], msg.as_string())
        print(f"📧 Daily summary email sent for {summary_date}")
    except Exception as e:
        print(f"❌ Failed to send daily summary email: {e}")



#LOG TIME OF EACH STEP
def init_log_file():
    """Initialize the CSV log file with headers."""
    headers = [
        "TAR Name",
        "Copy TAR to working directory",
        "Untar",
        "Create preview images",
        "Early preview classification",
        "Copy Background.tif",
        "Extract hitsmisses.txt",
        "Extract and save EXIF metadata",
        "Classification and morphology extraction",
        "Generate top species CSV",
        "Total pipeline time (h)",
        "Number of images in TAR",
        "Model used",
        "Logged at"
    ]


    if not Path(log_file_path).exists():
        with open(log_file_path, "w", newline="") as log_file:
            writer = csv.writer(log_file)
            writer.writerow(headers)
        print("⚙ Initialized new processing time log file.")
    else:
        print("⚙ Log file already exists, appending new entries.")

# for the logfiles; these are the headers
step_names = [
    "Copy TAR to working directory",
    "Untar",
    "Create preview images",
    "Early preview classification",
    "Copy Background.tif",
    "Extract hitsmisses.txt",
    "Extract and save EXIF metadata",
    "Classification and morphology extraction",
    "Generate top species CSV"
]

def log_time_to_file(tar_name, times_dict, num_images):
    total_hours = sum(times_dict.values()) / 3600.0
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    row = [tar_name] + [times_dict.get(name, 0.0) for name in step_names] \
          + [total_hours, num_images, TIMESTAMP, timestamp]

    with open(log_file_path, "a", newline="") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(row)

    print(f"✅ Logged times for {tar_name} to file (total {total_hours:.2f} h).")


def track_time(start_time, module_name):
    """Calculate elapsed time and return the time taken."""
    elapsed_time = time.time() - start_time
    return elapsed_time




# === SETUP ===
source_dir = Path("/path/processed")
work_dir = Path("/path/PI10_tempUntarred")
quarantine_dir = Path("/path/processed/quarantine")
quarantine_dir.mkdir(parents=True, exist_ok=True)
os.makedirs(work_dir, exist_ok=True)
os.chdir(work_dir)
email_thread = threading.Thread(target=email_scheduler, daemon=True)
email_thread.start()
init_log_file()  # Initialize log file right after setup



# === SETUP ===
paths = {
    'tarred': work_dir / "data/tarred",
    'untarred': work_dir / "data/untarred",
    'output': work_dir / "output",
    'hitsmisses': work_dir / "data/hitsmisses"
}

for path in paths.values():
    path.mkdir(parents=True, exist_ok=True)

# Classification model setup
TIMESTAMP = '2025-09-22_153855-doe'
MODEL_NAME = 'final_model.h5'
TOP_K = 3
plk_paths.timestamp = TIMESTAMP
class_names = load_class_names(splits_dir=plk_paths.get_ts_splits_dir())
model = load_model(os.path.join(plk_paths.get_checkpoints_dir(), MODEL_NAME),
                   custom_objects=utils.get_custom_objects())
with open(os.path.join(plk_paths.get_conf_dir(), 'conf.json')) as f:
    conf = json.load(f)


# === HELPER FUNCTIONS ===
def outputs_exist_for_tar(tar_file):
    stem = tar_file.stem
    required = [
        source_dir / f"{stem}_gpstag.csv",
        source_dir / f"{stem}_hitsmisses.txt",
        source_dir / f"{stem}_Background.tif",
        source_dir / f"{stem}_predictions_relative.json",
        source_dir / f"{stem}_image_properties.csv",
        source_dir / f"{stem}_topspecies.csv"
    ]
    preview_dir = Path(r"/path/not_processed/previews") / f"{stem}_preview"

    if not preview_dir.exists() or not any(preview_dir.glob("*.tif")):
        return False

    for f in required:
        if not f.exists():
            return False
    return True

def get_new_tar_files(source_dir):
    all_tar = list(source_dir.glob("*.tar"))
    outputs_to_check = [
        "_gpstag.csv",
        "_hitsmisses.txt",
        "_Background.tif",
        "_predictions_relative.json",
        "_image_properties.csv",
        "_topspecies.csv"
    ]
    preview_dir = Path(r"/path/not_processed/previews")

    new_files = []
    for tar in all_tar:
        stem = tar.stem
        missing_output = False

        # Check all required files
        for suffix in outputs_to_check:
            expected = source_dir / f"{stem}{suffix}"
            if not expected.exists():
                missing_output = True
                break  # No need to check further if one is missing

        # Check preview folder
        preview_path = preview_dir / f"{stem}_preview"
        if not preview_path.exists() or not any(preview_path.glob("*.tif")):
            missing_output = True

        if missing_output:
            new_files.append(tar)

    return new_files


from time import time as timer

def clear_untarred_dir(dir_path):
    start_time = timer()  # Start timing
    print(f"⚙ Clear and created local directories")
    if dir_path.exists():
        shutil.rmtree(dir_path)
    dir_path.mkdir(parents=True)  # Recreate the directory

    elapsed_time = timer() - start_time  # Calculate elapsed time
    print(f"       ✅ Done in {elapsed_time:.2f} seconds.")
    return elapsed_time  # Return the time taken


def extract_tar(tar_path, extract_to):
    start_time = timer()  # Start timing
    print(f"⚙ Untarring {tar_path.name}...")

    with tarfile.open(tar_path) as tar:
        tar.extractall(path=extract_to)  # Extract the TAR file

    elapsed_time = timer() - start_time  # Calculate elapsed time
    print(f"       ✅ Done in {elapsed_time:.2f} seconds.")
    return elapsed_time  # Return the time taken


def count_images_in_tar(extract_dir, tar_file):
    """Count the number of .tif images in the extracted directory."""
    print(f"⚙ Counting images in {tar_file.name}...")
    tif_files = list(extract_dir.rglob("*.tif"))
    print(f"       ✅ Found {len(tif_files)} .tif files")
    return len(tif_files)

def copy_background_tif(extract_dir, dest_path):
    start = timer()
    print("⚙ Copying Background..")

    for root, _, files in os.walk(extract_dir):
        for f in files:
            if f == "Background.tif":
                full_path = os.path.join(root, f)
                if not dest_path.exists():
                    shutil.copy(full_path, dest_path)
                elapsed_time = timer() - start
                print(f"       ✅ Done in {elapsed_time:.2f} seconds.")
                return  # exit after success

    # If loop finishes without finding the file
    elapsed_time = timer() - start
    print(f"       ⚠️ Background.tif not found")



def extract_hitsmisses(tar_path, output_file):
    start = timer()
    print("⚙ Fetching hits and misses...")
    with tarfile.open(tar_path) as tar:
        hits_file = next((m for m in tar.getmembers() if "hitsmisses.txt" in m.name.lower()), None)
        if hits_file:
            f = tar.extractfile(hits_file)
            df = pd.read_csv(f, header=None)
            df.columns = ['hits', 'misses']
            df['minute'] = range(len(df))
            df['tar_source'] = tar_path.stem
            df.to_csv(output_file, index=False)
    elapsed_time = timer() - start
    print(f"       ✅ Done in {elapsed_time:.2f} seconds.")

def create_preview_images(extract_dir, preview_dir, tar_name, n=200):
    start = timer()
    print(f"⚙ Saving preview image....       ")
    tif_files = list(extract_dir.rglob("*.tif"))
    if len(tif_files) < n:
        print(f"       ❌ Not enough .tif files in {extract_dir}")
        return
    selected = random.sample(tif_files, n)
    preview_path = preview_dir / f"{tar_name}_preview"
    preview_path.mkdir(exist_ok=True)

    from concurrent.futures import ThreadPoolExecutor
    def copy_preview(tif):
        new_name = tif.stem + "_preview.tif"
        shutil.copy(tif, preview_path / new_name)
    with ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(copy_preview, selected)
    elapsed_time = timer() - start
    print(f"       ✅ Done in {elapsed_time:.2f} seconds.")



def extract_exif_metadata(tif_paths, tar_source, batch_size=200):
    print("⚙ Extracting EXIF metadata in batch...")

    exiftool_path = r"/path/PI10\exiftool-13.31_64\exiftool.exe"
    all_data = []
    tif_paths = list(tif_paths)
    n_batches = (len(tif_paths) + batch_size - 1) // batch_size

    total_start_time = time.time()  # Start overall time tracking

    #for batch_idx in tqdm(range(n_batches), desc="ExifTool Batches", unit="batch"): #with ocuntrer
    for batch_idx in range(n_batches):
        batch = tif_paths[batch_idx * batch_size: (batch_idx + 1) * batch_size]
        args = [
            exiftool_path,
            '-GPSLatitude',
            '-GPSLongitude',
            '-FileModifyDate',
            '-n',
            '-api', 'QuickTimeUTC',
            '-api', 'ExifToolVersion=12.31'
        ] + [str(p) for p in batch]

        try:
            result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode != 0:
                print(f"       ❌ Exiftool error (batch {batch_idx}): {result.stderr}")
                continue

            data = []
            exif = {}
            for line in result.stdout.strip().splitlines():
                if line.startswith("======== "):
                    if exif:
                        data.append(exif)
                    exif = {"SourceFile": line.replace("======== ", "").strip()}
                elif ":" in line:
                    k, v = line.split(":", 1)
                    exif[k.strip()] = v.strip()
            if exif:
                data.append(exif)

            all_data.extend(data)

        except Exception:
            continue  # fully silent


    total_elapsed_time = time.time() - total_start_time  # Calculate total elapsed time for EXIF extraction
    print(f"       ✅ Done in {total_elapsed_time:.2f} seconds ({len(all_data)-2} of {len(tif_paths)-2} )")

    df = pd.DataFrame(all_data)

    if not df.empty:
        df["tar_source"] = tar_source
        df["tif_name"] = df["SourceFile"].apply(lambda x: os.path.basename(x))

        if 'DateTimeOriginal' in df.columns:
            df['FileModifyDate_parsed'] = pd.to_datetime(
                df['DateTimeOriginal'], errors='coerce'
            ).dt.strftime('%-m/%-d/%Y  %-I:%M:%S %p')
    return df



def write_exif_csvs(df, tar_name, output_dir, backup_dir):
    df["tif_name"] = df["SourceFile"].apply(lambda x: os.path.basename(x))
    df.drop(columns=["SourceFile"], inplace=True)
    df.to_csv(output_dir / f"{tar_name}_gpstag.csv", index=False)
    df.to_csv(backup_dir / f"{tar_name}_gpstag.csv", index=False)


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

def classify_and_extract_regions(tar_file, extract_dir):
    start_time = time.time()
    base_name = tar_file.stem
    json_path = source_dir / f"{base_name}_predictions_relative.json"
    csv_path = source_dir / f"{base_name}_image_properties.csv"
    FILEPATHS = list(extract_dir.rglob("*.tif"))

    # Filter only useful files
    FILEPATHS = [p for p in FILEPATHS if "Background.tif" not in p.name and "FlowCellEdges.tif" not in p.name]

    if not FILEPATHS:
        print(f"⚠️ No valid .tif files in {base_name}, skipping.")
        return

    print(f"⚙ Predicting {len(FILEPATHS)} TIFF files in {base_name}...")

    # Run prediction
    pred_lab, pred_prob = predict(model, FILEPATHS, conf, top_K=TOP_K, filemode='local')

    results_json = []
    results_csv = []

    for i, path in enumerate(FILEPATHS):
        rel_path = str(path.relative_to(extract_dir))

        # === JSON prediction ===
        labels = [class_names[pred_lab[i, j]] for j in range(TOP_K)]
        probs = [float(pred_prob[i, j]) for j in range(TOP_K)]
        results_json.append({
            "filepath": rel_path,
            f"top{TOP_K}_labels": labels,
            f"top5{TOP_K}_probs": probs
        })

        # === Morphology extraction ===
        try:
            props = getMaxAreaDict(path)
            props["filepath"] = rel_path
            results_csv.append(props)
        except Exception as e:
            print(f"       ❌ Error processing {rel_path}: {e}")

    # Save JSON
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    elapsed_time = time.time() - start_time
    print(f"       ✅ Done in {elapsed_time / 3600:.1f} hours.")

    # Save CSV
    if results_csv:
        pd.DataFrame(results_csv).to_csv(csv_path, index=False)
        print(f"       ✅ Saved image properties CSV: {csv_path.name}")
    else:
        print(f"       ⚠️No region properties written for {base_name}")



def generate_topspecies_csv(json_path):
    print(f"⚙ Generating top species CSV from: {json_path}")

    upper_threshold = 0.95
    diff_threshold = 0.6

    if not json_path.exists():
        print(f"       ❌ JSON file not found: {json_path}")
        return

    try:
        with open(json_path, 'r') as f:
            data_list = json.load(f)
    except Exception as e:
        print(f"       ❌ Failed to read JSON: {e}")
        return

    if not isinstance(data_list, list) or not data_list:
        print("       ⚠️ JSON is empty or invalid.")
        return

    rows = []
    for entry in data_list:
        filepath = entry.get("filepath", "")
        labels = entry.get("top3_labels", [])
        probs = entry.get("top3_probs", entry.get("top53_probs", []))

        if isinstance(labels, str):
            labels = [s.strip() for s in labels.split(",")]
        if isinstance(probs, str):
            probs = [float(s.strip()) for s in probs.split(",")]

        if len(labels) < 2 or len(probs) < 2:
            continue

        label = labels[0]
        prob1 = probs[0]
        prob2 = probs[1]

        # Conditionally append _AI99
        if prob1 > upper_threshold and (prob1 - prob2) > diff_threshold:
            label = f"{label}_AI99"

        rows.append({
            "filename": os.path.basename(filepath),
            "top_species": label
        })

    if not rows:
        print("       ⚠️ No valid predictions to save.")
        return

    output_path = json_path.with_name(json_path.name.replace("_predictions_relative.json", "_topspecies.csv"))
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"       ✅ Done in {elapsed_time:.2f} seconds.")

def check_preview_class_distribution(preview_dir, threshold=0.3):
    print("⚙ Running preview classification check...")

    preview_tifs = list(preview_dir.glob("*_preview.tif"))
    if not preview_tifs:
        print("       ⚠️ No preview images found.")
        return True, None  # Allow pipeline to continue

    pred_lab, pred_prob = predict(model, preview_tifs, conf, top_K=1, filemode='local')
    top1_classes = [class_names[idx] for idx in pred_lab[:, 0]]

    class_counts = pd.Series(top1_classes).value_counts(normalize=True)
    print(f"       ✅ Class distribution in preview: {class_counts.to_dict()}")

    bubble_classes = [cls for cls in class_counts.index if 'bubbles' in cls.lower()]
    bubble_fraction = class_counts[bubble_classes].sum()

    if bubble_fraction > threshold:
        print(f"✅ Combined 'bubbles'-like classes exceed threshold ({threshold:.0%}): {bubble_fraction:.2%}, moved to quarantine")
        return False, 'bubbles'

    return True, None

def extract_only_morphology(tar_file, extract_dir, csv_path):
    base_name = tar_file.stem
    FILEPATHS = list(extract_dir.rglob("*.tif"))
    FILEPATHS = [p for p in FILEPATHS if "Background.tif" not in p.name and "FlowCellEdges.tif" not in p.name]

    if not FILEPATHS:
        print(f"       ⚠️ No valid .tif files in {base_name}, skipping morphology.")
        return

    print(f"       🧬 Extracting morphology for {len(FILEPATHS)} TIFFs...")

    results_csv = []

    for path in FILEPATHS:
        rel_path = str(path.relative_to(extract_dir))
        try:
            props = getMaxAreaDict(path)
            props["filepath"] = rel_path
            results_csv.append(props)
        except Exception as e:
            print(f"       ❌ Morphology error for {rel_path}: {e}")

    if results_csv:
        pd.DataFrame(results_csv).to_csv(csv_path, index=False)
        print(f"       ✅ Saved image properties CSV: {csv_path.name}")
    else:
        print(f"       ⚠️ No morphology data written for {base_name}")


# === MAIN PROCESS ===
def process_tar(tar_file):
    tar_name = tar_file.stem
    print(f"\n🔧🔧🔧 PROCESSING {tar_name.upper()} 🔧🔧🔧")

    # Initialize tracking variables
    times_dict = {}
    num_images = 0
    status_log = []
    quarantined = False
    quarantine_reason = None
    quarantine_target = None
    start_time = time.time()

    if outputs_exist_for_tar(tar_file):
        print(f"📦 All outputs exist for {tar_name}, skipping.")
        return


    # Define paths early
    json_output = source_dir / f"{tar_name}_predictions_relative.json"
    csv_output = source_dir / f"{tar_name}_image_properties.csv"
    topspecies_csv = source_dir / f"{tar_name}_topspecies.csv"
    exif_csv_path = source_dir / f"{tar_name}_gpstag.csv"
    hits_path = source_dir / f"{tar_name}_hitsmisses.txt"
    bg_path = source_dir / f"{tar_name}_Background.tif"
    preview_dir = Path(r"/path/not_processed/previews")
    preview_output = preview_dir / f"{tar_name}_preview"
    tar_dest = paths['tarred'] / tar_file.name
    extract_dir = paths['untarred'] / tar_name


    try:
        # === Step 1: Copy TAR to work dir ===
        start_time = time.time()
        shutil.copy(tar_file, tar_dest)
        status_log.append("TAR copied to working directory")
        times_dict["Copy TAR to working directory"] = track_time(start_time, "Copy TAR to working directory")


        # === Step 2: Untar ===
        start_time = time.time()
        clear_untarred_dir(paths['untarred'])
        extract_dir.mkdir()
        extract_tar(tar_dest, extract_dir)
        status_log.append("Untarred successfully")
        times_dict["Untar"] = track_time(start_time, "Untar")


        # === Step 3: Count number of .tif images ===
        start_time = time.time()
        num_images = count_images_in_tar(extract_dir, tar_file)
        status_log.append(f"Number of images in TAR: {num_images}")


        # === Step 3: Preview Images ===
        start_time = time.time()
        if not preview_output.exists():
            preview_dir.mkdir(parents=True, exist_ok=True)
            create_preview_images(extract_dir, preview_dir, tar_name)
            status_log.append("Preview images created")
        else:
            status_log.append("Preview images already exist (skipped)")
        times_dict["Create preview images"] = track_time(start_time, "Create preview images")


        # === Step 4: Early Preview Classification ===
        start_time = time.time()
        if json_output.exists():
            print("✅ Skipping preview classification (predictions already exist)")
            should_continue, reason = True, None
        else:
            should_continue, reason = check_preview_class_distribution(preview_output, threshold=0.3)

        status_log.append(f"Preview classification result: {reason if reason else 'OK'}")
        times_dict["Early preview classification"] = track_time(start_time, "Early preview classification")

        if not should_continue:
            quarantined = True
            quarantine_reason = reason
            quarantine_target = quarantine_dir / tar_file.name
            shutil.move(str(tar_file), str(quarantine_target))
            status_log.append(f"Moved to quarantine due to '{quarantine_reason}' class")

            # log immediately for quarantined TAR
            daily_tar_reports.append({
                "tar_name": tar_name,
                "status_log": status_log,
                "quarantine_reason": quarantine_reason
            })
            try:
                log_time_to_file(tar_name, times_dict, num_images)
            except Exception as log_err:
                print(f"⚠️ Failed to log quarantined {tar_name}: {log_err}")

            print(f"🚨 Quarantined {tar_name} due to '{quarantine_reason}'")
            return





        # Step 5: Copy Background.tif
        start_time = time.time()
        if not bg_path.exists():
            copy_background_tif(extract_dir, bg_path)
            if bg_path.exists():
                status_log.append("Background.tif copied successfully")
            else:
                status_log.append("❌ Background.tif missing after copy attempt")
        else:
            status_log.append("Background.tif already exists (skipped)")


        times_dict["Copy Background.tif"] = track_time(start_time, "Copy Background.tif")



        # Step 6: Extract hitsmisses.txt
        start_time = time.time()
        if not hits_path.exists():
            extract_hitsmisses(tar_file, hits_path)
            status_log.append("hitsmisses.txt extracted")
        else:
            status_log.append("hitsmisses.txt already exists (skipped)")
        times_dict["Extract hitsmisses.txt"] = track_time(start_time, "Extract hitsmisses.txt")


        # Step 7: Extract EXIF metadata
        start_time = time.time()
        if not exif_csv_path.exists():
            tif_files = list(extract_dir.rglob("*.tif"))
            exif_df = extract_exif_metadata(tif_files, tar_name)
            write_exif_csvs(exif_df, tar_name, paths['output'], source_dir)
            status_log.append("EXIF metadata extracted and saved")
        else:
            print("✅ Skipping EXIF extraction (already exists)")
            status_log.append("EXIF metadata already exists (skipped)")
        times_dict["Extract and save EXIF metadata"] = track_time(start_time, "Extract and save EXIF metadata")


        # Step 8: Classify and extract regions (or extract morphology)
        start_time = time.time()
        if not csv_output.exists():
            if json_output.exists():
                extract_only_morphology(tar_file, extract_dir, csv_output)
                status_log.append("Image properties CSV created from existing predictions")
            else:
                classify_and_extract_regions(tar_file, extract_dir)
                status_log.append("Classification and morphology run together (both files missing)")
        elif not json_output.exists():
            classify_and_extract_regions(tar_file, extract_dir)
            status_log.append("Re-ran full classification due to missing JSON")
        else:
            status_log.append("Classification already exists (skipped)")
        times_dict["Classification and morphology extraction"] = track_time(start_time, "Classification and morphology extraction")


        # Step 9: Generate top species CSV
        start_time = time.time()  # Start timing for this step
        if not topspecies_csv.exists():
            if json_output.exists():
                generate_topspecies_csv(json_output)
                status_log.append("Top species CSV generated")
            else:
                status_log.append("Top species CSV skipped (JSON not found)")
        else:
            status_log.append("Top species CSV already exists (skipped)")
        times_dict["Generate top species CSV"] = track_time(start_time, "Generate top species CSV")


    except Exception as e:
        status_log.append(f"❌ Unexpected error: {e}")

    finally:
        try:
            if tar_dest.exists():
                os.remove(tar_dest)
            clear_untarred_dir(paths['untarred'])
        except Exception as cleanup_err:
            status_log.append(f"⚠️ Cleanup failed: {cleanup_err}")

        # Track processed TAR for daily summary
        today_str = datetime.datetime.now().strftime('%Y-%m-%d')
        daily_tar_reports.append({
            "tar_name": tar_name,
            "status_log": status_log
        })

        # Always log times, even if pipeline errored
        try:
            print(f"Logging times for {tar_name}: {times_dict}")
            log_time_to_file(tar_name, times_dict, num_images)
        except Exception as log_err:
            print(f"⚠️ Failed to log times for {tar_name}: {log_err}")

        print(f"✅ Done: {tar_name}")



# === CONTINUOUS WATCH ===

print("⚙  Watching for new .tar files (press Ctrl+C to stop)...")
while True:
    now = datetime.datetime.now()
    today_str = now.strftime('%Y-%m-%d')

    new_files = get_new_tar_files(source_dir)
    pipeline_count = len(new_files)

    if pipeline_count == 0:
        print(f"[{time.ctime()}] No new .tar files. Sleeping...")
        time.sleep(3600)
    else:
        print(f"[{time.ctime()}] ✅ {pipeline_count} .tar file(s) in the processing pipeline.")
        for tar_file in new_files:
            lockfile = source_dir / f"{tar_file.stem}.lock"
            if lockfile.exists():
                continue
            try:
                lockfile.touch(exist_ok=False)
                process_tar(tar_file)
            except Exception as e:
                print(f"❌ Failed to process {tar_file.name}: {e}")
            finally:
                if lockfile.exists():
                    lockfile.unlink()
        print("🔁 Rechecking in 1 hour...")