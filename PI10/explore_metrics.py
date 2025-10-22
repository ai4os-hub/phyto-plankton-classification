import os
import pandas as pd
from skimage.io import imread
from skimage import measure, morphology
from skimage.color import rgb2gray
import numpy as np
from tqdm import tqdm
from renumics import spotlight

# ---------- FUNCTIONS ----------

def getImageRegionList(filename):
    """Return region properties of thresholded image."""
    image = imread(filename)

    if image.ndim == 3:
        image = rgb2gray(image)

    # Threshold image (simple mean threshold)
    image_threshold = np.where(image > np.mean(image), 0., 1.0)

    # Dilation
    image_dilated = morphology.dilation(image_threshold, np.ones((4, 4)))

    # Label connected regions
    label_list = measure.label(image_dilated)

    # Combine with thresholded image
    label_list = (image_threshold * label_list).astype(int)

    return measure.regionprops(label_list)


def getMaxArea(filename):
    """Return the region with the largest area."""
    region_list = getImageRegionList(filename)
    maxArea = None
    for prop in region_list:
        if maxArea is None or prop.area > maxArea.area:
            maxArea = prop
    return maxArea


def getMaxAreaDict(filename):
    """Return dictionary of features for the largest object in an image."""
    prop = getMaxArea(filename)

    if prop is None:
        return {'area': 0}

    return {
        'object_additional_diameter_equivalent': prop.equivalent_diameter,
        'object_additional_length_minor_axis': prop.minor_axis_length,
        'object_additional_length_major_axis': prop.major_axis_length,
        'object_additional_eccentricity': prop.eccentricity,
        'object_additional_area': prop.area,
        'object_additional_perimeter': prop.perimeter,
        'object_additional_orientation': prop.orientation,
        'object_additional_area_convex': prop.convex_area,
        'object_additional_area_filled': prop.filled_area,
        'object_additional_box_min_row': prop.bbox[0],
        'object_additional_box_max_row': prop.bbox[2],
        'object_additional_box_min_col': prop.bbox[1],
        'object_additional_box_max_col': prop.bbox[3],
        'object_additional_ratio_extent': prop.extent,
        'object_additional_ratio_solidity': prop.solidity,
        'object_additional_inertia_tensor_eigenvalue1': prop.inertia_tensor_eigvals[0],
        'object_additional_inertia_tensor_eigenvalue2': prop.inertia_tensor_eigvals[1],
        'object_additional_moments_hu1': prop.moments_hu[0],
        'object_additional_moments_hu2': prop.moments_hu[1],
        'object_additional_moments_hu3': prop.moments_hu[2],
        'object_additional_moments_hu4': prop.moments_hu[3],
        'object_additional_moments_hu5': prop.moments_hu[4],
        'object_additional_moments_hu6': prop.moments_hu[5],
        'object_additional_moments_hu7': prop.moments_hu[6],
        'object_additional_euler_number': prop.euler_number,
        'object_additional_countcoords': len(prop.coords)
    }


# ---------- MAIN SCRIPT ----------

# Input folder with .tiff images
image_folder = rf"\PATH\plankton-imager-10\not_processed\spotlight"

# Output CSV file
csv_folder = rf"\PATH\plankton-imager-10\not_processed\spotlight"
os.makedirs(csv_folder, exist_ok=True)
output_csv = os.path.join(csv_folder, "image_metrics.csv")



# ---------- SPOTLIGHT VISUALIZATION ----------
from renumics import spotlight


tif_files = []
for root, dirs, files in os.walk(image_folder):
    for f in files:
        if f.lower().endswith((".tif", ".tiff")):
            tif_files.append(os.path.join(root, f))

print(f"Found {len(tif_files)} TIFF images.")


image_properties = []

# Process all images with progress bar
for file_path in tqdm(tif_files, desc="Processing images", unit="image"):
    props = getMaxAreaDict(file_path)
    filename = os.path.basename(file_path)
    subfolder = os.path.basename(os.path.dirname(file_path))
    props = {
        "subfolder": subfolder,
        "image_name": filename,
        "image_path": file_path,
        **props
    }
    image_properties.append(props)

    df = pd.DataFrame(image_properties)

    # Save to CSV
df.to_csv(output_csv, index=False)
print(f"Saved metrics to {output_csv}")


embedding_columns = [col for col in df.columns if col.startswith("object_additional")]
# Create a new column 'embeddings' as a list of floats for each row
df['embeddings'] = df[embedding_columns].values.tolist()


# Tell Spotlight that 'embeddings' column is an embedding
dtype = {"embeddings": spotlight.Embedding}

# Display the DataFrame in Spotlight
spotlight.show(df, dtype=dtype)