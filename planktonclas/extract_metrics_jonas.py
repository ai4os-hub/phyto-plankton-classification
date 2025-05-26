import os
import pandas as pd
import zipfile
import shutil
from skimage.io import imread
from skimage import measure, morphology
from skimage.color import rgb2gray
import numpy as np
from tqdm import tqdm



# Define the base with cyz files (we need their name to find the csv files)
base_path1 = r"\\fs\SHARED\transfert\Wout_er_zijn_geen_vaste_stoelen\1740\RawImages"


def getImageRegionList(filename):
    # Read the image
    image = imread(filename)

    # If the image is colored (RGB), convert it to grayscale
    if image.ndim == 3:
        image = rgb2gray(image)

    # Threshold the image
    image_threshold = np.where(image > np.mean(image), 0., 1.0)

    # Perform dilation
    image_dilated = morphology.dilation(image_threshold, np.ones((4, 4)))

    # Label the regions
    label_list = measure.label(image_dilated)

    # Combine the thresholded image and labels
    label_list = (image_threshold * label_list).astype(int)

    # Return region properties
    return measure.regionprops(label_list)

# Find the region with the largest area
def getMaxArea(filename):
    region_list = getImageRegionList(filename)

    maxArea = None
    for property in region_list:
        if maxArea is None:
            maxArea = property
        else:
            if property.area > maxArea.area:
                maxArea = property
    return maxArea

def getMaxAreaDict(filename):
    property = getMaxArea(filename)

    if property is None:
        maxAreaDict = {'area': 0}
    else:
        maxAreaDict = {



            # Equivalent diameter (Unknown table prefix → Add "object_additional_")
            'object_additional_diameter_equivalent': property.equivalent_diameter,

            # Axis lengths (Unknown table prefix → Add "object_additional_")
            'object_additional_length_minor_axis': property.minor_axis_length,
            'object_additional_length_major_axis': property.major_axis_length,

            # Shape properties (Format must be Table_Field → Keep normal)
            'object_additional_eccentricity': property.eccentricity,
            'object_additional_area': property.area,
            'object_additional_perimeter': property.perimeter,
            'object_additional_orientation': property.orientation,

            # Additional area-related properties (Unknown table prefix → Add "object_additional_")
            'object_additional_area_convex': property.convex_area,
            'object_additional_area_filled': property.filled_area,

            # Bounding box (Unknown table prefix → Add "object_additional_")
            'object_additional_box_min_row': property.bbox[0],
            'object_additional_box_max_row': property.bbox[2],
            'object_additional_box_min_col': property.bbox[1],
            'object_additional_box_max_col': property.bbox[3],

            # Ratio properties (Unknown table prefix → Add "object_additional_")
            'object_additional_ratio_extent': property.extent,
            'object_additional_ratio_solidity': property.solidity,

            # Inertia tensor eigenvalues (Unknown table prefix → Add "object_additional_")
            'object_additional_inertia_tensor_eigenvalue1': property.inertia_tensor_eigvals[0],
            'object_additional_inertia_tensor_eigenvalue2': property.inertia_tensor_eigvals[1],

            # Hu moments (Unknown table prefix → Add "object_additional_")
            'object_additional_moments_hu1': property.moments_hu[0],
            'object_additional_moments_hu2': property.moments_hu[1],
            'object_additional_moments_hu3': property.moments_hu[2],
            'object_additional_moments_hu4': property.moments_hu[3],
            'object_additional_moments_hu5': property.moments_hu[4],
            'object_additional_moments_hu6': property.moments_hu[5],
            'object_additional_moments_hu7': property.moments_hu[6],

            # Euler number (Unknown table prefix → Add "object_additional_")
            'object_additional_euler_number': property.euler_number,

            # Count coordinates (Format must be Table_Field → Keep normal)
            'object_additional_countcoords': len(property.coords)
        }

    return maxAreaDict



# Function to add image properties to the dataset
def add_image_properties_to_data(data, image_folder):
    image_properties = []

    # Get all possible keys from getMaxAreaDict
    all_keys = [
        # Unknown table prefix → Add "object_additional_"
        'object_additional_diameter_equivalent',
        'object_additional_length_minor_axis', 'object_additional_length_major_axis', 'object_additional_area_convex',
        'object_additional_area_filled', 'object_additional_box_min_row', 'object_additional_box_max_row',
        'object_additional_box_min_col', 'object_additional_box_max_col', 'object_additional_ratio_extent',
        'object_additional_ratio_solidity', 'object_additional_inertia_tensor_eigenvalue1',
        'object_additional_inertia_tensor_eigenvalue2', 'object_additional_moments_hu1', 'object_additional_moments_hu2',
        'object_additional_moments_hu3', 'object_additional_moments_hu4', 'object_additional_moments_hu5',
        'object_additional_moments_hu6', 'object_additional_moments_hu7', 'object_additional_euler_number',
        'object_additional_eccentricity', 'object_additional_perimeter', 'object_additional_orientation', 'object_additional_area', 'object_additional_countcoords']
        # Format must be Table_Field → Keep normal
    #     'eccentricity', 'perim.', 'orientation', 'area', 'countcoords'
    # ]

    for _, row in data.iterrows():
        img_file = os.path.join(image_folder, str(row['object_id']))
        if os.path.exists(img_file):
            props = getMaxAreaDict(img_file)
        else:
            props = {key: None for key in all_keys}  # Assign None to all properties if file is missing

        image_properties.append(props)

    # Convert to DataFrame and merge
    properties_df = pd.DataFrame(image_properties)
    data = pd.concat([data, properties_df], axis=1)

    return data


import os
import tarfile
import shutil
import pandas as pd
from tqdm import tqdm

base_path1 = r"\\fs\SHARED\transfert\Wout_er_zijn_geen_vaste_stoelen\1740\RawImages"
zip_folder = r"\\qarchive\data_sensors\plankton-imager-10\2025-04-22\tarred"
output_folder = r"\\qarchive\data_sensors\plankton-imager-10\2025-04-22\lengths"
temp_folder_o = r"\\qarchive\data_sensors\plankton-imager-10\2025-04-22\temp"

# Make sure output and temp folders exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(temp_folder_o, exist_ok=True)
os.makedirs(temp_folder_o, exist_ok=True)
# Get list of tar files
tar_files = [f for f in os.listdir(zip_folder) if f.endswith('.tar')]

# Loop over all .tar files with progress bar
for tar_filename in tqdm(tar_files, desc="Processing tar files", unit="tar"):
    tar_path = os.path.join(zip_folder, tar_filename)
    temp_folder= os.path.join(temp_folder_o, tar_filename[:-4])
    print("opening tar file: ",tar_path)
    # Extract tar file into temp_folder
    with tarfile.open(tar_path) as tar:
        tar.extractall(path=temp_folder)
    # List to store image properties
    image_properties = []

    # Collect all .tif files inside temp_folder
    tif_files = []
    

    raw= os.path.join(temp_folder,"RawImages")

    files = os.listdir(raw)
    for filename in files:
        if filename.endswith('.tif'):
            tif_files.append(os.path.join(raw, filename))
    
    # Loop over all .tif files with progress bar
    for file_path in tqdm(tif_files, desc=f"Processing {tar_filename}", leave=False, unit="image"):
        # Get properties from your function
        props = getMaxAreaDict(file_path)
        filename = os.path.basename(file_path)

        props =props = {'image_name': filename, **props}
        # Add properties to list
        image_properties.append(props)
    # Convert the list of dictionaries into a DataFrame
    df = pd.DataFrame(image_properties)

    # Define output CSV name based on the tar filename (without '.tar')
    base_name = os.path.splitext(tar_filename)[0]
    output_file = os.path.join(output_folder, f"{base_name}_image_properties.csv")

    # Save the DataFrame
    df.to_csv(output_file, index=False)
    print(f"Saved {output_file}")
    # Move the processed tar file to the new folder
    tar_done_folder = r"\\qarchive\data_sensors\plankton-imager-10\2025-04-22\tarred_done"
    shutil.move(tar_path, os.path.join(tar_done_folder, tar_filename))

    print(f"Moved {tar_filename} to {tar_done_folder}")


    # # Clean up temp_folder by deleting all contents
    shutil.rmtree(temp_folder)
    os.makedirs(temp_folder, exist_ok=True)  # Recreate empty temp_folder
    break
shutil.rmtree(temp_folder_o)
