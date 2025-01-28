import base64
import json
import time

import boto3


# This information is retrieved from your deployment information window
MINIO_BUCKET = "ai4papi-*************************************************"
MINIO_URL = "https://****************************************************"
MINIO_ACCESS_KEY = "**********************************************@egi.eu"
MINIO_SECRET_KEY = "*****************************************************"

# This is how you decide to name your new prediction
prediction_ID = "test-prediction"

# Local paths (in current folder)
pth_local_input = f"input-{prediction_ID}.json"
pth_local_output = f"output-{prediction_ID}.json"
pth_local_logs = f"output-{prediction_ID}.log"

# Remote paths (in the bucket)
# Two files will be produced in the output folder of the bucket
# * <input_filename>.json: this file has the output of the prediction, in JSON format.
#   --> this will only be created if the prediction is successful
# * <input_filename>.log: this file has the logs of the prediction.
#   --> this will always be created
pth_remote_input = f"inputs/{prediction_ID}.json"
pth_remote_output = f"outputs/{prediction_ID}.json"
pth_remote_logs = f"outputs/{prediction_ID}.log"

# Prepare the data you want to predict
def get_base64(fpath):
    """
    Encode files as base64. We need to do this to pass files as prediction inputs in
    the JSON file.
    """
    with open(fpath, "rb") as f:
        encoded_str = base64.b64encode(f.read()).decode("utf-8")
    return encoded_str

data = {
    "oscar-files": [
        {
            "key": "image",
            "file_format": "jpg",
            "data": get_base64("oscar\plankton.jpg"),
        },
    ]
}

# Create the JSON file
with open(pth_local_input, "w") as f:
    json.dump(data, f)

# Init the Minio Object Store
client = boto3.client(
    "s3",
    endpoint_url=MINIO_URL,
    region_name="",
    verify=True,
    aws_access_key_id=MINIO_ACCESS_KEY,
    aws_secret_access_key=MINIO_SECRET_KEY,
)

# Upload the inputs to the bucket
with open(pth_local_input, "rb") as data:
    client.upload_fileobj(data, MINIO_BUCKET, pth_remote_input)
print(f"Uploaded data to {pth_remote_input} in bucket {MINIO_BUCKET}")

# Now we wait until the prediction is made
while True:
    # List objects in the bucket
    r = client.list_objects_v2(Bucket=MINIO_BUCKET)
    contents = [i["Key"] for i in r["Contents"]]

    # If the output is available, download it
    if pth_remote_logs in contents:
        with open(pth_local_logs, "wb") as data:
            client.download_fileobj(MINIO_BUCKET, pth_remote_logs, data)
        print(f"Downloaded logs from {pth_remote_logs} in bucket {MINIO_BUCKET}")

        # Prediction JSON will only be available if the prediction was successful
        if pth_remote_output in contents:
            with open(pth_local_output, "wb") as data:
                client.download_fileobj(MINIO_BUCKET, pth_remote_output, data)
            print(f"Downloaded data from {pth_remote_output} in bucket {MINIO_BUCKET}")

        break

    else:
        print("Waiting for the prediction to complete ...")
        time.sleep(5)