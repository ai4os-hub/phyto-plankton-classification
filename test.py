from deepaas.model.v2.wrapper import UploadedFile
from planktonclas.api import predict

image_path = r"C:\Users\wout.decrop\Documents\environments\phytoplankton_classifier\phyto-plankton-classification\data\demo-images\Actinoptychus\ecotaxa_Actinoptychus_0A02C03F-CE0F-420C-8325-8F1AC79B5F63.jpg"

image_file = UploadedFile(
    name="image",
    filename=image_path,
    content_type="image/jpeg",
    original_filename="ecotaxa_Actinoptychus.jpg"
)

result = predict(image=image_file, zip=None)

print(result)
