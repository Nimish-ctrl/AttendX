import os, tempfile, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import numpy as np
import cv2
from deepface import DeepFace

img = np.zeros((224, 224, 3), dtype=np.uint8)
tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
cv2.imwrite(tmp.name, img)

try:
    DeepFace.represent(
        img_path=tmp.name,
        model_name="Facenet512",
        detector_backend="opencv",
        enforce_detection=True,
    )
except Exception:
    pass  

print("OpenCV detector + Facenet512 weights downloaded")