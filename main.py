import numpy as np
import cv2
import tensorflow as tf
from fastapi import FastAPI
from fastapi import UploadFile, File
import shutil

"""# Global Objects"""

app = FastAPI(title="ML Models as API on Google Colab", description="with FastAPI and ColabCode", version="1.0")

"""# Constants"""

model_path = '3DCNN-model'


def load_and_read_video_frames(path, num_frames, frame_height, frame_width,num_channels):
    cap = cv2.VideoCapture(path)
    frames = []
    for _ in range(num_frames):
        ret,frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (frame_width, frame_height))
        normalized_frame = frame / 255.0
        frames.append(normalized_frame)
    cap.release()
    frames = np.array(frames)
    return frames


@app.on_event("startup")
def load_model():
    global model
    model = tf.keras.models.load_model(model_path)
    model.summary()

"""# Post Prediction"""

@app.post("/")
async def get_predictions(video: UploadFile = File(...)):
    try:
      dest_path = f"3DCNN-model/{video.filename}"
      with open(dest_path, "wb") as f:
       shutil.copyfileobj(video.file, f)
      video_data = load_and_read_video_frames(dest_path,150,160,160,3)
      video_data = np.expand_dims(video_data, axis=0)
      prediction = model.predict(video_data)
      if(prediction.round() == 1.0):
        predicted_class = "Violent"
      else:
        predicted_class = "Nonviolent"
      return {"Prediction": predicted_class}

    except Exception as e:
         return {"prediction": "error", "exception":str(e)}

