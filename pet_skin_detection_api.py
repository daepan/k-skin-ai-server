import argparse
import io
import pathlib

import torch
from flask import Flask, request
from PIL import Image
from flask_cors import CORS


app = Flask(__name__)
CORS(app, origins="http://localhost:3001")

pathlib.WindowsPath = pathlib.PosixPath
model = torch.hub.load('/home/ubuntu/yolov5', 'custom', path='/home/ubuntu/petom_weights.pt', source='local', force_reload=True)

@app.route("/predict", methods=["POST"])
def predict():
    if request.method != "POST":
        return

    if request.files.get("image"):
        im_file = request.files["image"]
        im_bytes = im_file.read()
        im = Image.open(io.BytesIO(im_bytes))

        results = model(im)
        return results.pandas().xyxy[0].to_json(orient='records')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask API exposing YOLOv5 model")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    opt = parser.parse_args()

    app.run(host="0.0.0.0", port=opt.port)