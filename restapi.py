import argparse
from flask import Flask, request, jsonify
import torch
from PIL import Image
import io

app = Flask(__name__)
models = {}

DETECTION_URL = "/v1/object-detection/<model>"

# YOLOv5 모델 로드를 위한 준비
def load_model(name, path):
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=path, force_reload=True)
        app.logger.info(f"Model {name} loaded successfully from {path}")
        return model
    except Exception as e:
        app.logger.error(f"Error loading model {name} from {path}: {e}")
        return None

@app.route(DETECTION_URL, methods=["POST"])
def predict(model):
    app.logger.info(f"Received request for model {model}")
    if request.method != "POST":
        return jsonify({"error": "Only POST method is supported"}), 405

    if request.files.get("image"):
        im_file = request.files["image"]
        im_bytes = im_file.read()
        im = Image.open(io.BytesIO(im_bytes))

        if model in models:
            results = models[model](im, size=640)
            app.logger.info(f"Object detection performed successfully for model {model}")
            # 결과를 JSON 형식으로 변환하여 반환
            return jsonify(results.pandas().xyxy[0].to_json(orient="records"))
        else:
            app.logger.warning(f"Model {model} not found")
            return jsonify({"error": "Model not found"}), 404
    else:
        app.logger.warning("Image file is missing in the request")
        return jsonify({"error": "Image file is missing"}), 400

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask API exposing YOLOv5 model")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    parser.add_argument("--model_path", type=str, default="petom_weights.pt", help="path to model weight file")
    opt = parser.parse_args()

    # 모델 이름을 path와 함께 직접 지정하여 로드
    model_name = "petom_weights"  # 이 예제에서는 모델 이름을 사용자 정의 이름으로 설정
    loaded_model = load_model(model_name, opt.model_path)
    if loaded_model:
        models[model_name] = loaded_model

    app.run(host="0.0.0.0", port=opt.port)
