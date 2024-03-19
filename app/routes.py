from flask import Flask, request, jsonify
import pymysql
from dotenv import load_dotenv
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import io

# 환경 변수 로드
load_dotenv()

app = Flask(__name__)

# 데이터베이스 연결 함수
def get_db_connection():
    connection = pymysql.connect(
        host=os.getenv('DB_HOST'),
        db=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        passwd=os.getenv('DB_PASSWORD'),
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )
    return connection


model = torch.hub.load('ultralytics/yolov5', 'custom', path='models_train/petom_weights.pt', force_reload=True)


def transform_image(image_bytes):
    my_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

def analyze_image(image_bytes):
    tensor = transform_image(image_bytes)
    outputs = model(tensor)
    _, prediction = outputs.max(1)
    # 예시: '질환 이름'과 확률 90.5%를 반환합니다. 실제 로직은 모델의 출력에 맞게 수정해야 합니다.
    return "질환 이름", 90.5

@app.route('/data')
def get_data():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM skin_db')  # 테이블 이름을 실제 테이블 이름으로 변경하세요.
    data = cursor.fetchall()
    cursor.close()
    conn.close()
    return str(data)

@app.route('/petskin/check', methods=['POST'])
def check_pet_skin():
    if 'file' not in request.files:
        return jsonify({"error": "no file"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "no file"}), 400

    try:
        prediction, confidence = analyze_image(file.read())

        if prediction is None:
            response_message = "정상 혹은 아직 학습되지 않은 피부질환입니다."
        else:
            response_message = f"{prediction}로 의심됩니다. 정확도는 {confidence}%입니다."

        return jsonify({"message": response_message})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/detect', methods=['POST'])
def detect():
    if request.method == 'POST':
        if 'file' not in request.files:
            # 바이너리 데이터가 body에 포함되어 있는지 확인
            if request.data:
                im_bytes = request.data  # 바이트 데이터 읽기
                img = Image.open(io.BytesIO(im_bytes))  # 이미지로 변환
            else:
                return jsonify({"error": "No file or binary data provided"}), 400
        else:
            # 파일 업로드 처리
            im_file = request.files['file']
            if im_file.filename == '':
                return jsonify({"error": "no file selected"}), 400
            im_bytes = im_file.read()
            img = Image.open(io.BytesIO(im_bytes))

        # 모델로 이미지 분석 실행
        results = model(img, size=640)

        # 분석 결과 처리 및 JSON 형식으로 결과 반환
        results_data = {"message": "Image processed successfully."}
        return jsonify(results_data)
    
    else:
        # GET 요청이 들어온 경우 에러 메시지 반환
        return jsonify({"message": "Please use POST request to analyze images."})
