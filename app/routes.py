from flask import Flask, request, jsonify
from mysql.connector import connect
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
    connection = connect(
        host=os.getenv('DB_HOST'),
        database=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD')
    )
    return connection

# 이미지 분석 모델 로드 (실제 모델 로드 로직에 따라 다를 수 있음)
model = torch.load('model_path.pt')
model.eval()

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
