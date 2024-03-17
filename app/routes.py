from flask import Flask
from mysql.connector import connect
from dotenv import load_dotenv
import os

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

# 예시 라우트
@app.route('/data')
def get_data():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM skin_db')  # your_table_name을 실제 테이블 이름으로 변경하세요.
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
