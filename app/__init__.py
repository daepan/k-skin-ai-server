from flask import Flask

app = Flask(__name__)

# 모델 로딩
from .models import load_model
model = load_model()

# 라우트 설정
from . import routes
