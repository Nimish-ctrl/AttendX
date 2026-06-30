FROM python:3.10-slim

WORKDIR /app


RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


RUN python -c "\
import os; \
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'; \
from deepface import DeepFace; \
DeepFace.build_model('Facenet512'); \
print('Facenet512 weights downloaded')"


COPY warmup.py .
RUN python warmup.py

COPY . .

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]