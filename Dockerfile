FROM python:3.11-slim

# TensorFlow needs OpenMP runtime; opencv-python-headless avoids GUI libs
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces run containers as UID 1000
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" \
    HF_HOME=/home/user/.cache/huggingface \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

WORKDIR /home/user/app

COPY --chown=user requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY --chown=user web ./web
COPY --chown=user samples ./samples
COPY --chown=user class_labels.txt plant_disease_model.h5 ./

EXPOSE 7860

CMD ["python", "web/app.py"]
