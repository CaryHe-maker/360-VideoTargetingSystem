# syntax=docker/dockerfile:1.7

# The official PyTorch runtime image already contains matching CUDA builds of
# torch and torchvision, so only the remaining dependencies are installed here.
FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python -m pip install \
        "numpy>=1.26,<3" \
        "opencv-python-headless>=4.11,<5" \
        "PyYAML>=6,<7" \
        "timm==0.5.4" \
        "easydict>=1.13,<2"

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
RUN python -m pip install --no-deps -e .

# Model weights are not baked into the image. Mount them at runtime:
#   docker run --gpus all -v $PWD/models:/app/models -v $PWD/data:/data track360 \
#     track --input /data/video.mp4 --init-box x,y,w,h --output /data/result.txt
VOLUME ["/app/models"]

ENTRYPOINT ["track360"]
CMD ["--help"]
