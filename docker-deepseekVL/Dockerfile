# 第一階段：準備帶有 CUDA 工具的環境
FROM nvidia/cuda:12.1.0-devel-ubuntu20.04 AS cuda-env

# 設置環境變數以避免交互式提示
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Taipei

# 安裝必要的工具和 Python 3.10
RUN apt-get update && apt-get install -y \
    wget \
    git \
    build-essential \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.10 \
    python3.10-distutils \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

# 確保 Python 3.10 為預設版本
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --config python3

# 確保 pip 使用正確版本
RUN apt-get update && apt-get install -y curl && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# 安裝 PyTorch 和其他依賴
RUN python3 -m pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121 --no-cache-dir --no-deps
RUN python3 -m pip install --upgrade setuptools wheel packaging --retries 100
RUN python3 -m pip install numpy sympy --retries 100
RUN python3 -m pip install flash-attn==2.5.8 --no-build-isolation --target=/usr/local/lib/python3.10/site-packages/ --retries 100

# 第二階段：基於 MineDojo，添加 CUDA 工具
FROM minedojo/minedojo:latest

# 從第一階段拷貝 CUDA 工具到當前環境
COPY --from=cuda-env /usr/local/cuda /usr/local/cuda
COPY --from=cuda-env /usr/local/lib/python3.10/site-packages/flash_attn /opt/conda/lib/python3.10/site-packages/flash_attn
COPY --from=cuda-env /usr/local/lib/python3.10/site-packages/flash_attn-2.5.8.dist-info /opt/conda/lib/python3.10/site-packages/flash_attn-2.5.8.dist-info
COPY --from=cuda-env /usr/local/lib/python3.10/site-packages/flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so /opt/conda/lib/python3.10/site-packages/

# 設置環境變數
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# 禁用用户路径
ENV PYTHONNOUSERSITE=1
ENV PYTHONPATH="/home/user/MineCLIP:/home/user/MineDojo:/opt/conda/lib/python3.10/site-packages"

# MineCLIP
RUN sudo /opt/conda/bin/conda install -y python=3.10
RUN sudo /opt/conda/bin/python3.10 -m pip install "pip<24.1"
RUN sudo /opt/conda/bin/python3.10 -m pip install wheel==0.38.4 setuptools==65.5.1 --retries 100
COPY --chown=user:user MineCLIP /home/user/MineCLIP
RUN sudo /opt/conda/bin/python3.10 -m pip install --no-user --retries 100 -e /home/user/MineCLIP

# 卸载 minedojo
RUN sudo /opt/conda/bin/pip uninstall -y minedojo
RUN sudo /opt/conda/bin/python3.10 -m pip uninstall -y minedojo

# MineDojo
COPY --chown=user:user MineDojo /home/user/MineDojo
RUN sudo /opt/conda/bin/python3.10 -m pip install --no-user --retries 100 -e /home/user/MineDojo

# 安装 requirements.txt
COPY --chown=user:user requirements.txt /home/user/requirements.txt
RUN sudo /opt/conda/bin/python3.10 -m pip install --no-user --retries 100 -r /home/user/requirements.txt

# Tools and scripts for Omniverse Farm
COPY thirdparty/omnicli /omnicli
COPY scripts/docker/run.sh /run.sh

RUN sudo chmod +x /omnicli/omnicli && sudo chmod +x /run.sh

RUN xvfb-run python /home/user/MineDojo/scripts/validate_install.py
