# 第一階段：準備帶有 CUDA 工具的環境
FROM nvidia/cuda:12.1.0-devel-ubuntu20.04 AS cuda-env

# 設置環境變數以避免交互式提示
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Taipei

# 安裝必要的工具和依賴（包含 Python 3.10.14 需要的依賴）
RUN apt-get update && apt-get install -y \
    wget \
    git \
    build-essential \
    software-properties-common \
    libssl-dev \
    libreadline-dev \
    zlib1g-dev \
    libsqlite3-dev \
    libbz2-dev \
    libncurses5-dev \
    libgdbm-dev \
    liblzma-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# 下載和安裝 Python 3.10.14
RUN cd /usr/src && \
    wget https://www.python.org/ftp/python/3.10.14/Python-3.10.14.tgz && \
    tar xzf Python-3.10.14.tgz && \
    cd Python-3.10.14 && \
    ./configure --enable-optimizations && \
    make -j$(nproc) && \
    make altinstall && \
    ln -sf /usr/local/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/local/bin/python3.10 /usr/bin/python3.10 && \
    rm -rf /usr/src/Python-3.10.14*

# 確保 pip 使用正確版本
RUN wget https://bootstrap.pypa.io/get-pip.py && python3.10 get-pip.py && rm get-pip.py

# 安裝 PyTorch 和其他依賴
RUN python3.10 -m pip install --no-cache-dir torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
RUN python3.10 -m pip install --upgrade setuptools wheel packaging
RUN python3.10 -m pip install numpy sympy

# 第二階段：基於 MineDojo，添加 CUDA 工具
FROM minedojo/minedojo:latest

# 從第一階段拷貝 CUDA 工具到當前環境
COPY --from=cuda-env /usr/local/cuda /usr/local/cuda

# 設置環境變數
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# 禁用用户路径
ENV PYTHONNOUSERSITE=1
ENV PYTHONPATH="/home/user/MineCLIP:/home/user/MineDojo:/opt/conda/lib/python3.10/site-packages"

# 設置 Conda 內 Python 版本為 3.10.14
RUN sudo /opt/conda/bin/conda install -y python=3.10.14
RUN sudo /opt/conda/bin/python3.10 -m pip install "pip<24.1"
RUN sudo /opt/conda/bin/python3.10 -m pip install wheel==0.38.4 setuptools==65.5.1

# 安裝 MineCLIP
COPY --chown=user:user MineCLIP /home/user/MineCLIP
RUN sudo /opt/conda/bin/python3.10 -m pip install --no-user --retries 100 -e /home/user/MineCLIP

# 卸载 minedojo
RUN sudo /opt/conda/bin/pip uninstall -y minedojo
RUN sudo /opt/conda/bin/python3.10 -m pip uninstall -y minedojo

# 安裝 MineDojo
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
