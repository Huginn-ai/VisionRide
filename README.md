# VisionRide

## 项目介绍



## 环境要求

**硬件环境**

Server端：16GB显存的显卡

Client端：树莓派 & 摄像头 & 蓝牙耳机

**软件环境**

Server端：ubuntu 22.04 & cuda 11.7+

Client端：python 3.6+

## 安装

### Server

安装xtts-api-server

> https://github.com/daswer123/xtts-api-server

安装OpenALPR

> https://github.com/openalpr/openalpr

安装ollama

> https://ollama.com/

下载LLM：Phi-4（可替换）

```
ollama run phi4
```

下载Depth Anything的度量深度估计模型checkpoint并放置到`./checkpoints`文件夹下

> https://huggingface.co/spaces/LiheYoung/Depth-Anything/tree/main/checkpoints_metric_depth

从配置文件中新建Conda环境

```
conda env create -f env.yml
```

修改`const.py`中的配置项，主要是修改下面的内容

```python
# XTTS Service URL
XTTS_URL = "http://127.0.0.1:8003/tts_to_audio"

# Absolute path of speaker file
SPEAKER_WAV_PATH = "/home/ubuntu/re_wav/en_re_man.wav"
```

**Note**：运行服务端时可能会报错说找不到torchhub，此时需要将torchhub文件夹放在用户根目录下

```
mv ./torchhub ~/torchhub
```

### Client

```
pip install -r requirements.txt
```

修改`config.py`中的配置项，主要是要修改下面的内容，与服务端保持一致

```python
# Server Host Address
HOST = '127.0.0.1'

# Server port
PORT = 8002
```

## 使用方式

### Server

启动xtts-api-server，在xtts-api-server安装文件夹下运行命令

```
python -m xtts_api_server --deepspeed -p 8003 --listen
```

启动服务端程序

```
conda activate VisionRide
nohup python -u server.py >> server.log 2>&1 &
```

### Client

启动客户端程序

```
python client.py
```

