# IOCAPI

## For paper _API Recommendation for Novice Programmers: From Clear Expressions to Effective Results_

### 1. Preparation
1. Clone the project to your local machine.
   ```bash
    git clone https://github.com/WhiteRain2/IOCAPI.git
   ```

2. You need to obtain the port address and API KEY from your large language model service provider, create and fill in the .env file in the project root directory.
   ```env
    API_KEY=""
    BASE_URL=""
   ```
> Due to cost constraints, we do not provide the key.
> - QWen model service address: https://bailian.console.aliyun.com/
> - GPT series model address: https://platform.openai.com/playground


3. Due to GitHub storage limitations, some data files have been placed on [Google Drive](https://drive.google.com/file/d/1otyMuU6S5NJwlXqi7TXdTes2LJwGZYwA/view?usp=sharing). Please download and extract them, then place the files in the `src/iocapi/get_top_k_q/` directory of the project.

#### 2. Project Deployment

1. Clone the project to your local machine(If you don't complete it in first step).
   ```bash
    git clone https://github.com/WhiteRain2/IOCAPI.git
   ```

2. It is recommended to use [uv](https://docs.astral.sh/uv/guides/install-python/) tool for quick environment setup.
   ```bash
    uv sync
   ```

#### 3. Execution

1. Interactive Mode.
   ```bash
    python dialog.py
   ```

2. RQ Mode.
   ```bash
    python rq.py
   ```

> RQ mode allows you to select datasets in the code, this code utilizes the [APIUtils library](https://github.com/WhiteRain2/APIUtils)
