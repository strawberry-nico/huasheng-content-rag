# Server

这个目录只放统一后端服务。

主入口：

- [app.py](./app.py)

依赖：

- [requirements.txt](./requirements.txt)

启动方式：

```bash
set -a
source .env
set +a
uvicorn server.app:app --host 0.0.0.0 --port 8090
```

接口：

- `GET /health`
- `POST /generate`
- `POST /retrieve`
- `POST /index/rebuild`
- `POST /?action=save`

详细项目说明看根目录：

- [README.md](../README.md)
