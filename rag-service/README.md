# 华胜统一后端

这是后续建议作为主后端部署的 FastAPI 服务，职责包括：

1. 读取构建好的知识 chunks
2. 用 embedding + hybrid search 返回最相关的知识片段
3. 调用 Kimi 生成内容
4. 保存飞书记录

Cloudflare Worker 后续可以保留为兼容层，也可以直接被替代。

## 模型选择

- Embedding: `BAAI/bge-m3`
- Rerank: `BAAI/bge-reranker-v2-m3`（默认关闭，按需开启）

## 推荐流程

1. 在项目根目录运行：

```bash
node build-knowledge.js
```

会生成：

- `knowledge-data.generated.js`
- `knowledge-data.generated.json`

2. 安装依赖：

```bash
pip install -r rag-service/requirements.txt
```

3. 启动服务：

```bash
uvicorn rag-service.app:app --reload --port 8090
```

如果你后续要开启 rerank，可加环境变量：

```bash
RAG_ENABLE_RERANK=true
```

## 接口

### 健康检查

`GET /health`

### 统一生成接口

`POST /generate`

也兼容：

`POST /?action=generate`

### 重新构建向量索引

`POST /index/rebuild`

### 检索知识

`POST /retrieve`

请求示例：

```json
{
  "topic": "equipment",
  "platform": "douyin",
  "brief": "凝胶贴膏设备运行实拍，重点表达自动化程度高、换型效率高、适合中试放大",
  "raw_input": "",
  "top_k": 5
}
```

说明：

- 内容生成默认召回 `top_k=3`
- 通用检索接口默认 `top_k=5`
- 当前更推荐先用 `hybrid search + recall`，待效果稳定后再开启 rerank

### 保存飞书

`POST /?action=save`

## 前端对接

前端 HTML 后续可以直接把 API 地址改到这个后端。

如果你还暂时保留 Worker，Worker 侧配置环境变量：

- `RAG_API_URL=http://127.0.0.1:8090`

之后 Worker 会优先调用这个后端里的 `/retrieve` 能力。
