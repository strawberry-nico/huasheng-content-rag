# 华胜内容生成与业务 RAG 底座

一个围绕华胜业务知识构建的轻量 RAG 项目。

当前这套仓库不是“只会吐文案的 Prompt 页面”，而是已经拆成了两层：

- **上层应用**：内容生成工具页
- **底层能力**：知识库切块、embedding 检索、hybrid search、统一后端接口

这意味着它后面不只能做内容生成，也可以继续扩成：

- 售前客服
- 售后客服
- 业务问答 Agent

---

## 项目定位

项目当前解决的是两个问题：

1. **让 AI 先理解华胜业务，再生成内容**
2. **把内容生成、客服问答、业务问答共用到同一套知识底座上**

所以现在的核心不是“写了多少提示词”，而是：

- 知识库怎么结构化
- 如何把知识切成可检索 chunks
- 检索结果如何稳定送给大模型

---

## 当前能力

### 1. 内容生成前端

前端是一个单文件 HTML 工具：

- [华胜内容生成工具.html](./华胜内容生成工具.html)

用户输入：

- 主题
- 视频素材/表达重点
- 发布平台
- 内容类型

然后交给统一后端生成成品内容。

### 2. 知识库构建

知识源是**本地私有 Markdown 知识库**，不随仓库上传。

通过：

- [build-knowledge.js](./build-knowledge.js)

将 Markdown 处理成：

- 本地生成的 `knowledge-data.generated.json`

当前支持两类知识进入检索层：

- 前部结构化知识包样例
- 后部原始资料层切块

### 3. 统一后端

后端入口：

- [server/app.py](./server/app.py)

当前后端能力：

- `POST /generate`：内容生成
- `POST /retrieve`：知识检索
- `POST /index/rebuild`：重建索引
- `GET /health`：健康检查
- `POST /?action=save`：保存飞书

### 4. 检索方式

当前采用的是：

- DashScope embedding
- 本地向量缓存
- dense recall + keyword + metadata 的 hybrid search

当前没有启用 rerank。

这套方案的目标不是一次做到最重，而是先把：

- 知识底座
- 检索质量
- 系统边界

跑顺。

---

## 技术结构

### 前端

- 单文件 HTML
- 同源优先请求后端
- `file://` 场景下自动回退本地 `127.0.0.1:8090`

### 后端

- FastAPI
- DashScope chat completions
- DashScope embeddings
- 飞书多维表格保存

### 检索

- `knowledge-data.generated.json`
- `.server-cache/embeddings.npz`

`.npz` 只是本地向量缓存，不是向量数据库。

当前数据规模下不需要额外引入 Qdrant / Milvus / pgvector。

---

## 目录结构

```text
.
├── README.md
├── .env.example
├── build-knowledge.js
├── 华胜内容生成工具.html
└── server
    ├── app.py
    ├── requirements.txt
    └── README.md
```

说明：

- 私有知识库 Markdown 不随仓库提交
- 生成后的 `knowledge-data.generated.json` 也不提交
- 需要使用者在本地准备知识源后自行执行构建

---

## 本地启动

### 1. 创建虚拟环境

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r server/requirements.txt
```

### 2. 生成知识 chunks

```bash
node build-knowledge.js
```

前提：

- 你本地已经放好私有知识库 Markdown
- 文件名或路径与 `build-knowledge.js` 当前约定一致，或通过环境变量指定

### 3. 配置环境变量

复制模板：

```bash
cp .env.example .env
```

填写至少这些变量：

- `DASHSCOPE_API_KEY`
- `DASHSCOPE_BASE_URL`
- `DASHSCOPE_GENERATION_MODEL`
- `DASHSCOPE_EMBEDDING_MODEL`

如果需要飞书保存，再补：

- `FEISHU_APP_ID`
- `FEISHU_APP_SECRET`
- `FEISHU_BASE_TOKEN`
- `FEISHU_TABLE_ID`

### 4. 启动服务

```bash
set -a
source .env
set +a
uvicorn server.app:app --host 0.0.0.0 --port 8090
```

### 5. 打开前端

直接打开：

- [华胜内容生成工具.html](./华胜内容生成工具.html)

---

## 知识库策略

当前知识库不是简单标签库，而是朝“统一业务底座”方向走。

优先承载的知识类型包括：

- `fact`
- `pain_point`
- `solution`
- `evidence`
- `expression_rule`
- `compliance_rule`

目标不是只服务一个内容工具，而是服务多类业务应用。

后续重点优化方向包括：

- 售前高频问答
- 售后支持知识
- 合规边界
- 素材场景知识

---

## 这个项目适合怎么展示

如果以后你要放到 GitHub 或写进简历，这个项目最值得强调的不是“做了一个 AI 页面”，而是：

### 1. 你搭了业务知识驱动的生成系统

不是单纯手写 Prompt，而是：

- 结构化私有知识库
- 自动切块
- embedding 检索
- hybrid search
- LLM 生成

### 2. 你把内容生成和客服问答的底层统一了

同一套知识底座，可以支撑：

- 自媒体内容生成
- 售前问答
- 售后问答
- 业务问答 Agent

### 3. 你做的是可演进架构

现在是：

- HTML + FastAPI + DashScope embedding + hybrid search

后面可以继续扩成：

- rerank
- 客服对话前端
- 更复杂的知识源解析
- 多业务场景 Agent

---

## 当前状态

当前版本已经完成：

- 统一后端收口
- Cloudflare Worker 路线退出主路径
- 前端去掉受众和固定 tag 强约束
- 结构化知识包进入构建链
- DashScope embedding + 生成链打通

当前仍在持续优化：

- 检索降噪
- 知识库补充
- 客服场景扩展

---

## 备注

这个仓库默认不提交：

- `.env`
- `.venv`
- `.server-cache`
- 本地临时说明文档和私有草稿

保留提交的应是：

- 代码
- 模板配置
- 知识源
- 构建脚本
- 可复用说明文档
