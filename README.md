# 华胜 Content Agent

一个面向 `To B 医药制造场景` 的内容生成 Agent。

它不是单纯的 Prompt 页面，也不是把知识库塞给模型就结束的 Demo，而是一套围绕华胜业务搭建的 `业务知识底座 + 检索增强 + 内容生成 Agent`。

当前版本已经可以稳定完成：

- 院内制剂、设备、中试平台等主题的内容生产
- 不同平台内容风格适配
- 基于私有业务知识的事实兜底与表达约束
- 检索、生成、反馈的统一后端闭环

---

## 项目一句话

> 用华胜自己的业务知识，驱动一个可控、可解释、可扩展的 `To B 内容生成 Agent`。

---

## 为什么不是普通 RAG Demo

这套项目和常见“上传 PDF -> 问答/写文案”的 Demo 不一样，重点不在模型炫技，而在业务落地：

- 知识不是一坨原文，而是按业务结构切成可检索 chunk
- 检索不是只看向量相似度，还叠加关键词、metadata、场景边界与轻量 rerank
- 生成不是直接把用户输入丢给模型，而是先走 `intent -> retrieve -> generate`
- 输出不是泛化营销文案，而是 `华胜市场/销售可用的 To B 自媒体内容初稿`

换句话说，这个项目解决的是：

1. `怎么让模型先理解华胜业务，再说话`
2. `怎么让内容输出像业务人员，而不是像 AI`
3. `怎么在能用的前提下，尽量少瞎说`

---

## 当前产品定义

当前这只 Agent 的产品边界非常明确：

- 它是 `内容生成 Agent`
- 不是通用聊天助手
- 不是备案顾问
- 不是万能营销机器人

它当前最适合的角色是：

- 华胜市场
- 华胜销售
- 懂业务的一线负责人

输出目标是：

- 口播脚本
- 图文正文
- 标题 / 封面文案
- 评论区置顶话术
- 朋友圈文案

项目边界说明见：

- [agent-产品边界说明.md](./agent-产品边界说明.md)

---

## 核心能力

### 1. 内容生成 Agent

当前主入口已经不是单纯 `/generate`，而是：

- `POST /agent/generate`

这一层会先做 Agent 规划，再调用底层生成链。

当前 Agent 会显式输出：

- `agentMode`
- `retrievalMode`
- `riskMode`
- `outputGoal`
- `planningNotes`

也就是说，它已经不是“黑盒直接吐文案”，而是一个会先判断任务类型和风险等级，再决定如何检索与生成的轻量 Agent。

### 2. 显式生成链

当前内容生成链已经拆成三段：

- `intent`
- `retrieve`
- `generate`

这使系统具备三个非常关键的能力：

- 可解释：能知道是理解错了，还是检索错了，还是生成错了
- 可定位：能知道问题大致出在哪一层
- 可扩展：未来换 rerank、rewrite 或多工具路由时，不需要推翻整个系统

### 3. 业务知识底座

知识源并不是单一文档，而是私有业务资料与业务沉淀的组合。

这些私有知识不会随仓库公开。

当前知识切块已不是简单文档分段，而是带有业务属性的 chunk，例如：

- `document`
- `conversation_insight`
- `conversation_evidence`

并带有多维 metadata，例如：

- `topic`
- `scene`
- `knowledgeType`
- `businessScenario`
- `executionSite`
- `pilotContext`
- `conceptType`

这使它比普通 RAG 更接近真实业务知识系统，而不是“向量化资料箱”。

### 4. 检索增强

当前检索方式是：

- DashScope embeddings
- 本地 embedding 缓存
- dense recall
- keyword score
- metadata score
- rule-based rerank

重点不是堆最重的基础设施，而是让当前规模下的检索结果：

- 更可控
- 更可解释
- 更接近业务直觉

同时系统已经补了几类关键约束：

- `设备` 与 `中试` 的业务边界
- `试机` 与 `中试` 的概念区分
- `院内制剂` 场景下的事实保守表达
- 口播中对缩写、专业词、夸张表述的压制

### 5. 反馈与可观测性

当前不是“生成完看运气”，而是已经带有基础可观测性：

- 检索 trace
- score breakdown
- intent trace
- agent trace
- usage events 日志
- 飞书保存反馈

也就是说，这个项目已经具备一个很关键的工程特征：

> `能定位问题，而不是只能感觉问题。`

---

## 前端体验

当前前端仍然保持轻量，但已经够支撑真实测试和业务试用。

主要页面：

- [华胜内容生成工具.html](./华胜内容生成工具.html)

其中主页面已经接入 Agent 生成链，并展示：

- 本次任务理解
- Agent 策略
- 知识来源
- 飞书保存

## 技术栈

### 前端

- 单文件 HTML
- 原生 JS
- 本地直连 FastAPI

### 后端

- FastAPI
- DashScope Chat Completions
- DashScope Embeddings
- 飞书多维表格保存

### 数据与检索

- 本地知识构建产物
- 本地 embedding 缓存
- hybrid retrieval
- rule-based rerank

当前没有引入额外向量数据库，也没有拆复杂微服务。

这是刻意的：

- 当前知识规模下，本地缓存已够用
- 系统当前主要瓶颈在业务边界与表达控制，不在 infra
- 先把业务 Agent 跑顺，比先堆基础设施更重要

---

## 项目结构

```text
.
├── README.md
├── agent-产品边界说明.md
├── build-knowledge.js
├── 华胜内容生成工具.html
└── server/
    ├── app.py
    ├── requirements.txt
    └── README.md
```

关键文件：

- [build-knowledge.js](./build-knowledge.js)：知识构建与 metadata 生成
- [server/app.py](./server/app.py)：统一后端、检索逻辑、Agent 生成链
- [华胜内容生成工具.html](./华胜内容生成工具.html)：主内容生产前端

---

## 本地启动

### 1. 创建虚拟环境

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r server/requirements.txt
```

### 2. 构建知识数据

```bash
node build-knowledge.js
```

### 3. 配置环境变量

```bash
cp .env.example .env
```

至少需要：

- `DASHSCOPE_API_KEY`
- `DASHSCOPE_BASE_URL`
- `DASHSCOPE_GENERATION_MODEL`
- `DASHSCOPE_EMBEDDING_MODEL`

如果要保存飞书，再补：

- `FEISHU_APP_ID`
- `FEISHU_APP_SECRET`
- `FEISHU_BASE_TOKEN`
- `FEISHU_TABLE_ID`

### 4. 启动后端

```bash
source .venv/bin/activate
set -a
source .env
set +a
python -m uvicorn server.app:app --host 0.0.0.0 --port 8090
```

### 5. 打开前端

直接打开：

- [华胜内容生成工具.html](./华胜内容生成工具.html)

---

## 当前阶段完成了什么

当前版本已经完成：

- 统一后端收口
- 多源知识构建链
- DashScope embedding + 生成链打通
- hybrid retrieval
- 轻量 rerank / rewrite hook
- `intent -> retrieve -> generate` 分层
- `/agent/generate` 轻量内容生成 Agent
- 主页面接入 Agent 模式
- trace / scoreBreakdown / usage events 可观测性
- 多轮测试后的表达边界与事实约束收紧

---

## 当前最有价值的亮点

如果把这个项目放到 GitHub、简历或者答辩里，最值得讲的不是“做了一个 AI 内容页”，而是下面这几件事：

### 1. 你做的是业务知识驱动的生成系统

不是裸 Prompt，不是模板拼接，而是：

- 多源知识构建
- embedding 检索
- hybrid search
- Agent 路由
- 内容生成

### 2. 你把“能生成”做成了“能定位”

大多数 Demo 到“生成文本”就结束了。

这套项目往前多做了一步：

- 能看 intent
- 能看 retrieval trace
- 能看 score breakdown
- 能看 agent plan

这让它更像一个可持续打磨的系统，而不是一次性作品。

### 3. 你解决的是一个真实 To B 场景

不是泛内容写作，不是情绪陪伴，也不是开放问答。

它服务的是：

- 医药制造设备
- 中试平台
- 院内制剂
- To B 业务表达

这使它天然比通用文案工具更有业务壁垒。

---

## 已知边界

当前系统仍然有明确边界，项目里也没有回避这些问题：

- 院内制剂与备案支持内容仍是最高风险主题
- 案例、数字、参数类内容仍需要强约束
- 目前更适合 `AI 初稿 + 人工确认`，而不是直接自动发布
- 当前不建议扩成开放式聊天产品
- 当前不建议加入长期会话记忆

这不是缺点掩盖，而是当前版本刻意保持的产品边界。

---

## 下一步最值得做什么

后续优先级建议按下面顺序继续推进：

1. 继续收紧高风险主题
   - 院内制剂备案支持
   - 中试平台价值

2. 继续补可公开知识
   - 可宣传案例
   - 统一术语口径
   - 高风险事实白名单

3. 再考虑扩产品形态
   - 编辑型多轮
   - 更细粒度角色控制
   - 更成熟的前端工作台

不是现在优先做：

- 开放式对话
- 长期记忆
- 重型基础设施迁移

---

## 备注

这个仓库默认不提交：

- `.env`
- `.venv`
- `.server-cache`
- 本地私有测试产物

应该保留提交的内容包括：

- 代码
- 构建脚本
- 配置模板
- 产品边界与使用说明
