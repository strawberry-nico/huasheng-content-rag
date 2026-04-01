from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from FlagEmbedding import BGEM3FlagModel, FlagReranker
except ImportError as exc:  # pragma: no cover
    BGEM3FlagModel = None
    FlagReranker = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "knowledge-data.generated.json"
EMBED_CACHE = ROOT / ".rag-cache" / "embeddings.npz"
DEFAULT_GENERATION_TOP_K = 3
DEFAULT_RETRIEVE_TOP_K = 5

CONTENT_TYPE_NAMES = {
    "oral": "口播脚本",
    "title": "标题+封面文案",
    "article": "图文正文",
    "comment": "评论区置顶话术",
    "moments": "朋友圈文案",
}

COMPANY = {
    "slogan": "创百年华胜 造世界精品",
    "history": "30余年",
    "positioning": "聚焦外用制剂领域，从小试到中试到规模量产，提供全链路交钥匙服务",
    "pilotBase": "12000平米开放式共享贴剂中试基地，具备C级、D级洁净区条件",
}

TOPICS = {
    "pilot": {"label": "中试服务", "desc": "12000平开放式共享基地，从小试到量产全链路"},
    "equipment": {"label": "制药设备", "desc": "贴剂/凝胶膏剂生产线"},
    "material": {"label": "包材产品", "desc": "压花膜/TPU弹力布/外包装袋"},
    "hospital": {"label": "院内制剂", "desc": "医院药剂科备案/GMP合规/全程技术服务"},
    "strength": {"label": "企业实力", "desc": "30余年/370+药企/22%百强占比/出口20+国"},
    "onestop": {"label": "一站式采购", "desc": "设备+耗材+辅料+包材+中试"},
    "training": {"label": "人才培训", "desc": "研究生实践基地/设备操作/工艺培训"},
    "tdts": {"label": "经皮知识", "desc": "透皮给药技术/行业趋势/专业解读"},
}

PLATFORMS = {
    "douyin": {
        "label": "抖音",
        "rhythm": "极快节奏",
        "hookInterval": "每5秒",
        "sentenceLength": "每句不超过12字",
        "emotion": "极高",
        "style": "口语化短句、情绪强烈、网络热词",
        "structure": "3秒致命钩子 → 15秒痛点放大 → 10秒解决方案 → 5秒强引导",
        "tactics": "利用完播率算法，前3秒必须制造认知冲突",
        "contentTypes": ["oral", "title"],
        "wordCount": {"oral": "150-200字", "title": "15-25字"},
    },
    "shipinhao": {
        "label": "视频号",
        "rhythm": "中等节奏",
        "hookInterval": "每15秒",
        "sentenceLength": "长短句结合",
        "emotion": "中等",
        "style": "专业但不刻板，有温度，用'我们'拉近距离",
        "structure": "10秒问题引入 → 30秒深度分析 → 15秒案例佐证 → 5秒信任引导",
        "tactics": "利用社交关系链，强调'同行都在看'",
        "contentTypes": ["oral", "title", "article", "comment", "moments"],
        "wordCount": {"oral": "200-300字", "article": "400-600字", "comment": "50-80字", "moments": "100-200字", "title": "20-30字"},
    },
    "xiaohongshu": {
        "label": "小红书",
        "rhythm": "轻快节奏",
        "hookInterval": "每10秒",
        "sentenceLength": "活泼短句",
        "emotion": "中高",
        "style": "亲和、表情丰富、干货感、朋友口吻",
        "structure": "5秒好奇钩子 → 20秒干货输出 → 10秒案例 → 5秒互动引导",
        "tactics": "利用收藏心理，强调'建议收藏'、'整理好了'",
        "contentTypes": ["article", "title", "comment"],
        "wordCount": {"article": "300-500字", "comment": "40-60字", "title": "15-25字"},
    },
    "zhihu": {
        "label": "知乎",
        "rhythm": "慢节奏",
        "hookInterval": "每段落",
        "sentenceLength": "长句为主，逻辑严密",
        "emotion": "低",
        "style": "理性、数据支撑、专业术语",
        "structure": "问题引入 → 背景分析 → 深度论证 → 数据佐证 → 总结建议",
        "tactics": "利用SEO长尾流量，强调专业性和权威性",
        "contentTypes": ["article", "title", "comment"],
        "wordCount": {"article": "600-1000字", "comment": "80-120字", "title": "25-40字"},
    },
}

TOPIC_LABELS = {
    "pilot": "中试服务",
    "equipment": "制药设备",
    "material": "包材产品",
    "hospital": "院内制剂",
    "strength": "企业实力",
    "onestop": "一站式采购",
    "training": "人才培训",
    "tdts": "经皮知识",
}

PLATFORM_SCENE_HINTS = {
    "douyin": ["equipment_video", "client_visit", "industry_content"],
    "shipinhao": ["technical_talk", "base_showcase", "sales_qa"],
    "xiaohongshu": ["industry_content", "base_showcase", "technical_talk"],
    "zhihu": ["technical_talk", "industry_content", "sales_qa"],
}

BRIEF_SCENE_HINTS = {
    "equipment_video": ["运行", "运转", "实拍", "设备", "生产线", "自动化", "切片机", "涂布机"],
    "technical_talk": ["讲解", "技术", "参数", "工艺", "原理", "放大", "研发"],
    "base_showcase": ["基地", "车间", "实验室", "洁净", "平台", "全景"],
    "client_visit": ["客户", "参观", "接待", "来访", "会议室"],
    "hospital_consultation": ["院内制剂", "医院", "药剂科", "备案"],
    "sales_qa": ["能不能", "适合", "怎么选", "选型", "方案", "优势"],
    "aftersales_qa": ["维保", "售后", "培训", "驻厂", "陪产", "维护"],
    "industry_content": ["趋势", "赛道", "政策", "行业", "市场", "认知"],
}

DOWNRANK_PATTERNS = [
    "目标受众",
    "受众说话方式",
    "实施路径",
    "短期目标",
    "中期目标",
    "长期愿景",
    "下一步行动清单",
    "高效内容模板",
]


class GeneratePayload(BaseModel):
    topic: str = ""
    brief: str = ""
    platform: str = ""
    types: list[str] = Field(default_factory=list)
    content: str = ""
    rawInput: str = ""
    prompt: str = ""


class RetrievePayload(BaseModel):
    topic: str = ""
    platform: str = ""
    brief: str = ""
    rawInput: str = ""
    top_k: int = DEFAULT_RETRIEVE_TOP_K


class RetrievedChunk(BaseModel):
    id: str
    title: str
    headingPath: list[str]
    content: str
    topics: list[str]
    businessStages: list[str]
    scenes: list[str]
    knowledgeTypes: list[str]
    publicLevel: str
    keywords: list[str]
    denseScore: float
    hybridScore: float
    rerankScore: float


@dataclass
class LoadedChunk:
    raw: dict[str, Any]
    text: str


class RAGEngine:
    def __init__(self) -> None:
        self.source = ""
        self.chunks: list[LoadedChunk] = []
        self.embeddings: np.ndarray | None = None
        self.embedding_model = None
        self.reranker = None
        self.ready = False

    def use_rerank(self) -> bool:
        return os.getenv("RAG_ENABLE_RERANK", "").strip().lower() in {"1", "true", "yes", "on"}

    def ensure_models(self, *, need_reranker: bool = False) -> None:
        if IMPORT_ERROR is not None:
            raise RuntimeError(
                "FlagEmbedding 未安装，无法启动 RAG 服务。请先安装 rag-service/requirements.txt"
            ) from IMPORT_ERROR
        if self.embedding_model is None:
            self.embedding_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)
        if need_reranker and self.reranker is None:
            self.reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=False)

    def load_chunks(self) -> None:
        if not DATA_FILE.exists():
            raise FileNotFoundError(f"未找到知识数据文件: {DATA_FILE}")

        payload = json.loads(DATA_FILE.read_text("utf-8"))
        self.source = payload.get("source", DATA_FILE.name)
        self.chunks = []
        for item in payload.get("chunks", []):
            text = " ".join(
                [
                    item.get("title", ""),
                    " ".join(item.get("headingPath", [])),
                    item.get("content", ""),
                    " ".join(item.get("keywords", [])),
                ]
            ).strip()
            self.chunks.append(LoadedChunk(raw=item, text=text))

    def build_index(self, force: bool = False) -> None:
        self.ensure_models()
        self.load_chunks()

        source_hash = str(DATA_FILE.stat().st_mtime_ns)
        if not force and EMBED_CACHE.exists():
            cached = np.load(EMBED_CACHE, allow_pickle=True)
            if cached["source_hash"].item() == source_hash:
                self.embeddings = cached["embeddings"]
                self.ready = True
                return

        texts = [chunk.text for chunk in self.chunks]
        encoded = self.embedding_model.encode(
            texts,
            batch_size=8,
            max_length=4096,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        dense = np.asarray(encoded["dense_vecs"], dtype=np.float32)
        dense = self._normalize(dense)
        EMBED_CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(EMBED_CACHE, embeddings=dense, source_hash=np.array(source_hash))
        self.embeddings = dense
        self.ready = True

    def retrieve(self, payload: RetrievePayload) -> list[dict[str, Any]]:
        if not self.ready or self.embeddings is None:
            self.build_index()

        query_text = self._build_query_text(payload)
        query_vec = self.embedding_model.encode(
            [query_text],
            batch_size=1,
            max_length=2048,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )["dense_vecs"][0]
        query_vec = self._normalize(np.asarray(query_vec, dtype=np.float32).reshape(1, -1))[0]
        inferred_scenes = self._infer_scenes(query_text.lower(), payload.platform)
        dense_scores = self.embeddings @ query_vec

        candidates: list[tuple[int, float, float]] = []
        for idx, chunk in enumerate(self.chunks):
            hybrid = float(dense_scores[idx]) + self._keyword_score(chunk.raw, query_text.lower()) + self._meta_score(
                chunk.raw, payload.topic, inferred_scenes
            )
            if hybrid <= 0:
                continue
            candidates.append((idx, float(dense_scores[idx]), hybrid))

        candidates.sort(key=lambda item: item[2], reverse=True)
        shortlist = candidates[: max(payload.top_k * 4, 12)]
        if not shortlist:
            return []

        rerank_enabled = self.use_rerank()
        rerank_scores: list[float]
        if rerank_enabled:
            self.ensure_models(need_reranker=True)
            pairs = [[query_text, self.chunks[idx].text] for idx, _, _ in shortlist]
            rerank_scores = [float(score) for score in self.reranker.compute_score(pairs, normalize=True)]
        else:
            rerank_scores = [float(hybrid_score) for _, _, hybrid_score in shortlist]

        enriched: list[dict[str, Any]] = []
        for (idx, dense_score, hybrid_score), rerank_score in zip(shortlist, rerank_scores):
            raw = dict(self.chunks[idx].raw)
            raw["denseScore"] = round(float(dense_score), 6)
            raw["hybridScore"] = round(float(hybrid_score), 6)
            raw["rerankScore"] = round(float(rerank_score), 6)
            enriched.append(raw)

        enriched.sort(key=lambda item: item["rerankScore"], reverse=True)
        return enriched[: payload.top_k]

    def _build_query_text(self, payload: RetrievePayload) -> str:
        topic_label = TOPIC_LABELS.get(payload.topic, payload.topic or "")
        return " ".join(part for part in [topic_label, payload.platform, payload.brief, payload.rawInput] if part).strip()

    def _infer_scenes(self, query_text: str, platform: str) -> set[str]:
        scenes = set(PLATFORM_SCENE_HINTS.get(platform, []))
        for scene, keywords in BRIEF_SCENE_HINTS.items():
            if any(keyword in query_text for keyword in keywords):
                scenes.add(scene)
        return scenes

    def _keyword_score(self, chunk: dict[str, Any], query_text: str) -> float:
        haystack = " ".join(
            [
                chunk.get("title", ""),
                " ".join(chunk.get("headingPath", [])),
                chunk.get("content", ""),
                " ".join(chunk.get("keywords", [])),
            ]
        ).lower()
        tokens = tokenize(query_text)
        return sum(0.4 for token in tokens if token in haystack)

    def _meta_score(self, chunk: dict[str, Any], topic: str, inferred_scenes: set[str]) -> float:
        score = 0.0
        if topic and topic in chunk.get("topics", []):
            score += 1.6
        score += 0.8 * len(inferred_scenes.intersection(set(chunk.get("scenes", []))))
        if chunk.get("publicLevel") == "public":
            score += 0.1
        heading_text = " ".join(chunk.get("headingPath", []))
        if any(pattern in heading_text for pattern in DOWNRANK_PATTERNS):
            score -= 1.8
        if chunk.get("title") == "五类目标受众说话方式":
            score -= 3.5
        return score

    def _normalize(self, matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
        return matrix / norms


engine = RAGEngine()
app = FastAPI(title="Huasheng Unified Backend", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event() -> None:
    try:
        engine.build_index(force=False)
    except Exception as exc:  # pragma: no cover
        print(f"[backend] startup skipped: {exc}")


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "ready": engine.ready,
        "source": engine.source,
        "chunk_count": len(engine.chunks),
        "rag_cache": str(EMBED_CACHE),
        "rerank_enabled": engine.use_rerank(),
    }


@app.post("/")
async def root_post(payload: GeneratePayload, action: str = Query(default="generate")) -> dict[str, Any]:
    if action == "save":
        if not payload.content.strip():
            raise HTTPException(status_code=400, detail="缺少可保存内容")
        feishu = await save_to_feishu(payload)
        return {"success": True, "feishu": feishu}

    normalized = normalize_generate_payload(payload)
    validate_generate_payload(normalized)
    retrieved = engine.retrieve(
        RetrievePayload(
            topic=normalized.topic,
            platform=normalized.platform,
            brief=normalized.brief,
            rawInput=normalized.rawInput,
            top_k=DEFAULT_GENERATION_TOP_K,
        )
    )
    prompt = build_prompt(normalized, retrieved, engine.source)
    content = await generate_with_kimi(prompt)

    feishu = {"saved": False}
    if has_feishu_config():
        try:
            feishu = await save_to_feishu(
                GeneratePayload(
                    **normalized.model_dump(),
                    content=content,
                    prompt=prompt,
                )
            )
        except Exception as exc:  # pragma: no cover
            feishu = {"saved": False, "error": str(exc)}

    return {
        "success": True,
        "content": content,
        "promptPreview": prompt[:500],
        "knowledgeSource": engine.source,
        "feishu": feishu,
    }


@app.post("/generate")
async def generate(payload: GeneratePayload) -> dict[str, Any]:
    return await root_post(payload, action="generate")


@app.post("/retrieve")
def retrieve(payload: RetrievePayload) -> dict[str, Any]:
    results = engine.retrieve(payload)
    return {"source": engine.source, "total_chunks": len(engine.chunks), "results": [RetrievedChunk(**item) for item in results]}


@app.post("/index/rebuild")
def rebuild_index() -> dict[str, Any]:
    engine.build_index(force=True)
    return {"ok": True, "source": engine.source, "chunk_count": len(engine.chunks)}


def normalize_generate_payload(payload: GeneratePayload) -> GeneratePayload:
    return GeneratePayload(
        topic=payload.topic.strip(),
        brief=payload.brief.strip(),
        platform=payload.platform.strip(),
        types=[item for item in payload.types if isinstance(item, str)],
        content=payload.content,
        rawInput=payload.rawInput.strip(),
        prompt=payload.prompt,
    )


def validate_generate_payload(payload: GeneratePayload) -> None:
    if payload.topic not in TOPICS:
        raise HTTPException(status_code=400, detail="主题参数无效")
    if not payload.brief:
        raise HTTPException(status_code=400, detail="请填写视频素材或表达重点")
    if payload.platform not in PLATFORMS:
        raise HTTPException(status_code=400, detail="平台参数无效")
    if not payload.types:
        raise HTTPException(status_code=400, detail="至少选择一种内容类型")

    allowed_types = set(PLATFORMS[payload.platform]["contentTypes"])
    for item in payload.types:
        if item not in allowed_types:
            raise HTTPException(status_code=400, detail=f"内容类型不支持当前平台: {item}")


def build_prompt(payload: GeneratePayload, knowledge_items: list[dict[str, Any]], knowledge_source: str) -> str:
    topic = TOPICS[payload.topic]
    platform = PLATFORMS[payload.platform]
    selected_types = "、".join(CONTENT_TYPE_NAMES.get(item, item) for item in payload.types)
    supplemental = f"【补充补充说明】\n{payload.rawInput}\n" if payload.rawInput else ""
    scene_prompt = build_scene_prompt(payload.topic)
    knowledge_section = "\n\n".join(
        f"资料{idx + 1}｜{item['title']}\n路径：{' > '.join(item.get('headingPath', []))}\n相关度：{item.get('rerankScore', 0)}\n{item['content']}"
        for idx, item in enumerate(knowledge_items)
    )

    return f"""【角色设定】
你是华胜品牌内容团队一员，代表华胜对外发声。华胜始创于1993年，{COMPANY['history']}持续深耕外用制剂领域，业务覆盖设备、中试转化、工艺放大与配套服务。中试基地是华胜能力体系中的关键模块之一，但不等于华胜全部业务。

【华胜背景资料】（按相关性自然融入，不要每条都机械堆砌）
- 品牌口号：{COMPANY['slogan']}
- 品牌积累：始创于1993年，{COMPANY['history']}持续深耕外用制剂领域
- 中试基地：{COMPANY['pilotBase']}
- 品牌定位：{COMPANY['positioning']}

【内容生成参数】
主题：{topic['label']}
视频素材/表达重点：{payload.brief}
平台：{platform['label']}
内容类型：{selected_types}

【品牌层级要求】
- 默认主体是“华胜”这一整体品牌，不要默认把华胜写成只有中试基地
- 当素材明确展示基地、车间、实验室、中试服务时，可以突出“中试基地”这一能力模块
- 当素材更偏设备、包材、院内制剂、行业认知或一站式服务时，应回到华胜整体解决方案视角
- 可以写“华胜具备中试转化能力”，不要写成“华胜只做中试”

【平台风格要求 - 必须严格执行】
- 节奏：{platform['rhythm']}，{platform['hookInterval']}必须有一个钩子
- 句式：{platform['sentenceLength']}
- 情绪强度：{platform['emotion']}
- 语言风格：{platform['style']}
- 结构模板：{platform['structure']}
- 平台策略：{platform['tactics']}

【总口吻原则】
- 优先服从当前平台的表达习惯，但整体保持专业、可信、克制的B端品牌气质
- 可以更口语化，但不要油腻、浮夸、喊口号、过度像泛流量营销号
- 可以有冲突感和传播感，但结论要落在专业判断、业务价值或解决方案上
- 不要为了迎合平台牺牲真实性和专业度

【字数要求】
{chr(10).join(f"- {CONTENT_TYPE_NAMES.get(key, key)}：{value}" for key, value in platform['wordCount'].items())}

【素材理解要求】
- 先根据“视频素材/表达重点”判断当前内容更适合讲设备展示、工艺能力、基地实力、案例背书还是行业认知
- 主题只用于限定业务范围，不代表固定标签、固定卖点或固定开场
- 优先围绕素材真实内容组织表达，不要被固定卖点带偏
- 同一条内容最多突出1-2个核心卖点
- 没有明确数据来源，不要虚构具体金额、百分比、节省天数
- 没有检索证据支撑时，不要主动补充具体设备能力、洁净等级、清洗效率或成本节省数据
- 如果素材更适合讲专业性、稳定性、合规性、自动化，就不要硬转成“省钱文案”

【开场要求】
- 开场必须从当前素材或表达重点中提炼，不要套预设钩子库
- 可以用问题、反常识、场景切入、结果切入，但必须和当前素材直接相关

{supplemental}【动态知识库召回】
以下资料来自华胜知识库（来源：{knowledge_source}），请优先使用与当前主题、素材描述、平台最相关的证据，不要为了凑数字而全部堆砌：
{knowledge_section}

【知识使用规则】
- 优先使用上方已召回资料中的事实、场景和表述
- 若召回资料与硬编码背景资料冲突，以召回资料为准
- 若召回资料没有覆盖某个数字、认证级别、设备名称或能力点，就不要自行补全
- 引用数字时，优先解释业务意义，不要孤立堆数字

【本次主题子提示词】
{scene_prompt}

【立场要求】
- 不说"我是专家"，说"华胜30余年经验发现..."
- 不提具体技术团队成员姓名
- 可以说"与国内顶尖研究机构共建"
- 用"我们"代表华胜团队，不用"我"

【品牌用词规范】
- 必用："30余年"、"全链路"、"交钥匙工程"、"一站式"
- 禁用："最好"、"第一"、"最强"
- 合规：不替客户做疗效声称，不说具体配方工艺
- 所有数字都要尽量解释业务意义，例如节省的不是抽象成本，而是清洗时间、人力、换型或交付效率

【成品输出要求】
- 输出必须像可直接发布的成品，不要复述任务，不要解释写作思路，不要写“按要求生成”之类提示词痕迹
- 不要把时间结构直接写成标题
- 先写痛点，再给证据，再给价值，再给行动，不要一上来堆品牌荣誉
- 每个模块都要体现与素材和场景相关的关注点，而不是默认套受众模板
- 即使是抖音、小红书等平台，也要避免低质营销腔和过度夸张表达

【内容类型具体要求】
{build_content_requirements(platform, payload.types)}

【输出格式】
严格按以下内容类型输出，每个模块用【】标注：
{chr(10).join(f"【{CONTENT_TYPE_NAMES.get(item, item)}】" for item in payload.types)}"""


def build_scene_prompt(topic_id: str) -> str:
    prompts = {
        "pilot": """- 本条内容以“中试转化能力”作为核心切口
- 重点讲中试如何连接研发与产业化，不只讲场地和设备
- 优先突出工艺放大、验证衔接、转移效率、项目承接能力
- 表达主体可以是“华胜的中试基地/中试平台”，但结论仍落回华胜整体能力""",
        "equipment": """- 本条内容以“设备如何服务工艺放大和稳定量产”作为核心切口
- 不只罗列设备参数，要讲换型效率、验证友好、质量一致性和后续产业化价值
- 若素材不涉及清洗、换型或GMP清洁要求，不要主动展开在线清洗能力
- 表达主体优先是华胜设备能力，而不是把内容全部写成基地介绍""",
        "material": """- 本条内容以“包材/耗材如何影响制剂稳定性和量产一致性”作为核心切口
- 重点讲适配性、配套性、稳定性和与工艺协同的价值
- 不要把包材内容硬写成设备介绍或中试基地宣传""",
        "hospital": """- 本条内容以“院内制剂开发与备案落地支持”作为核心切口
- 重点讲备案配套、工艺验证、稳定制备、合规支持和协同服务
- 表达主体应是华胜的院内制剂服务能力，不要弱化为单纯设备销售""",
        "strength": """- 本条内容以“华胜整体实力与长期积累”作为核心切口
- 重点讲长期深耕、技术沉淀、标准化能力、整体解决方案和产业协同
- 不要默认堆客户数、出口数、专利数等高波动数字，除非召回资料明确支持""",
        "onestop": """- 本条内容以“整体解决方案协同价值”作为核心切口
- 重点讲设备、包材、中试、工艺与服务协同如何减少沟通损耗、缩短周期、降低系统性风险
- 避免把一站式理解成简单的产品打包销售""",
        "training": """- 本条内容以“人才培训与能力转移”作为核心切口
- 重点讲实操培训、工艺理解、上手效率和团队能力建设
- 不要把培训内容写成泛泛企业宣传""",
        "tdts": """- 本条内容以“行业认知 + 产业落地判断”作为核心切口
- 先建立行业认知，再落到中试转化、设备适配、产业落地或合规挑战
- 不要只谈趋势概念，最终要回到华胜能解决什么问题""",
    }
    return prompts.get(
        topic_id,
        """- 以华胜整体品牌视角组织内容
- 根据素材判断更适合突出设备能力、中试转化、工艺理解还是整体解决方案
- 不要把单一能力模块误写成华胜全部业务""",
    )


def build_content_requirements(platform: dict[str, Any], types: list[str]) -> str:
    requirements: list[str] = []
    for item in types:
        if item == "oral":
            requirements.append(
                f"""【口播脚本 - {platform['label']}版】
- 字数：{platform['wordCount']['oral']}
- 节奏：{platform['rhythm']}，{platform['hookInterval']}一个转折点
- 开头：必须从当前素材或表达重点里提炼一个强切口
- 中间：痛点放大+华胜解决方案，优先植入与召回资料一致的华胜背景和证据
- 结尾：围绕当前场景做自然引导，不要写成固定受众分层话术
- 句式：{platform['sentenceLength']}"""
            )
        elif item == "title":
            requirements.append(
                f"""【标题+封面文案 - {platform['label']}版】
- 数量：3个备选标题 + 1条封面文案
- 标题字数：{platform['wordCount']['title']}
- 标题要求：{get_title_requirement(payload_platform_id(platform))}
- 封面文案：10-15字，强调冲突感或悬念"""
            )
        elif item == "article":
            requirements.append(
                f"""【图文正文 - {platform['label']}版】
- 字数：{platform['wordCount']['article']}
- 结构：{platform['structure']}
- 格式：{get_article_format(payload_platform_id(platform))}
- 植入：自然融入与召回资料一致的华胜背景，如30余年、基地能力、客户基础等"""
            )
        elif item == "comment":
            requirements.append(
                f"""【评论区置顶话术 - {platform['label']}版】
- 字数：{platform['wordCount']['comment']}
- 风格：{get_comment_style(payload_platform_id(platform))}
- 类型：资料引流型/互动争议型/参观预约型"""
            )
        elif item == "moments":
            requirements.append(
                f"""【朋友圈文案 - {platform['label']}版】
- 字数：{platform['wordCount']['moments']}
- 结构：场景引入 → 反常识观点 → 华胜植入 → 分层引导
- 人设：华胜团队一员，专业但不高冷"""
            )
    return "\n\n".join(requirements)


def payload_platform_id(platform: dict[str, Any]) -> str:
    for key, value in PLATFORMS.items():
        if value is platform:
            return key
    return ""


def get_title_requirement(platform_id: str) -> str:
    if platform_id == "douyin":
        return '数字+冲突+情绪词，例如"90%死在这"、"血亏300万"'
    if platform_id == "xiaohongshu":
        return "emoji+干货词+收藏引导"
    if platform_id == "zhihu":
        return "专业术语+完整问题，SEO友好"
    return "专业洞察+价值承诺"


def get_article_format(platform_id: str) -> str:
    if platform_id == "xiaohongshu":
        return "多用 emoji 分段，清单式排版"
    if platform_id == "zhihu":
        return "标题层级清楚，必要时用数据和来源增强可信度"
    return "段落清晰，每段不超过3-5行"


def get_comment_style(platform_id: str) -> str:
    if platform_id == "douyin":
        return '口语化、紧迫感，适合"评论区扣XX"'
    if platform_id == "xiaohongshu":
        return '亲和互动，适合"整理好了"'
    if platform_id == "zhihu":
        return "专业、价值延续、深度引导"
    return "有温度、建立信任"


async def generate_with_kimi(prompt: str) -> str:
    api_key = os.getenv("KIMI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="KIMI_API_KEY 未配置")

    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            "https://api.moonshot.cn/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            json={
                "model": os.getenv("KIMI_MODEL", "moonshot-v1-32k"),
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.85,
            },
        )
        if response.status_code >= 400:
            try:
                detail = response.json()
            except Exception:
                detail = response.text
            raise HTTPException(status_code=500, detail=f"Kimi 请求失败: {detail}")
        payload = response.json()
        return payload["choices"][0]["message"]["content"]


async def save_to_feishu(payload: GeneratePayload) -> dict[str, Any]:
    if not has_feishu_config():
        return {"saved": False}

    app_id = os.getenv("FEISHU_APP_ID", "")
    app_secret = os.getenv("FEISHU_APP_SECRET", "")
    base_token = os.getenv("FEISHU_BASE_TOKEN", "")
    table_id = os.getenv("FEISHU_TABLE_ID", "")

    async with httpx.AsyncClient(timeout=60) as client:
        token_res = await client.post(
            "https://open.feishu.cn/open-apis/auth/v3/app_access_token/internal",
            json={"app_id": app_id, "app_secret": app_secret},
        )
        token_data = token_res.json()
        if token_data.get("code") != 0:
            raise HTTPException(status_code=500, detail=f"获取飞书token失败: {token_data.get('msg')}")

        access_token = token_data["app_access_token"]
        fields = {
            "主题": map_topic(payload.topic),
            "受众": "",
            "平台": map_platform(payload.platform),
            "内容类型": "、".join(map_content_type(item) for item in payload.types),
            "原始需求": "\n\n".join(filter(None, [payload.brief, payload.rawInput])),
            "生成内容": payload.content,
            "提示词": payload.prompt[:500] if payload.prompt else "",
            "生成时间": int(__import__("time").time() * 1000),
            "状态": "已生成",
        }
        record_res = await client.post(
            f"https://open.feishu.cn/open-apis/bitable/v1/apps/{base_token}/tables/{table_id}/records",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
            json={"fields": fields},
        )
        record_data = record_res.json()
        if record_data.get("code") != 0:
            raise HTTPException(status_code=500, detail=f"飞书API错误: {record_data}")
        return {
            "saved": True,
            "recordId": record_data.get("data", {}).get("record", {}).get("record_id"),
        }


def has_feishu_config() -> bool:
    return all(
        [
            os.getenv("FEISHU_APP_ID"),
            os.getenv("FEISHU_APP_SECRET"),
            os.getenv("FEISHU_BASE_TOKEN"),
            os.getenv("FEISHU_TABLE_ID"),
        ]
    )


def map_topic(topic: str) -> str:
    mapping = {
        "pilot": "🏭 中试服务",
        "equipment": "⚙️ 制药设备",
        "material": "📦 包材产品",
        "hospital": "🏥 院内制剂",
        "strength": "⭐ 企业实力",
        "onestop": "🛒 一站式采购",
        "training": "🎓 人才培训",
        "tdts": "📚 经皮知识",
    }
    return mapping.get(topic, topic)


def map_platform(platform: str) -> str:
    return {"douyin": "抖音", "shipinhao": "视频号", "xiaohongshu": "小红书", "zhihu": "知乎"}.get(platform, platform)


def map_content_type(content_type: str) -> str:
    return CONTENT_TYPE_NAMES.get(content_type, content_type)


def tokenize(text: str) -> list[str]:
    return list({m.group(0).lower() for m in re.finditer(r"[\u4e00-\u9fa5]{2,12}|[a-z0-9-]{2,}", text)})
