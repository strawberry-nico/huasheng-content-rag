from __future__ import annotations

import json
import os
import re
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "knowledge-data.generated.json"
EMBED_CACHE = ROOT / ".server-cache" / "embeddings.npz"
EVENT_LOG_FILE = ROOT / ".server-cache" / "usage-events.jsonl"
DEFAULT_GENERATION_TOP_K = 3
DEFAULT_RETRIEVE_TOP_K = 5
DEFAULT_EMBED_BATCH_SIZE = 32
DEFAULT_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

CONTENT_TYPE_NAMES = {
    "oral": "口播脚本",
    "title": "标题+封面文案",
    "article": "图文正文",
    "comment": "评论区置顶话术",
    "moments": "朋友圈文案",
}

TOPICS = {
    "pilot": {"label": "中试服务"},
    "equipment": {"label": "制药设备"},
    "material": {"label": "包材产品"},
    "hospital": {"label": "院内制剂"},
    "strength": {"label": "企业实力"},
    "onestop": {"label": "一站式采购"},
    "training": {"label": "人才培训"},
    "tdts": {"label": "经皮知识"},
}

PLATFORMS = {
    "douyin": {
        "label": "抖音",
        "rhythm": "极快节奏",
        "hookInterval": "每5秒",
        "sentenceLength": "每句不超过12字",
        "emotion": "极高",
        "style": "口语化短句、情绪强烈、网络热词",
        "structure": "快速切入 → 重点展开 → 自然收住",
        "tactics": "前段尽快进入重点，不要为了冲突硬拔高",
        "contentTypes": ["oral", "title"],
        "wordCount": {"oral": "80-140字", "title": "15-25字"},
    },
    "shipinhao": {
        "label": "视频号",
        "rhythm": "中等节奏",
        "hookInterval": "每15秒",
        "sentenceLength": "长短句结合",
        "emotion": "中等",
        "style": "专业但不刻板，有温度，用'我们'拉近距离",
        "structure": "问题引入 → 重点说明 → 自然收束",
        "tactics": "优先可信和自然，不要硬做背书感",
        "contentTypes": ["oral", "title", "article", "comment", "moments"],
        "wordCount": {"oral": "120-180字", "article": "400-600字", "comment": "50-80字", "moments": "100-200字", "title": "20-30字"},
    },
    "xiaohongshu": {
        "label": "小红书",
        "rhythm": "轻快节奏",
        "hookInterval": "每10秒",
        "sentenceLength": "活泼短句",
        "emotion": "中高",
        "style": "亲和、表情丰富、干货感、朋友口吻",
        "structure": "轻快切入 → 清楚展开 → 自然收束",
        "tactics": "重点是好读、清楚，不要硬做收藏诱导",
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
        "tactics": "重点是讲清逻辑，不要堆权威感和术语感",
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


class AgentGeneratePayload(GeneratePayload):
    agentGoal: str = ""


class RetrievePayload(BaseModel):
    topic: str = ""
    platform: str = ""
    brief: str = ""
    rawInput: str = ""
    top_k: int = DEFAULT_RETRIEVE_TOP_K


class ChunkDebugResponse(BaseModel):
    source: str
    total_chunks: int
    filtered_chunks: int
    items: list[dict[str, Any]]


class RetrievedChunk(BaseModel):
    id: str
    title: str
    headingPath: list[str]
    content: str
    triggerCondition: str = ""
    usageRule: str = ""
    sourceModule: str = ""
    confidence: str = ""
    topics: list[str]
    businessStages: list[str]
    scenes: list[str]
    knowledgeTypes: list[str]
    publicLevel: str
    keywords: list[str]
    sourceType: str = ""
    sourceFile: str = ""
    sourceFormat: str = ""
    sourceLabel: str = ""
    sourceTimeRange: str = ""
    conceptType: str = ""
    executionSite: str = ""
    pilotContext: str = ""
    businessScenario: str = ""
    conversationSpeakers: list[str] = Field(default_factory=list)
    conversationGroupId: str = ""
    statementType: str = ""
    intentType: str = ""
    certainty: str = ""
    decisionStatus: str = ""
    denseScore: float
    hybridScore: float
    rerankScore: float = 0.0
    finalScore: float
    scoreBreakdown: dict[str, Any] = Field(default_factory=dict)


class RetrievalTrace(BaseModel):
    queryBundle: dict[str, Any]
    shortlistSize: int
    returnedCount: int


class GenerateIntentSummary(BaseModel):
    intentType: str
    generationMode: str
    inputCompleteness: str
    focus: str
    negativeConstraints: str
    shouldRetrieve: bool


class GenerateExecutionTrace(BaseModel):
    intent: GenerateIntentSummary
    queryBundle: dict[str, Any]
    retrievalTrace: RetrievalTrace


class AgentPlan(BaseModel):
    agentMode: str
    retrievalMode: str
    riskMode: str
    outputGoal: str
    planningNotes: list[str] = Field(default_factory=list)


class AgentExecutionTrace(BaseModel):
    plan: AgentPlan
    generateTrace: GenerateExecutionTrace


@dataclass
class LoadedChunk:
    raw: dict[str, Any]
    text: str


class RAGEngine:
    def __init__(self) -> None:
        self.source = ""
        self.chunks: list[LoadedChunk] = []
        self.embeddings: np.ndarray | None = None
        self.ready = False

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
                    item.get("triggerCondition", ""),
                    item.get("usageRule", ""),
                    item.get("sourceModule", ""),
                    item.get("confidence", ""),
                    item.get("sourceType", ""),
                    item.get("sourceFile", ""),
                    item.get("sourceLabel", ""),
                    item.get("sourceTimeRange", ""),
                    item.get("conceptType", ""),
                    item.get("executionSite", ""),
                    item.get("pilotContext", ""),
                    item.get("businessScenario", ""),
                    " ".join(item.get("conversationSpeakers", [])),
                    item.get("conversationGroupId", ""),
                    item.get("statementType", ""),
                    item.get("intentType", ""),
                    item.get("certainty", ""),
                    item.get("decisionStatus", ""),
                    " ".join(item.get("keywords", [])),
                ]
            ).strip()
            item.setdefault("conceptType", infer_concept_type(text))
            item.setdefault("executionSite", infer_execution_site(text))
            item.setdefault("pilotContext", infer_pilot_context(text))
            item.setdefault("businessScenario", infer_business_scenario(text))
            self.chunks.append(LoadedChunk(raw=item, text=text))

    def build_index(self, force: bool = False) -> None:
        self.load_chunks()

        source_hash = self._build_source_hash()
        if not force and EMBED_CACHE.exists():
            cached = np.load(EMBED_CACHE, allow_pickle=True)
            if cached["source_hash"].item() == source_hash:
                self.embeddings = cached["embeddings"]
                self.ready = True
                return

        texts = [chunk.text for chunk in self.chunks]
        dense = self._embed_documents(texts)
        dense = self._normalize(dense)
        EMBED_CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(EMBED_CACHE, embeddings=dense, source_hash=np.array(source_hash))
        self.embeddings = dense
        self.ready = True

    def retrieve(self, payload: RetrievePayload) -> list[dict[str, Any]]:
        results, _ = self.retrieve_with_trace(payload)
        return results

    def retrieve_with_trace(self, payload: RetrievePayload) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if not self.ready or self.embeddings is None:
            self.build_index()

        query_bundle = rewrite_query(payload)
        query_text = query_bundle["query_text"]
        query_vec = self._embed_query(query_text)
        query_vec = self._normalize(np.asarray(query_vec, dtype=np.float32).reshape(1, -1))[0]
        inferred_scenes = self._infer_scenes(query_bundle["intent_text"].lower(), payload.platform)
        dense_scores = self.embeddings @ query_vec

        candidates: list[tuple[int, float, float, float, float]] = []
        for idx, chunk in enumerate(self.chunks):
            dense_score = float(dense_scores[idx])
            keyword_score = self._keyword_score(chunk.raw, query_bundle)
            meta_score = self._meta_score(chunk.raw, payload.topic, inferred_scenes, query_bundle)
            hybrid = dense_score + keyword_score + meta_score
            if hybrid <= 0:
                continue
            candidates.append((idx, dense_score, keyword_score, meta_score, hybrid))

        candidates.sort(key=lambda item: item[4], reverse=True)
        shortlist_size = max(payload.top_k * 4, 12)
        shortlisted = candidates[:shortlist_size]
        if not shortlisted:
            return [], {
                "queryBundle": query_bundle,
                "shortlistSize": 0,
                "returnedCount": 0,
            }

        enriched: list[dict[str, Any]] = []
        for idx, dense_score, keyword_score, meta_score, hybrid_score in shortlisted:
            raw = dict(self.chunks[idx].raw)
            raw["denseScore"] = round(float(dense_score), 6)
            raw["keywordScore"] = round(float(keyword_score), 6)
            raw["metaScore"] = round(float(meta_score), 6)
            raw["hybridScore"] = round(float(hybrid_score), 6)
            rerank_score = rerank_candidates(query_bundle, raw)
            raw["rerankScore"] = round(float(rerank_score), 6)
            raw["finalScore"] = round(float(hybrid_score + rerank_score), 6)
            raw["scoreBreakdown"] = {
                "dense": raw["denseScore"],
                "keyword": raw["keywordScore"],
                "meta": raw["metaScore"],
                "hybridBase": raw["hybridScore"],
                "rerank": raw["rerankScore"],
                "final": raw["finalScore"],
            }
            enriched.append(raw)

        enriched.sort(key=lambda item: item["finalScore"], reverse=True)
        final_results = enriched[: payload.top_k]
        return final_results, {
            "queryBundle": query_bundle,
            "shortlistSize": len(shortlisted),
            "returnedCount": len(final_results),
        }

    def _build_query_bundle(self, payload: RetrievePayload) -> dict[str, Any]:
        return build_rule_based_query_bundle(payload)

    def _infer_scenes(self, query_text: str, platform: str) -> set[str]:
        scenes = set(PLATFORM_SCENE_HINTS.get(platform, []))
        for scene, keywords in BRIEF_SCENE_HINTS.items():
            if any(keyword in query_text for keyword in keywords):
                scenes.add(scene)
        return scenes

    def _keyword_score(self, chunk: dict[str, Any], query_bundle: dict[str, Any]) -> float:
        haystack = " ".join(
            [
                chunk.get("title", ""),
                " ".join(chunk.get("headingPath", [])),
                chunk.get("content", ""),
                " ".join(chunk.get("keywords", [])),
            ]
        ).lower()
        query_text = query_bundle["query_text"]
        tokens = tokenize(query_text)
        intent_tokens = query_bundle["intent_tokens"]
        negative_tokens = query_bundle["negative_tokens"]
        score = sum(0.35 for token in tokens if token in haystack)
        score += sum(0.65 for token in intent_tokens if token in haystack)
        score -= sum(0.45 for token in negative_tokens if token in haystack)
        return score

    def _meta_score(self, chunk: dict[str, Any], topic: str, inferred_scenes: set[str], query_bundle: dict[str, Any]) -> float:
        score = 0.0
        chunk_topics = set(chunk.get("topics", []))
        if topic and topic in chunk_topics:
            score += 1.6
        if topic and topic not in {"strength", "onestop"} and len(chunk_topics) >= 4:
            score -= 0.45
        score += topic_boundary_score(chunk_topics, query_bundle)
        score += 0.8 * len(inferred_scenes.intersection(set(chunk.get("scenes", []))))
        if chunk.get("publicLevel") == "public":
            score += 0.1
        source_type = chunk.get("sourceType", "")
        decision_status = chunk.get("decisionStatus", "")
        certainty = chunk.get("certainty", "")
        if decision_status == "confirmed":
            score += 0.25
        elif decision_status == "tentative":
            score -= 0.05
        if certainty == "high":
            score += 0.15
        elif certainty == "low":
            score -= 0.05
        heading_text = " ".join(chunk.get("headingPath", []))
        if any(pattern in heading_text for pattern in DOWNRANK_PATTERNS):
            score -= 1.8
        if chunk.get("title") == "五类目标受众说话方式":
            score -= 3.5
        if source_type == "conversation_evidence" and "evidence" not in chunk.get("knowledgeTypes", []):
            score -= 0.2
        score += scenario_alignment_score(chunk, query_bundle)
        return score

    def _rerank_score(self, chunk: dict[str, Any], query_bundle: dict[str, Any]) -> float:
        return rule_based_rerank_score(chunk, query_bundle)

    def _normalize(self, matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
        return matrix / norms

    def _build_source_hash(self) -> str:
        payload = f"{DATA_FILE.stat().st_mtime_ns}:{resolve_embedding_model()}:{resolve_dashscope_base_url()}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _embed_documents(self, texts: list[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        batch_size = int(os.getenv("EMBED_BATCH_SIZE", str(DEFAULT_EMBED_BATCH_SIZE)))
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            vectors.extend(request_embeddings(batch))
        return np.asarray(vectors, dtype=np.float32)

    def _embed_query(self, text: str) -> np.ndarray:
        vectors = request_embeddings([text])
        if not vectors:
            raise RuntimeError("embedding API 未返回查询向量")
        return np.asarray(vectors[0], dtype=np.float32)


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
        engine.load_chunks()
    except Exception as exc:  # pragma: no cover
        print(f"[server] startup skipped: {exc}")


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "ready": engine.ready,
        "source": engine.source,
        "chunk_count": len(engine.chunks),
        "rag_cache": str(EMBED_CACHE),
        "embedding_mode": "dashscope",
        "dashscope_base_url": resolve_dashscope_base_url(),
        "embedding_model": resolve_embedding_model(),
        "generation_model": resolve_generation_model(),
    }


@app.post("/save")
async def save(payload: GeneratePayload) -> dict[str, Any]:
    if not payload.content.strip():
        raise HTTPException(status_code=400, detail="缺少可保存内容")
    feishu = await save_to_feishu(payload)
    log_usage_event(
        "save_to_feishu",
        {
            "topic": payload.topic,
            "platform": payload.platform,
            "types": payload.types,
            "saved": feishu.get("saved", False),
            "recordId": feishu.get("recordId", ""),
        },
    )
    return {"success": True, "feishu": feishu}


@app.post("/agent/generate")
async def agent_generate(payload: AgentGeneratePayload) -> dict[str, Any]:
    result = await execute_agent_generate_pipeline(payload)
    generate_result = result["generate_result"]
    plan = result["agent_plan"]
    log_usage_event(
        "agent_generate",
        {
            "topic": generate_result["payload"].topic,
            "platform": generate_result["payload"].platform,
            "types": generate_result["payload"].types,
            "agentMode": plan["agentMode"],
            "retrievalMode": plan["retrievalMode"],
            "riskMode": plan["riskMode"],
        },
    )
    return {
        "success": True,
        "content": generate_result["content"],
        "promptPreview": generate_result["prompt"][:500],
        "knowledgeSource": engine.source,
        "retrievedKnowledge": {
            "documents": [item for item in generate_result["retrieved"] if item.get("sourceType") == "document"],
            "conversations": [item for item in generate_result["retrieved"] if item.get("sourceType", "").startswith("conversation_")],
        },
        "trace": GenerateExecutionTrace(
            intent=GenerateIntentSummary(**generate_result["intent_summary"]),
            queryBundle=generate_result["query_bundle"],
            retrievalTrace=RetrievalTrace(**generate_result["retrieval_trace"]),
        ),
        "agentTrace": AgentExecutionTrace(
            plan=AgentPlan(**plan),
            generateTrace=GenerateExecutionTrace(
                intent=GenerateIntentSummary(**generate_result["intent_summary"]),
                queryBundle=generate_result["query_bundle"],
                retrievalTrace=RetrievalTrace(**generate_result["retrieval_trace"]),
            ),
        ),
    }


@app.post("/retrieve")
def retrieve(payload: RetrievePayload) -> dict[str, Any]:
    results, trace = engine.retrieve_with_trace(payload)
    log_usage_event(
        "retrieve_only",
        {
            "topic": payload.topic,
            "platform": payload.platform,
            "top_k": payload.top_k,
            "returnedCount": len(results),
        },
    )
    return {
        "source": engine.source,
        "total_chunks": len(engine.chunks),
        "trace": RetrievalTrace(**trace),
        "results": [RetrievedChunk(**item) for item in results],
    }


@app.get("/debug/chunks")
def debug_chunks(
    source_type: str = Query(default=""),
    q: str = Query(default=""),
    limit: int = Query(default=50, ge=1, le=500),
) -> ChunkDebugResponse:
    if not engine.ready:
        engine.build_index()

    query = q.strip().lower()
    items: list[dict[str, Any]] = []

    for chunk in engine.chunks:
        raw = dict(chunk.raw)
        if source_type and raw.get("sourceType", "") != source_type:
            continue
        if query:
            haystack = " ".join(
                [
                    raw.get("title", ""),
                    " ".join(raw.get("headingPath", [])),
                    raw.get("content", ""),
                    raw.get("sourceFile", ""),
                    raw.get("sourceLabel", ""),
                    raw.get("statementType", ""),
                    raw.get("intentType", ""),
                    raw.get("decisionStatus", ""),
                    " ".join(raw.get("keywords", [])),
                ]
            ).lower()
            if query not in haystack:
                continue
        items.append(raw)

    return ChunkDebugResponse(
        source=engine.source,
        total_chunks=len(engine.chunks),
        filtered_chunks=len(items),
        items=items[:limit],
    )


@app.get("/debug/stats")
def debug_stats() -> dict[str, Any]:
    if not engine.ready:
        engine.build_index()

    by_source_type: dict[str, int] = {}
    by_statement_type: dict[str, int] = {}
    by_decision_status: dict[str, int] = {}

    for chunk in engine.chunks:
        raw = chunk.raw
        source_type = raw.get("sourceType", "") or "unknown"
        statement_type = raw.get("statementType", "") or "none"
        decision_status = raw.get("decisionStatus", "") or "none"
        by_source_type[source_type] = by_source_type.get(source_type, 0) + 1
        by_statement_type[statement_type] = by_statement_type.get(statement_type, 0) + 1
        by_decision_status[decision_status] = by_decision_status.get(decision_status, 0) + 1

    return {
        "source": engine.source,
        "total_chunks": len(engine.chunks),
        "bySourceType": by_source_type,
        "byStatementType": by_statement_type,
        "byDecisionStatus": by_decision_status,
    }


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


def build_generate_intent_summary(payload: GeneratePayload) -> dict[str, Any]:
    intent_text = "；".join(part for part in [payload.brief, payload.rawInput] if part).strip()
    focus = extract_primary_focus(intent_text or payload.brief or TOPICS[payload.topic]["label"])
    negative_constraints = extract_negative_constraints(intent_text)
    generation_mode = infer_generation_mode(payload)
    input_completeness = infer_input_completeness(intent_text)
    intent_type = infer_generation_intent(payload)
    should_retrieve = bool(payload.brief.strip() or payload.rawInput.strip())
    return {
        "intentType": intent_type,
        "generationMode": generation_mode,
        "inputCompleteness": input_completeness,
        "focus": focus,
        "negativeConstraints": negative_constraints,
        "shouldRetrieve": should_retrieve,
    }


def build_content_agent_plan(payload: AgentGeneratePayload) -> dict[str, Any]:
    normalized = normalize_generate_payload(payload)
    intent_summary = build_generate_intent_summary(normalized)
    agent_mode = infer_agent_mode(normalized, intent_summary)
    retrieval_mode = infer_agent_retrieval_mode(normalized, intent_summary, agent_mode)
    risk_mode = infer_agent_risk_mode(normalized)
    output_goal = infer_agent_output_goal(normalized, payload.agentGoal)
    planning_notes = build_agent_planning_notes(normalized, intent_summary, agent_mode, retrieval_mode, risk_mode)
    return {
        "agentMode": agent_mode,
        "retrievalMode": retrieval_mode,
        "riskMode": risk_mode,
        "outputGoal": output_goal,
        "planningNotes": planning_notes,
    }


async def execute_generate_pipeline(payload: GeneratePayload) -> dict[str, Any]:
    normalized = normalize_generate_payload(payload)
    validate_generate_payload(normalized)
    intent_summary = build_generate_intent_summary(normalized)
    retrieve_payload = RetrievePayload(
        topic=normalized.topic,
        platform=normalized.platform,
        brief=normalized.brief,
        rawInput=normalized.rawInput,
        top_k=max(DEFAULT_GENERATION_TOP_K, 6),
    )
    retrieved, retrieval_trace = engine.retrieve_with_trace(retrieve_payload) if intent_summary["shouldRetrieve"] else ([], {
        "queryBundle": build_empty_query_bundle(normalized.platform),
        "shortlistSize": 0,
        "returnedCount": 0,
    })
    query_bundle = retrieval_trace["queryBundle"]
    prompt_knowledge = select_generation_knowledge_items(normalized, retrieved)
    prompt = build_prompt(normalized, prompt_knowledge, engine.source)
    content = await generate_with_dashscope(prompt)
    content = normalize_generated_content(content)
    return {
        "payload": normalized,
        "intent_summary": intent_summary,
        "retrieved": retrieved,
        "retrieval_trace": retrieval_trace,
        "query_bundle": query_bundle,
        "prompt": prompt,
        "content": content,
    }


async def execute_agent_generate_pipeline(payload: AgentGeneratePayload) -> dict[str, Any]:
    normalized = AgentGeneratePayload(**payload.model_dump())
    validate_generate_payload(normalize_generate_payload(normalized))
    agent_plan = build_content_agent_plan(normalized)
    routed_payload = apply_agent_plan_to_payload(normalized, agent_plan)
    generate_result = await execute_generate_pipeline(routed_payload)
    return {
        "agent_plan": agent_plan,
        "generate_result": generate_result,
    }


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


def infer_agent_mode(payload: GeneratePayload, intent_summary: dict[str, Any]) -> str:
    if intent_summary["generationMode"] == "忠实改写":
        return "rewrite"
    if "教育" in intent_summary["intentType"] or "认知" in intent_summary["intentType"]:
        return "educate"
    return "sell"


def infer_agent_retrieval_mode(payload: GeneratePayload, intent_summary: dict[str, Any], agent_mode: str) -> str:
    if payload.topic == "hospital" or payload.platform == "zhihu" or "article" in payload.types:
        return "strict"
    if agent_mode == "rewrite" and intent_summary["inputCompleteness"] == "complete":
        return "light"
    return "normal"


def infer_agent_risk_mode(payload: GeneratePayload) -> str:
    intent_text = " ".join(part for part in [payload.brief, payload.rawInput] if part).strip()
    if payload.topic == "hospital" or payload.platform == "zhihu" or any(token in intent_text for token in ["备案", "审评", "申报"]):
        return "compliance_sensitive"
    return "normal"


def infer_agent_output_goal(payload: GeneratePayload, agent_goal: str) -> str:
    if agent_goal.strip():
        return agent_goal.strip()
    if payload.types:
        return " / ".join(CONTENT_TYPE_NAMES.get(item, item) for item in payload.types)
    return "内容生成"


def build_agent_planning_notes(
    payload: GeneratePayload,
    intent_summary: dict[str, Any],
    agent_mode: str,
    retrieval_mode: str,
    risk_mode: str,
) -> list[str]:
    notes = [
        f"优先模式：{agent_mode}",
        f"检索策略：{retrieval_mode}",
        f"风险等级：{risk_mode}",
        f"主轴：{intent_summary['focus']}",
    ]
    if intent_summary["negativeConstraints"]:
        notes.append(f"负向约束：{intent_summary['negativeConstraints']}")
    if payload.platform:
        notes.append(f"平台约束：{PLATFORMS[payload.platform]['label']}")
    return notes


def apply_agent_plan_to_payload(payload: AgentGeneratePayload, agent_plan: dict[str, Any]) -> GeneratePayload:
    strategy_lines = [
        f"Agent模式：{agent_plan['agentMode']}",
        f"检索模式：{agent_plan['retrievalMode']}",
        f"风险模式：{agent_plan['riskMode']}",
        f"输出目标：{agent_plan['outputGoal']}",
    ]
    if agent_plan["agentMode"] == "rewrite":
        strategy_lines.append("优先忠实改写当前输入，不主动新增第二主题。")
    elif agent_plan["agentMode"] == "educate":
        strategy_lines.append("优先讲清原因、判断逻辑、适用场景，不要写成硬广。")
    else:
        strategy_lines.append("优先讲清能力价值、适用场景和解决的问题，不要空泛罗列。")

    if agent_plan["retrievalMode"] == "strict":
        strategy_lines.append("严格使用有明确依据的知识，不补具体案例、数字和结果。")
    elif agent_plan["retrievalMode"] == "light":
        strategy_lines.append("检索只做轻度补位，避免知识压过用户原始表达。")
    else:
        strategy_lines.append("检索用于补充事实和业务理解，但不要抢主轴。")

    if agent_plan["riskMode"] == "compliance_sensitive":
        strategy_lines.append("默认保守表达，避免确定性承诺和未证实结论。")

    agent_context = "【Agent策略】\n" + "\n".join(f"- {line}" for line in strategy_lines)
    merged_raw_input = "\n".join(part for part in [payload.rawInput.strip(), agent_context] if part).strip()
    return GeneratePayload(
        topic=payload.topic,
        brief=payload.brief,
        platform=payload.platform,
        types=payload.types,
        content=payload.content,
        rawInput=merged_raw_input,
        prompt=payload.prompt,
    )


def build_prompt(payload: GeneratePayload, knowledge_items: list[dict[str, Any]], knowledge_source: str) -> str:
    topic = TOPICS[payload.topic]
    platform = PLATFORMS[payload.platform]
    selected_types = "、".join(CONTENT_TYPE_NAMES.get(item, item) for item in payload.types)
    supplemental = f"【补充补充说明】\n{payload.rawInput}\n" if payload.rawInput else ""
    scene_prompt = build_scene_prompt(payload.topic)
    knowledge_section = build_generation_knowledge_section(knowledge_items)
    task_section = build_generation_task_section(payload)
    process_guidance = build_process_guidance(payload, knowledge_items)
    trial_pilot_guidance = build_trial_pilot_guidance(payload, knowledge_items)
    compliance_guidance = build_compliance_guidance(payload)

    return f"""【角色】
你是华胜品牌内容团队一员。你要做的不是汇报材料，也不是品牌口号堆砌，而是把当前输入改写或生成成可直接发布的内容。

【参数】
主题：{topic['label']}
视频素材/表达重点：{payload.brief}
平台：{platform['label']}
内容类型：{selected_types}

{task_section}

【核心规则】
- 页面输入框里的“视频素材/表达重点”优先级最高，高于主题按钮、平台按钮和历史卖点
- 如果用户输入已经是一段完整观点、类比、故事或论证，默认做忠实改写，不要自由扩写
- 只允许一个主轴，最多两个支撑点；不服务主轴的知识、数字、设备名、背书全部删掉
- 相关不等于适用。知识可以帮助你理解，但不一定可以写进正文

【知识用法】
- 结构化知识负责事实兜底；没有明确证据的数字、能力、资质、案例，不要补
- 对话知识负责业务理解和表达角度；用来帮助你讲得更懂业务，不要原句照抄
- 如果处于忠实改写模式，知识库只能帮助你换成华胜语境、补足必要边界和润色措辞，不能新增第二主题、具体案例、具体百分比或夸张结论

【表达】
- 优先自然、可信、像人说话，不要写成 PPT、汇报稿、企业简介摘要
- 不要为了显得专业而主动堆术语；如果一个专业词不能明显增强理解，就换成更自然的说法
- 开场必须从当前输入里提炼，不能套固定钩子
- 结论要落在专业判断、业务价值或解决方案上，但不要硬拔高
{process_guidance}
{trial_pilot_guidance}

【边界】
- 符合广告法，不写绝对化、最高级、保底承诺、疗效承诺、无法证实的夸张结论
- 客户名称默认视为隐私，除非资料明确可公开，否则写成“某药企”“某客户”“某合作方”
- 禁用术语“CIP”，统一写成“自动清洗站”
{compliance_guidance}

【参考知识】
以下资料来自华胜知识库（来源：{knowledge_source}）：
{knowledge_section}

{supplemental}【补充提醒】
- 主题按钮只用于限定业务范围，不代表固定卖点
- 平台按钮只用于约束表达形式，不要反过来决定主轴
- 当前主题补充参考：{scene_prompt}

【字数要求】
{build_selected_word_counts(payload.types, platform['wordCount'])}

【内容类型具体要求】
{build_content_requirements(platform, payload.types)}

【输出格式】
严格按以下内容类型输出，每个模块用【】标注：
    {chr(10).join(f"【{CONTENT_TYPE_NAMES.get(item, item)}】" for item in payload.types)}"""


def select_generation_knowledge_items(payload: GeneratePayload, knowledge_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    intent_text = " ".join(part for part in [payload.brief, payload.rawInput] if part).strip()
    is_capacity_context = payload.topic == "hospital" and any(token in intent_text for token in ["扩产", "扩大产能", "产能", "放量", "放大生产"])

    selected = list(knowledge_items)
    if is_capacity_context:
        filtered: list[dict[str, Any]] = []
        for item in selected:
            text = " ".join(
                [
                    item.get("title", ""),
                    item.get("content", ""),
                    " ".join(item.get("keywords", [])),
                ]
            )
            if re.search(r"\b\d+\s*万贴\b", text):
                continue
            filtered.append(item)
        if filtered:
            selected = filtered
    return selected[:3]


def build_generation_task_section(payload: GeneratePayload) -> str:
    intent_text = "；".join(part for part in [payload.brief, payload.rawInput] if part).strip()
    intent_type = infer_generation_intent(payload)
    focus = extract_primary_focus(intent_text or payload.brief or TOPICS[payload.topic]["label"])
    avoid = extract_negative_constraints(intent_text)
    generation_mode = infer_generation_mode(payload)
    input_completeness = infer_input_completeness(intent_text)
    value_floor_line = build_value_floor_guidance(payload, intent_text)
    avoid_line = f"- 不要展开：{avoid}" if avoid else "- 不要偏离用户当前输入去平均铺开历史卖点"
    mode_line = (
        "- 写法：忠实改写，保留原文逻辑，只换成更适合华胜对外表达的说法"
        if generation_mode == "忠实改写"
        else "- 写法：可适度组织，但仍然只能围绕一个主轴展开"
    )
    button_line = (
        "- 按钮作用：当前输入已足够完整，主题和平台只收边界"
        if input_completeness == "complete"
        else "- 按钮作用：当前输入较短，主题和平台只做少量补位"
        if input_completeness == "partial"
        else "- 按钮作用：当前输入给主方向，主题和平台只做轻度校正"
    )
    return f"""【本次生成任务定义】
- 生成模式：{generation_mode}
- 输入完整度：{input_completeness}
- 任务类型：{intent_type}
- 本次主轴：{focus}
{avoid_line}
- 页面输入优先级：最高
- {button_line}
- {mode_line}
- {value_floor_line}
- {build_scope_guard(payload, intent_text)}
- 如果用户当前输入与知识库历史重点不一致，以用户当前输入为准"""


def build_process_guidance(payload: GeneratePayload, knowledge_items: list[dict[str, Any]]) -> str:
    process_terms = extract_process_terms(knowledge_items)
    brief_text = " ".join(part for part in [payload.brief, payload.rawInput] if part).strip()
    is_equipment_context = payload.topic == "equipment" or any(
        token in brief_text for token in ["设备", "产线", "生产线", "涂布机", "涂布", "凝胶贴膏"]
    )
    if not is_equipment_context or len(process_terms) < 3:
        return "- 如果知识里出现设备流程细节，优先保留与主轴直接相关的关键环节，不要写成参数清单"

    process_chain = " → ".join(process_terms[:6])
    return (
        f"- 如果参考资料出现同一设备的明确工序链（如：{process_chain}），这属于流程骨架，不算平行卖点；"
        "在不偏离主轴的前提下，优先保留完整链路里的关键环节，避免只写其中一段导致设备理解变窄"
    )


def build_trial_pilot_guidance(payload: GeneratePayload, knowledge_items: list[dict[str, Any]]) -> str:
    intent_text = " ".join(part for part in [payload.brief, payload.rawInput] if part).strip()
    mentions_trial = "试机" in intent_text
    mentions_pilot = any(token in intent_text for token in ["中试", "放大", "中试基地", "中试车间"])
    if not mentions_trial and not mentions_pilot:
        return ""

    sites = {item.get("executionSite", "") for item in knowledge_items if item.get("executionSite")}
    pilot_contexts = {item.get("pilotContext", "") for item in knowledge_items if item.get("pilotContext")}
    if mentions_trial and not mentions_pilot:
        return (
            "- 试机是设备验证行为，不默认等于中试；只有资料明确说明发生在中试基地或中试车间时，才能写成中试环境里的试机。"
            "如果资料只说明现场试机、车间试机或设备试运行，就按对应场景表达，不要自动扩写成中试。"
        )
    if mentions_pilot:
        if "explicit_pilot" in pilot_contexts or "pilot_base" in sites:
            return "- 本次可以按中试场景表达，但仍要区分“中试阶段验证”和“设备试机”不是同一个概念，不要混写。"
        return "- 如果资料没有明确中试基地或中试车间信息，不要把试机默认写成在中试环境完成。"
    return ""


def build_value_floor_guidance(payload: GeneratePayload, intent_text: str) -> str:
    needs_value_clarity = payload.topic in {"pilot", "equipment", "hospital", "onestop"} or any(
        token in intent_text for token in ["中试", "平台", "设备", "备案", "工艺", "协同", "放大"]
    )
    if not needs_value_clarity:
        return "保持克制表达，但不要把价值讲虚、讲散或讲没"
    return (
        "即使需要克制表达，也必须讲清这项能力到底解决什么问题、适合什么场景、为什么值得使用；"
        "不要因为避免企业宣传或避免拔高，就把自身能力弱化成“只是提供一下”这类无效表述；"
        "禁止出现“不是卖设备”“不卖设备”“不是代工”“不包结果”“不接代工”“不是简单租个车间”“它不替代你的研发判断”这类自我降调句式"
    )


def build_compliance_guidance(payload: GeneratePayload) -> str:
    brief_text = " ".join(part for part in [payload.brief, payload.rawInput] if part).strip()
    is_filing_context = payload.topic == "hospital" or any(token in brief_text for token in ["备案", "审评", "申报", "院内制剂"])
    is_capacity_context = any(token in brief_text for token in ["扩产", "扩大产能", "产能", "放量", "放大生产"])
    is_long_form = "article" in payload.types or payload.platform == "zhihu"
    input_completeness = infer_input_completeness(brief_text)
    if not is_filing_context:
        guidance = ["- 没有明确资料支持时，不要补具体客户案例、具体数据、具体百分比、具体时长收益或具体审评结果"]
        if is_capacity_context:
            guidance.append("- 扩产或产能场景下，不要自动补单日产量、原料重量、收率比例、节拍提升这类具体数字")
        if input_completeness == "partial":
            guidance.append("- 当前输入较短时，不要主动补“合规、注册、审评、备案、GMP”这类背书型表达，除非用户明确提到")
        return "\n".join(guidance)
    if is_long_form:
        return """- 备案支持场景下，禁止虚构或脑补具体医院等级、具体项目名、具体客户身份、具体百分比、具体批次数、具体时长收益、具体审评问询结果
- 如果资料没有明确给出，不要写“已获备案号”“顺利过审”“提升到XX%”“缩短X个月”这类结论
- 如果资料没有明确可公开证据，不要写“多个客户经验”“某客户案例”“我们做过很多这类项目”这类经验背书或半案例化表达
- 没有明确资料支持时，不要主动补CPP、CQA、GPP/GMP逻辑、温度±1℃、转速±2%这类审评或参数细节
- 没有明确资料支持时，不要用“比如/例如”去举具体设备接口、具体检测窗口、具体残留验证方式这类细节例子
- 不要写“某客户在……时发现……”这类案例式段落，也不要补RSD、采样频率、记录颗粒度、F0值、电导率、采样窗口、时间点、温度时长组合等审评细节
- 优先写成通用场景判断、常见问题归纳、设备侧支持边界，不要把长文写成案例复盘
- 图文长文结尾不要写强销售式行动号召，优先用边界说明、专业判断或适用条件收束"""
    guidance = [
        "- 备案支持场景下，没有明确资料支持时，不要补具体医院、具体案例、具体百分比、具体审评结果",
        "- 可以讲“支持备案协同”，但不要写成“代办备案”“包过审”或暗示确定性通过",
        "- 没有明确资料支持时，不要主动补CPP、CQA、GPP/GMP逻辑、具体参数精度或自动生成日志这类硬技术背书",
    ]
    if is_capacity_context:
        guidance.append("- 扩产或产能场景下，不要自动补单日产量、原料重量、收率比例、节拍提升这类具体数字")
    if input_completeness == "partial":
        guidance.append("- 当前输入较短时，优先讲协同边界和实际作用，不要自动补成完整备案解决方案")
    return "\n".join(guidance)


def build_empty_query_bundle(platform: str = "") -> dict[str, Any]:
    return {
        "query_text": "",
        "intent_text": "",
        "intent_tokens": [],
        "negative_tokens": [],
        "primary_focus": "",
        "highlighted_phrases": [],
        "input_completeness": "none",
        "topic_label": "",
        "platform": platform,
        "retrieval_intent": "none",
    }


def extract_process_terms(knowledge_items: list[dict[str, Any]]) -> list[str]:
    ordered_terms = ["放卷", "涂布", "复合", "测厚", "裁切", "分切", "切片", "收集", "收卷", "装袋", "包装"]
    text = " ".join(
        " ".join(
            [
                item.get("title", ""),
                item.get("content", ""),
                " ".join(item.get("keywords", [])),
            ]
        )
        for item in knowledge_items
    )
    found: list[str] = []
    for term in ordered_terms:
        if term in text and term not in found:
            found.append(term)
    return found


def normalize_generated_content(text: str) -> str:
    normalized = text.replace("CIP", "自动清洗站").replace("cip", "自动清洗站")
    normalized = re.sub(r"自动清洗站（自动清洗站）", "自动清洗站", normalized)
    normalized = re.sub(r"自动清洗站\s*\(\s*自动清洗站\s*\)", "自动清洗站", normalized)
    normalized = re.sub(r"自动清洗站\s*[（(]\s*非自动清洗站\s*[）)]", "自动清洗站", normalized)
    normalized = normalized.replace("不接代工，不包结果", "重点是把工艺在量产前跑通")
    normalized = normalized.replace("不接代工", "重点是提前把工艺跑通")
    normalized = normalized.replace("不包结果", "重点是把关键问题提前暴露出来")
    normalized = normalized.replace("不是简单租个车间", "不是只给一个场地")
    normalized = normalized.replace("它不是租个车间那么简单", "重点不是给一个场地，而是把量产前的问题提前跑出来")
    normalized = normalized.replace("不是租个车间那么简单", "不是只给一个场地")
    normalized = normalized.replace("它不替代你的研发判断", "它让你的研发判断更有依据")
    normalized = normalized.replace("它让你的研发判断更有依据，但给你一套", "它给你一套")
    normalized = normalized.replace("GMP产线节奏", "药企现场试机节奏")
    normalized = normalized.replace("按GMP产线节奏", "按药企现场试机节奏")
    normalized = normalized.replace("符合GMP要求", "符合药企生产要求")
    normalized = normalized.replace("符合GMP规范", "符合药企规范要求")
    normalized = normalized.replace("GMP要求", "药企生产要求")
    normalized = normalized.replace("GMP规范", "药企规范要求")
    normalized = normalized.replace("GMP合规性出厂就带", "生产要求相关基础配置出厂就带")
    normalized = normalized.replace("GPP/GMP逻辑对齐", "生产与验证逻辑对齐")
    normalized = normalized.replace("CPP", "关键工艺参数")
    normalized = normalized.replace("CQA", "关键质量属性")
    normalized = normalized.replace("小试能用，中试不断，量产稳得住", "从研发到放大，设备链路更顺")
    normalized = normalized.replace("关键质量属性（关键质量属性）", "关键质量属性")
    normalized = normalized.replace("GMP材质全到位", "药用接触材质到位")
    normalized = normalized.replace("重点不是给一个场地，而是把量产前的问题提前跑出来，而是", "重点不是给一个场地，而是把量产前的问题提前跑出来：")
    normalized = normalized.replace("小试、中试、量产，一套逻辑跑到底", "从研发到放大，一套设备逻辑跑顺")
    normalized = normalized.replace("F0值", "灭菌相关连续数据")
    normalized = normalized.replace("电导率", "清洗相关过程数据")
    normalized = normalized.replace("采样窗口", "采样条件")
    normalized = re.sub(r"\b\d+\s*万贴\b", "稳定放量", normalized)
    return normalized


def build_scope_guard(payload: GeneratePayload, intent_text: str) -> str:
    compact_text = re.sub(r"\s+", "", intent_text)
    if payload.topic == "equipment" and len(compact_text) <= 8:
        return "如果输入只是泛泛提到设备，不要自动扩写到中试、备案、注册或完整产线方案；只选一个最核心的设备侧痛点来讲"
    if payload.topic == "hospital":
        return "院内制剂主题下，优先讲设备侧支撑边界和协同作用，不要自动补成完整备案解决方案"
    if payload.topic == "pilot":
        return "中试主题下，重点讲放大验证价值和现场能力，不要反复用反衬句式降低自身价值"
    return "如果输入很短，就收窄范围，只讲一个最核心的问题和一个最关键的支撑点"


def build_scene_prompt(topic_id: str) -> str:
    prompts = {
        "pilot": """- 主题边界：可参考中试转化、中试平台、研发到量产衔接
- 不要因为主题是中试，就自动展开基地规模、验证流程或案例""",
        "equipment": """- 主题边界：可参考设备能力、自动化、稳定性、放大适配
- 不要因为主题是设备，就自动罗列参数、验证、清洗或全部卖点""",
        "material": """- 主题边界：可参考包材、适配性、稳定性、协同关系
- 不要把包材主题硬写成设备或基地宣传""",
        "hospital": """- 主题边界：可参考院内制剂、备案支持、工艺协同
- 不要自动扩写成完整解决方案总述""",
        "strength": """- 主题边界：可参考长期积累、整体能力、品牌实力
- 不要默认堆客户数、专利数、出口数等高波动数字""",
        "onestop": """- 主题边界：可参考协同效率、整体配合、系统性风险
- 不要把一站式写成简单产品打包或卖点合集""",
        "training": """- 主题边界：可参考培训、上手效率、能力转移
- 不要自动写成企业宣传或团队介绍""",
        "tdts": """- 主题边界：可参考行业认知、产业落地判断、经皮知识
- 不要只谈趋势概念，也不要强行落到所有业务模块""",
    }
    return prompts.get(
        topic_id,
        """- 主题只限定业务范围，不直接决定主轴
- 不要把单一能力模块误写成华胜全部业务""",
    )


def build_selected_word_counts(types: list[str], word_count_map: dict[str, str]) -> str:
    lines = [
        f"- {CONTENT_TYPE_NAMES.get(item, item)}：{word_count_map[item]}"
        for item in types
        if item in word_count_map
    ]
    return "\n".join(lines)


def build_generation_knowledge_section(knowledge_items: list[dict[str, Any]]) -> str:
    document_items = [item for item in knowledge_items if item.get("sourceType") == "document"]
    conversation_items = [item for item in knowledge_items if item.get("sourceType", "").startswith("conversation_")]

    document_section = build_prompt_knowledge_group(
        "文档事实依据",
        document_items,
        include_conversation_meta=False,
    )
    conversation_section = build_prompt_knowledge_group(
        "对话业务理解",
        conversation_items,
        include_conversation_meta=True,
    )

    parts = [part for part in [document_section, conversation_section] if part]
    return "\n\n".join(parts) if parts else "暂无召回资料。"


def build_prompt_knowledge_group(title: str, items: list[dict[str, Any]], include_conversation_meta: bool) -> str:
    if not items:
        return ""

    lines = [f"【{title}】"]
    for idx, item in enumerate(items, start=1):
        item_lines = [
            f"资料{idx}｜{item['title']}",
            f"路径：{' > '.join(item.get('headingPath', []))}",
            f"相关度：{item.get('finalScore', 0)}",
        ]
        if item.get("triggerCondition"):
            item_lines.append(f"触发条件：{item.get('triggerCondition', '')}")
        if item.get("usageRule"):
            item_lines.append(f"使用规则：{item.get('usageRule', '')}")
        if item.get("sourceModule"):
            item_lines.append(f"来源模块：{item.get('sourceModule', '')}")
        if item.get("confidence"):
            item_lines.append(f"引用置信度：{item.get('confidence', '')}")
        if include_conversation_meta:
            if item.get("statementType"):
                item_lines.append(f"陈述类型：{item.get('statementType', '')}")
            if item.get("intentType"):
                item_lines.append(f"意图类型：{item.get('intentType', '')}")
            if item.get("decisionStatus"):
                item_lines.append(f"讨论状态：{item.get('decisionStatus', '')}")
            if item.get("certainty"):
                item_lines.append(f"把握度：{item.get('certainty', '')}")
            if item.get("sourceTimeRange"):
                item_lines.append(f"时间片段：{item.get('sourceTimeRange', '')}")
        item_lines.append(item["content"])
        lines.append("\n".join(part for part in item_lines if part))

    return "\n\n".join(lines)


def build_content_requirements(platform: dict[str, Any], types: list[str]) -> str:
    requirements: list[str] = []
    for item in types:
        if item == "oral":
            requirements.append(
                f"""【口播脚本 - {platform['label']}版】
- 字数：{platform['wordCount']['oral']}
- 时长感：按30秒到1分钟口播来写，宁可短而有力，也不要写成图文正文
- 形式：{platform['rhythm']}，{platform['sentenceLength']}
- 开头：从当前输入里提炼切口，不要套模板
- 中间：只围绕一个主轴展开
- 结尾：自然收住，不要硬引导
- 节奏：要有问题感、反差感或现场感，避免平铺直叙写成行业标准答案
- 用词：优先说人话，少用缩写和硬专业词；像GMP这类词，除非用户明确要求专业版，否则改写成“药企生产要求”“现场试机要求”这类自然表达
- 约束：最多两个支撑点，禁止罗列多个设备功能或多组平行卖点"""
            )
        elif item == "title":
            requirements.append(
                f"""【标题+封面文案 - {platform['label']}版】
- 数量：3个备选标题 + 1条封面文案
- 标题字数：{platform['wordCount']['title']}
- 标题要求：{get_title_requirement(payload_platform_id(platform))}
- 封面文案：10-15字，简洁直接
- 不要把多个卖点都塞进同一个标题"""
            )
        elif item == "article":
            requirements.append(
                f"""【图文正文 - {platform['label']}版】
- 字数：{platform['wordCount']['article']}
- 结构：{platform['structure']}
- 格式：{get_article_format(payload_platform_id(platform))}
- 写法：不要写成资料拼盘，要围绕一个清晰主轴递进展开
- 约束：正文最多展开两个支撑点，每个支撑点都必须服务主轴"""
            )
        elif item == "comment":
            requirements.append(
                f"""【评论区置顶话术 - {platform['label']}版】
- 字数：{platform['wordCount']['comment']}
- 风格：{get_comment_style(payload_platform_id(platform))}
- 类型：补充说明型/互动回应型/预约承接型
- 要求：延续正文主轴，不要新开第二主题"""
            )
        elif item == "moments":
            requirements.append(
                f"""【朋友圈文案 - {platform['label']}版】
- 字数：{platform['wordCount']['moments']}
- 结构：场景引入 → 反常识观点 → 华胜植入 → 分层引导
- 人设：华胜团队一员，专业但不高冷
- 要求：像业务一线的真实判断，不要像企业简介摘要"""
            )
    return "\n\n".join(requirements)


def infer_generation_intent(payload: GeneratePayload) -> str:
    text = " ".join(part for part in [payload.brief, payload.rawInput] if part).strip()
    if any(token in text for token in ["为什么", "怎么理解", "趋势", "行业", "认知"]):
        return "客户教育 / 行业认知"
    if any(token in text for token in ["痛点", "卡在", "问题", "失败", "浪费", "不放行"]):
        return "痛点切入 / 价值表达"
    if any(token in text for token in ["参数", "检测", "设备", "工艺", "能力", "自动化", "全检"]):
        return "能力说明 / 卖点表达"
    if payload.platform in {"douyin", "shipinhao", "xiaohongshu"}:
        return "内容表达 / 卖点包装"
    return "业务说明"


def infer_generation_mode(payload: GeneratePayload) -> str:
    text = " ".join(part for part in [payload.brief, payload.rawInput] if part).strip()
    if not text:
        return "自由生成"
    sentence_count = len(re.findall(r"[。！？?!.；;]", text))
    has_story_markers = any(token in text for token in ["比如", "例如", "看到", "发现", "于是", "结果", "为什么", "很多人", "有个", "一滴油"])
    has_long_structure = len(text) >= 70 and sentence_count >= 2
    if has_story_markers or has_long_structure:
        return "忠实改写"
    return "自由生成"


def infer_input_completeness(text: str) -> str:
    normalized = text.strip()
    if not normalized:
        return "partial"
    sentence_count = len(re.findall(r"[。！？?!.；;\n]", normalized))
    has_explicit_focus = bool(extract_highlight_phrases(normalized)) or bool(extract_negative_constraints(normalized))
    if len(normalized) >= 70 or sentence_count >= 2 or has_explicit_focus:
        return "complete"
    if len(normalized) >= 28:
        return "medium"
    return "partial"


def extract_primary_focus(text: str) -> str:
    lowered = text.strip()
    highlight_phrases = extract_highlight_phrases(lowered)
    if highlight_phrases:
        return highlight_phrases[0]
    negative_removed = re.sub(r"(不要|别|不想|不强调|不要总讲|不要总是强调)[^。；，,\n]{1,18}", "", lowered).strip()
    if not negative_removed:
        return "围绕当前素材的核心价值点"
    parts = re.split(r"[。；，,\n]|但是|而是|主要是|重点是", negative_removed)
    for part in parts:
        cleaned = part.strip(" ：:;；，,")
        if len(cleaned) >= 4:
            return cleaned[:24]
    return negative_removed[:24] if negative_removed else "围绕当前素材的核心价值点"


def payload_platform_id(platform: dict[str, Any]) -> str:
    for key, value in PLATFORMS.items():
        if value is platform:
            return key
    return ""


def get_title_requirement(platform_id: str) -> str:
    if platform_id == "douyin":
        return "短、准、直接，有切口但不要夸张过头"
    if platform_id == "xiaohongshu":
        return "清楚好懂，别堆词"
    if platform_id == "zhihu":
        return "问题明确、表达完整，不要写太硬"
    return "主题明确、价值清楚，不要许诺过多"


def get_article_format(platform_id: str) -> str:
    if platform_id == "xiaohongshu":
        return "多用 emoji 分段，清单式排版"
    if platform_id == "zhihu":
        return "标题层级清楚，必要时用数据和来源增强可信度"
    return "段落清晰，每段不超过3-5行"


def get_comment_style(platform_id: str) -> str:
    if platform_id == "douyin":
        return "口语化、自然延续正文，不要套路互动"
    if platform_id == "xiaohongshu":
        return "亲和互动，像补一句有用的话"
    if platform_id == "zhihu":
        return "专业、克制、顺着正文补充"
    return "有温度、建立信任"


async def generate_with_dashscope(prompt: str) -> str:
    api_key = resolve_dashscope_api_key()
    if not api_key:
        raise HTTPException(status_code=500, detail="DASHSCOPE_API_KEY 未配置")

    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            f"{resolve_dashscope_base_url()}/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            json={
                "model": resolve_generation_model(),
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.85,
            },
        )
        if response.status_code >= 400:
            try:
                detail = response.json()
            except Exception:
                detail = response.text
            raise HTTPException(status_code=500, detail=f"DashScope 生成请求失败: {detail}")
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


def log_usage_event(event_type: str, payload: dict[str, Any]) -> None:
    EVENT_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "event": event_type,
        "timestamp": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
        "payload": payload,
    }
    with EVENT_LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


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


def rewrite_query(payload: RetrievePayload) -> dict[str, Any]:
    return build_rule_based_query_bundle(payload)


def rerank_candidates(query_bundle: dict[str, Any], chunk: dict[str, Any]) -> float:
    return rule_based_rerank_score(chunk, query_bundle)


def tokenize(text: str) -> list[str]:
    return list({m.group(0).lower() for m in re.finditer(r"[\u4e00-\u9fa5]{2,12}|[a-z0-9-]{2,}", text)})


def infer_retrieval_intent(payload: RetrievePayload, text: str) -> str:
    if any(token in text for token in ["参数", "检测", "设备", "工艺", "资质", "认证", "标准", "功能", "全检"]):
        return "fact"
    if any(token in text for token in ["痛点", "卖点", "表达", "怎么讲", "内容", "文案", "包装", "切口", "客户"]):
        return "business"
    if payload.platform in {"douyin", "shipinhao", "xiaohongshu"}:
        return "business"
    return "balanced"


def infer_query_scenario_target(text: str) -> str:
    has_trial = "试机" in text or "试运行" in text
    has_pilot = any(token in text for token in ["中试", "放大", "中试基地", "中试车间"])
    if has_trial and not has_pilot:
        return "trial_run"
    if has_pilot and not has_trial:
        return "pilot"
    if has_trial and has_pilot:
        return "mixed"
    return "general"


def infer_topic_boundary_mode(topic: str, text: str) -> str:
    if topic == "equipment":
        if any(token in text for token in ["中试", "放大", "中试基地", "中试车间", "中试平台"]):
            return "cross_to_pilot_allowed"
        return "equipment_only"
    if topic == "pilot":
        if any(token in text for token in ["设备", "涂布机", "切片机", "清洗站", "生产线", "包装机"]):
            return "cross_to_equipment_allowed"
        return "pilot_only"
    if topic == "hospital":
        return "hospital_only"
    return "open"


def infer_concept_type(text: str) -> str:
    if "试机" in text or "试运行" in text:
        return "action"
    if any(token in text for token in ["中试", "放大", "中试基地", "中试车间"]):
        return "stage"
    return ""


def infer_execution_site(text: str) -> str:
    if any(token in text for token in ["中试基地", "中试车间", "中试平台"]):
        return "pilot_base"
    if any(token in text for token in ["制造车间", "生产车间", "设备车间", "车间试机", "现场试机"]):
        return "production_workshop"
    if any(token in text for token in ["客户现场", "客户来厂", "来厂试机", "参观展示区"]):
        return "client_site"
    if "实验室" in text:
        return "lab"
    return ""


def infer_pilot_context(text: str) -> str:
    if any(token in text for token in ["中试基地", "中试车间", "中试平台"]):
        return "explicit_pilot"
    if "试机" in text:
        return "non_pilot_default"
    return ""


def infer_business_scenario(text: str) -> str:
    if "试机" in text or "试运行" in text:
        return "trial_run_validation"
    if any(token in text for token in ["中试", "放大"]):
        return "pilot_scaleup"
    if any(token in text for token in ["备案", "申报", "审评"]):
        return "filing_support"
    return ""


def scenario_alignment_score(chunk: dict[str, Any], query_bundle: dict[str, Any]) -> float:
    scenario_target = query_bundle.get("scenario_target", "general")
    if scenario_target == "general":
        return 0.0

    execution_site = chunk.get("executionSite", "")
    pilot_context = chunk.get("pilotContext", "")
    business_scenario = chunk.get("businessScenario", "")
    score = 0.0

    if scenario_target == "trial_run":
        if business_scenario == "trial_run_validation":
            score += 0.35
        if execution_site == "pilot_base":
            score -= 0.5
        if pilot_context == "non_pilot_default":
            score += 0.15
    elif scenario_target == "pilot":
        if business_scenario == "pilot_scaleup":
            score += 0.35
        if execution_site == "pilot_base":
            score += 0.25
        if pilot_context == "non_pilot_default":
            score -= 0.2
    elif scenario_target == "mixed":
        if business_scenario in {"trial_run_validation", "pilot_scaleup"}:
            score += 0.15

    return score


def topic_boundary_score(chunk_topics: set[str], query_bundle: dict[str, Any]) -> float:
    mode = query_bundle.get("topic_boundary_mode", "open")
    score = 0.0
    if mode == "equipment_only":
        if "equipment" in chunk_topics:
            score += 0.5
        if "pilot" in chunk_topics:
            score -= 1.2
    elif mode == "pilot_only":
        if "pilot" in chunk_topics:
            score += 0.5
        if "equipment" in chunk_topics:
            score -= 0.8
        if "hospital" in chunk_topics:
            score -= 0.2
    elif mode == "hospital_only":
        if "hospital" in chunk_topics:
            score += 0.65
        if "pilot" in chunk_topics:
            score -= 0.45
        if "equipment" in chunk_topics:
            score -= 0.35
    elif mode == "cross_to_pilot_allowed":
        if "equipment" in chunk_topics:
            score += 0.35
        if "pilot" in chunk_topics:
            score += 0.15
    elif mode == "cross_to_equipment_allowed":
        if "pilot" in chunk_topics:
            score += 0.35
        if "equipment" in chunk_topics:
            score += 0.15
    return score


def build_rule_based_query_bundle(payload: RetrievePayload) -> dict[str, Any]:
    topic_label = TOPIC_LABELS.get(payload.topic, payload.topic or "")
    intent_text = " ".join(part for part in [payload.brief, payload.rawInput] if part).strip()
    primary_focus = extract_primary_focus(intent_text or topic_label)
    highlighted_phrases = extract_highlight_phrases(intent_text)
    emphasis_phrases = [primary_focus, *highlighted_phrases]
    emphasis_text = " ".join(phrase for phrase in emphasis_phrases if phrase).strip()
    input_completeness = infer_input_completeness(intent_text)
    query_parts = [
        primary_focus,
        emphasis_text,
        intent_text,
        emphasis_text,
        intent_text,
        payload.brief,
    ]
    if input_completeness == "partial":
        query_parts = [topic_label, payload.platform, *query_parts]
    elif input_completeness == "medium":
        query_parts = [topic_label, *query_parts, payload.platform]
    weighted_query = " ".join(part for part in query_parts if part).strip()
    negative_text = extract_negative_constraints(intent_text)
    retrieval_intent = infer_retrieval_intent(payload, intent_text)
    scenario_target = infer_query_scenario_target(intent_text)
    topic_boundary_mode = infer_topic_boundary_mode(payload.topic, intent_text)
    return {
        "query_text": weighted_query,
        "intent_text": intent_text,
        "intent_tokens": tokenize(intent_text),
        "negative_tokens": tokenize(negative_text),
        "primary_focus": primary_focus.lower(),
        "highlighted_phrases": [phrase.lower() for phrase in highlighted_phrases],
        "input_completeness": input_completeness,
        "topic_label": topic_label,
        "platform": payload.platform,
        "retrieval_intent": retrieval_intent,
        "scenario_target": scenario_target,
        "topic_boundary_mode": topic_boundary_mode,
    }


def source_type_bias(source_type: str, retrieval_intent: str) -> float:
    if retrieval_intent == "fact":
        if source_type == "document":
            return 0.18
        if source_type == "conversation_insight":
            return 0.04
        if source_type == "conversation_evidence":
            return -0.04
        return 0.0
    if retrieval_intent == "business":
        if source_type == "conversation_insight":
            return 0.18
        if source_type == "document":
            return 0.06
        if source_type == "conversation_evidence":
            return 0.02
        return 0.0
    if source_type == "document":
        return 0.1
    if source_type == "conversation_insight":
        return 0.1
    if source_type == "conversation_evidence":
        return -0.02
    return 0.0


def rule_based_rerank_score(chunk: dict[str, Any], query_bundle: dict[str, Any]) -> float:
    haystack = " ".join(
        [
            chunk.get("title", ""),
            " ".join(chunk.get("headingPath", [])),
            chunk.get("content", ""),
            " ".join(chunk.get("keywords", [])),
            chunk.get("statementType", ""),
            chunk.get("intentType", ""),
        ]
    ).lower()
    intent_tokens = query_bundle["intent_tokens"]
    negative_tokens = query_bundle["negative_tokens"]

    score = 0.0
    score += sum(0.22 for token in intent_tokens if token in haystack)
    score -= sum(0.3 for token in negative_tokens if token in haystack)

    primary_focus = query_bundle.get("primary_focus", "")
    if primary_focus and primary_focus in haystack:
        score += 0.9

    highlighted_phrases = query_bundle.get("highlighted_phrases", [])
    for phrase in highlighted_phrases:
        if phrase in haystack:
            score += 0.7

    if chunk.get("sourceType") == "conversation_evidence":
        score -= 0.08
    score += source_type_bias(chunk.get("sourceType", ""), query_bundle.get("retrieval_intent", "balanced"))
    return score


def extract_negative_constraints(text: str) -> str:
    matches = re.findall(r"(?:不要|别|不想|不强调|不要总讲|不要总是强调)([^。；，,\n]{1,18})", text)
    return " ".join(item.strip() for item in matches if item.strip())


def extract_highlight_phrases(text: str) -> list[str]:
    matches = re.findall(r"(?:重点讲|重点是|想表达|想突出|突出|强调)([^。；，,\n]{1,18})", text)
    return [item.strip().lower() for item in matches if len(item.strip()) >= 2]

def resolve_dashscope_base_url() -> str:
    return os.getenv("DASHSCOPE_BASE_URL", DEFAULT_DASHSCOPE_BASE_URL).rstrip("/")


def resolve_embedding_model() -> str:
    return os.getenv("DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v4")


def resolve_generation_model() -> str:
    return os.getenv("DASHSCOPE_GENERATION_MODEL", "qwen-plus")


def resolve_dashscope_api_key() -> str:
    return os.getenv("DASHSCOPE_API_KEY", "")


def request_embeddings(texts: list[str]) -> list[list[float]]:
    api_key = resolve_dashscope_api_key()
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY 未配置")

    headers = {"Content-Type": "application/json"}
    headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": resolve_embedding_model(),
        "input": texts,
    }

    with httpx.Client(timeout=120) as client:
        response = client.post(f"{resolve_dashscope_base_url()}/embeddings", headers=headers, json=payload)
        if response.status_code >= 400:
            raise RuntimeError(f"DashScope embedding 请求失败: {response.status_code} {response.text}")
        data = response.json()

    if isinstance(data, dict) and isinstance(data.get("data"), list):
        return [item["embedding"] for item in data["data"]]
    if isinstance(data, dict) and isinstance(data.get("embeddings"), list):
        return data["embeddings"]
    raise RuntimeError("embedding API 返回格式无法识别")
