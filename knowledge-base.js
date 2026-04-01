import { KNOWLEDGE_CHUNKS, KNOWLEDGE_SOURCE } from "./knowledge-data.generated.js";

const TOPIC_LABELS = {
  pilot: "中试服务",
  equipment: "制药设备",
  material: "包材产品",
  hospital: "院内制剂",
  strength: "企业实力",
  onestop: "一站式采购",
  training: "人才培训",
  tdts: "经皮知识",
};

const PLATFORM_SCENE_HINTS = {
  douyin: ["equipment_video", "client_visit", "industry_content"],
  shipinhao: ["technical_talk", "base_showcase", "sales_qa"],
  xiaohongshu: ["industry_content", "base_showcase", "technical_talk"],
  zhihu: ["technical_talk", "industry_content", "sales_qa"],
};

const BRIEF_SCENE_HINTS = {
  equipment_video: ["运行", "运转", "实拍", "设备", "生产线", "自动化", "切片机", "涂布机"],
  technical_talk: ["讲解", "技术", "参数", "工艺", "原理", "放大", "研发"],
  base_showcase: ["基地", "车间", "实验室", "洁净", "平台", "全景"],
  client_visit: ["客户", "参观", "接待", "来访", "会议室"],
  hospital_consultation: ["院内制剂", "医院", "药剂科", "备案"],
  sales_qa: ["能不能", "适合", "怎么选", "选型", "方案", "优势"],
  aftersales_qa: ["维保", "售后", "培训", "驻厂", "陪产", "维护"],
  industry_content: ["趋势", "赛道", "政策", "行业", "市场", "认知"],
};

const ALWAYS_INCLUDE_PATTERNS = [
  /品牌用词.*规则/,
  /合规红线/,
  /内容生产规则/,
];

const DOWNRANK_PATTERNS = [
  /目标受众/,
  /受众说话方式/,
  /实施路径/,
  /短期目标/,
  /中期目标/,
  /长期愿景/,
  /下一步行动清单/,
  /高效内容模板/,
];

export function retrieveKnowledge(payload) {
  const query = buildQuery(payload);
  const inferredScenes = inferScenes(query.text, payload.platform);
  const targetTopK = Number.isFinite(payload.topK) && payload.topK > 0 ? payload.topK : 3;

  const scored = KNOWLEDGE_CHUNKS.map((chunk) => ({
    chunk,
    score: scoreChunk(chunk, query, inferredScenes),
  }))
    .filter((item) => item.score > 0)
    .sort((a, b) => b.score - a.score);

  const selected = [];
  const seenPaths = new Set();

  for (const item of scored) {
    const pathKey = item.chunk.headingPath?.slice(0, 2).join(" > ") || item.chunk.title;
    if (seenPaths.has(pathKey) && item.score < 8) continue;
    selected.push(item.chunk);
    seenPaths.add(pathKey);
    if (selected.length >= targetTopK) break;
  }

  for (const chunk of KNOWLEDGE_CHUNKS) {
    if (selected.some((item) => item.id === chunk.id)) continue;
    if (ALWAYS_INCLUDE_PATTERNS.some((pattern) => pattern.test(chunk.headingPath.join(" ")))) {
      selected.push(chunk);
    }
  }

  return selected.slice(0, Math.max(targetTopK + 2, targetTopK));
}

export function getKnowledgeSource() {
  return KNOWLEDGE_SOURCE;
}

function buildQuery(payload) {
  const topicLabel = TOPIC_LABELS[payload.topic] || payload.topic || "";
  const platform = payload.platform || "";
  const brief = payload.brief || "";
  const rawInput = payload.rawInput || "";
  const text = [topicLabel, platform, brief, rawInput].filter(Boolean).join(" ").toLowerCase();
  const terms = tokenize(text);

  return {
    topic: payload.topic,
    topicLabel,
    platform,
    brief,
    rawInput,
    text,
    terms,
  };
}

function inferScenes(queryText, platform) {
  const scenes = new Set(PLATFORM_SCENE_HINTS[platform] || []);

  for (const [scene, keywords] of Object.entries(BRIEF_SCENE_HINTS)) {
    if (keywords.some((keyword) => queryText.includes(keyword.toLowerCase()))) {
      scenes.add(scene);
    }
  }

  return [...scenes];
}

function scoreChunk(chunk, query, inferredScenes) {
  let score = 0;
  const haystack = [
    chunk.title,
    chunk.headingPath.join(" "),
    chunk.content,
    (chunk.keywords || []).join(" "),
  ]
    .join(" ")
    .toLowerCase();

  if (chunk.topics?.includes(query.topic)) score += 8;
  if (query.topicLabel && haystack.includes(query.topicLabel.toLowerCase())) score += 4;

  const sceneHits = inferredScenes.filter((scene) => chunk.scenes?.includes(scene)).length;
  score += sceneHits * 4;

  for (const term of query.terms) {
    if (term.length < 2) continue;
    if (haystack.includes(term)) score += 1.5;
  }

  if (chunk.publicLevel === "public") score += 0.5;
  if (chunk.knowledgeTypes?.includes("compliance_rule")) score += 1;
  if (chunk.knowledgeTypes?.includes("expression_rule")) score += 1;
  if (DOWNRANK_PATTERNS.some((pattern) => pattern.test(chunk.headingPath.join(" ")))) score -= 3;
  if (chunk.title.includes("五类目标受众说话方式")) score -= 5;

  return score;
}

function tokenize(text) {
  const unique = new Set();
  const phraseMatches = text.match(/[\u4e00-\u9fa5]{2,12}|[a-z0-9-]{2,}/gi) || [];
  for (const token of phraseMatches) unique.add(token.toLowerCase());
  return [...unique];
}
