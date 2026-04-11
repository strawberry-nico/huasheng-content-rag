import fs from "node:fs";
import path from "node:path";
import { execFileSync } from "node:child_process";

const cwd = process.cwd();
const DOC_ROOT = path.join(cwd, "doc");
const CONVERSATION_ROOTS = [
  path.join(cwd, "conversation"),
  path.join(DOC_ROOT, "conversation"),
];
const STRUCTURED_CHUNK_MAX_LEN = 760;
const CONVERSATION_EVIDENCE_MAX_LEN = 560;
const CONVERSATION_INSIGHT_MAX_LEN = 420;
const MIN_CHUNK_LEN = 18;
const SOURCE_FORMATS = new Set([".md", ".txt", ".docx", ".pdf"]);

let chunkCounter = 0;

await main();

async function main() {
  const chunks = await buildKnowledgeChunks();
  const outputPayload = {
    source: "doc/* + conversation/*",
    generatedAt: new Date().toISOString(),
    chunkCount: chunks.length,
    chunks,
  };

  fs.writeFileSync(path.join(cwd, "knowledge-data.generated.json"), JSON.stringify(outputPayload, null, 2), "utf8");
  console.log(`Generated ${chunks.length} knowledge chunks from doc sources`);
}

async function buildKnowledgeChunks() {
  if (!fs.existsSync(DOC_ROOT)) {
    throw new Error(`未找到知识目录: ${DOC_ROOT}`);
  }

  const conversationRoots = CONVERSATION_ROOTS.filter((root) => fs.existsSync(root));
  const structuredFiles = walkFiles(DOC_ROOT).filter(
    (file) =>
      !conversationRoots.some((root) => file.startsWith(root)) &&
      SOURCE_FORMATS.has(path.extname(file).toLowerCase())
  );
  const conversationFiles = conversationRoots.flatMap((root) =>
    walkFiles(root).filter((file) => path.extname(file).toLowerCase() === ".md")
  );

  const structuredChunks = structuredFiles.flatMap((file) => buildStructuredChunksFromFile(file));
  const conversationChunksNested = await Promise.all(conversationFiles.map((file) => buildConversationChunksFromFile(file)));
  const conversationChunks = conversationChunksNested.flat();

  return [...structuredChunks, ...conversationChunks].filter((item) => item.content.length >= MIN_CHUNK_LEN);
}

function walkFiles(root) {
  const entries = fs.readdirSync(root, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    if (entry.name.startsWith(".")) continue;
    const fullPath = path.join(root, entry.name);
    if (entry.isDirectory()) {
      files.push(...walkFiles(fullPath));
    } else {
      files.push(fullPath);
    }
  }
  return files.sort((a, b) => a.localeCompare(b, "zh-CN"));
}

function buildStructuredChunksFromFile(filePath) {
  const rawText = extractText(filePath);
  const text = normalizeDocumentText(rawText);
  if (!text) return [];

  const relativePath = path.relative(cwd, filePath);
  const sourceFormat = path.extname(filePath).slice(1).toLowerCase() || "txt";
  const title = path.basename(filePath, path.extname(filePath));
  const headingPath = buildHeadingPath(relativePath, title);
  const blocks = splitStructuredText(text, STRUCTURED_CHUNK_MAX_LEN);

  return blocks.map((content, index) => {
    const contextText = `${title} ${headingPath.join(" ")} ${content}`;
    return createChunk({
      title: blocks.length > 1 ? `${title}（资料片段${index + 1}）` : title,
      headingPath,
      content,
      sourceType: "document",
      sourceFile: relativePath,
      sourceFormat,
      sourceLabel: title,
      publicLevel: inferPublicLevel(contextText),
      topics: inferTopics(contextText),
      businessStages: inferBusinessStages(contextText),
      scenes: inferScenes(contextText),
      conceptType: inferConceptType(contextText),
      executionSite: inferExecutionSite(contextText),
      pilotContext: inferPilotContext(contextText),
      businessScenario: inferBusinessScenario(contextText),
      knowledgeTypes: inferKnowledgeTypes(contextText),
      keywords: extractKeywords(contextText),
    });
  });
}

async function buildConversationChunksFromFile(filePath) {
  const markdown = fs.readFileSync(filePath, "utf8");
  const relativePath = path.relative(cwd, filePath);
  const title = extractConversationTitle(markdown, filePath);
  const headingPath = ["对话知识", title];
  const parsed = parseConversationKnowledgeDocument(markdown);
  const chunks = [];
  const groupMap = new Map();

  const insightEntries = [
    ...parsed.knowledgeEntries,
    ...parsed.conceptEntries.map((item) => ({
      title: item.title,
      content: item.content,
      statementType: "evidence",
      intentType: "fact_grounding",
      certainty: "medium",
      decisionStatus: "tentative",
      knowledgeTypes: ["fact"],
    })),
    ...parsed.quoteEntries.map((item) => ({
      title: item.title,
      content: `对话表达：${item.content}`,
      statementType: "discussion",
      intentType: "goal_alignment",
      certainty: "low",
      decisionStatus: "tentative",
      knowledgeTypes: ["expression_rule"],
    })),
  ];

  for (const [index, entry] of insightEntries.entries()) {
    const content = splitContent(entry.content, STRUCTURED_CHUNK_MAX_LEN)[0] || "";
    if (!content) continue;
    const contextText = `${title} ${entry.title} ${content}`;
    const groupKey = buildConversationEntryKey(entry.title, index);
    const groupId = getOrCreateConversationGroupId(groupMap, relativePath, groupKey);
    chunks.push(createChunk({
      title: `${title}｜${entry.title}`,
      headingPath,
      content,
      sourceType: "conversation_insight",
      sourceFile: relativePath,
      sourceFormat: "md",
      sourceLabel: title,
      sourceTimeRange: "",
      conversationSpeakers: [],
      conversationGroupId: groupId,
      statementType: entry.statementType,
      intentType: entry.intentType,
      certainty: entry.certainty,
      decisionStatus: entry.decisionStatus,
      publicLevel: "internal",
      topics: inferTopics(contextText),
      businessStages: inferBusinessStages(contextText),
      scenes: [...new Set(["sales_qa", ...inferScenes(contextText)])],
      conceptType: inferConceptType(contextText),
      executionSite: inferExecutionSite(contextText),
      pilotContext: inferPilotContext(contextText),
      businessScenario: inferBusinessScenario(contextText),
      knowledgeTypes: mergeKnowledgeTypes(["fact", ...entry.knowledgeTypes, ...inferKnowledgeTypes(contextText)]),
      keywords: extractKeywords(contextText),
    }));
  }

  for (const [index, entry] of parsed.evidenceEntries.entries()) {
    const content = splitContent(entry.content, CONVERSATION_EVIDENCE_MAX_LEN)[0] || "";
    if (!content) continue;
    const contextText = `${title} ${entry.title} ${content}`;
    const groupKey = buildConversationEntryKey(entry.title, index);
    const groupId = getOrCreateConversationGroupId(groupMap, relativePath, groupKey);
    chunks.push(createChunk({
      title: `${title}｜${entry.title}`,
      headingPath,
      content,
      sourceType: "conversation_evidence",
      sourceFile: relativePath,
      sourceFormat: "md",
      sourceLabel: title,
      sourceTimeRange: "",
      conversationSpeakers: [],
      conversationGroupId: groupId,
      statementType: "evidence",
      intentType: "fact_grounding",
      certainty: "medium",
      decisionStatus: "discussing",
      publicLevel: "internal",
      topics: inferTopics(contextText),
      businessStages: inferBusinessStages(contextText),
      scenes: [...new Set(["sales_qa", ...inferScenes(contextText)])],
      conceptType: inferConceptType(contextText),
      executionSite: inferExecutionSite(contextText),
      pilotContext: inferPilotContext(contextText),
      businessScenario: inferBusinessScenario(contextText),
      knowledgeTypes: mergeKnowledgeTypes(["evidence", ...inferKnowledgeTypes(contextText)]),
      keywords: extractKeywords(contextText),
    }));
  }

  return chunks;
}

function buildConversationEntryKey(title, index) {
  const normalized = cleanText(title || "")
    .toLowerCase()
    .replace(/[^\u4e00-\u9fa5a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
  return normalized || `entry-${index + 1}`;
}

function getOrCreateConversationGroupId(groupMap, relativePath, key) {
  if (!groupMap.has(key)) {
    groupMap.set(key, buildConversationGroupId(relativePath, key));
  }
  return groupMap.get(key);
}

function extractText(filePath) {
  const ext = path.extname(filePath).toLowerCase();
  if (ext === ".md" || ext === ".txt") return fs.readFileSync(filePath, "utf8");
  if (ext === ".docx") return runExtractor("textutil", ["-convert", "txt", "-stdout", filePath]);
  if (ext === ".pdf") return runExtractor("pdftotext", [filePath, "-"]);
  return "";
}

function runExtractor(command, args) {
  try {
    return execFileSync(command, args, { encoding: "utf8", maxBuffer: 20 * 1024 * 1024 });
  } catch (error) {
    console.warn(`[build-knowledge] extract failed: ${command} ${args.join(" ")} -> ${error.message}`);
    return "";
  }
}

function normalizeDocumentText(text) {
  return text
    .replace(/\r\n/g, "\n")
    .replace(/\f/g, "\n")
    .replace(/\u200f|\u200e|\u2060/g, "")
    .replace(/Syntax Warning:[^\n]*\n/g, "")
    .replace(/十万级/g, "D 级")
    .replace(/万级/g, "C 级")
    .replace(/[ \t]+\n/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .split("\n")
    .map((line) => line.replace(/^[•·▪●◦]/, "-").trim())
    .filter((line, index, list) => {
      if (!line) return true;
      const next = list[index + 1] || "";
      if (line === next && line.length <= 40) return false;
      return true;
    })
    .join("\n")
    .trim();
}

function splitStructuredText(text, maxLen) {
  const paragraphs = text
    .split(/\n{2,}/)
    .map((part) => cleanText(part))
    .filter((part) => part.length >= MIN_CHUNK_LEN);

  if (!paragraphs.length) return [];

  const chunks = [];
  let current = "";

  for (const paragraph of paragraphs) {
    const next = current ? `${current}\n${paragraph}` : paragraph;
    if (next.length <= maxLen) {
      current = next;
      continue;
    }
    if (current) chunks.push(current);
    if (paragraph.length <= maxLen) {
      current = paragraph;
    } else {
      const parts = splitContent(paragraph, maxLen);
      chunks.push(...parts.slice(0, -1));
      current = parts[parts.length - 1] || "";
    }
  }

  if (current) chunks.push(current);
  return chunks.filter((part) => part.length >= MIN_CHUNK_LEN);
}

function parseConversationKnowledgeDocument(markdown) {
  const sections = splitMarkdownSections(markdown);
  const knowledgeEntries = [];
  const evidenceEntries = [];
  const conceptEntries = [];
  const quoteEntries = [];

  for (const section of sections) {
    const body = section.body.trim();
    if (!body) continue;

    const transcriptItems = parseTranscriptItems(body);
    const conceptItems = parseConceptItems(body);
    const quoteItems = parseQuoteItems(body);
    const knowledgeBlocks = parseKnowledgeBlocks(body);

    const isInfoSection = /对话信息|智能总结/.test(section.heading);
    const isTranscriptSection = /对话实录/.test(section.heading) || transcriptItems.length >= 2;
    const isConceptSection = /核心概念/.test(section.heading) || conceptItems.length >= 2;
    const isQuoteSection = /对话金句|金句/.test(section.heading) || quoteItems.length >= 1;

    if (isTranscriptSection) {
      evidenceEntries.push(...transcriptItems);
      continue;
    }
    if (isConceptSection) {
      conceptEntries.push(...conceptItems);
      continue;
    }
    if (isQuoteSection) {
      quoteEntries.push(...quoteItems);
      continue;
    }
    if (!isInfoSection) {
      knowledgeEntries.push(...knowledgeBlocks);
    }
  }

  return {
    knowledgeEntries: dedupeEntries(knowledgeEntries),
    evidenceEntries: dedupeEntries(evidenceEntries),
    conceptEntries: dedupeEntries(conceptEntries),
    quoteEntries: dedupeEntries(quoteEntries),
  };
}

function splitMarkdownSections(markdown) {
  const lines = markdown.replace(/\r\n/g, "\n").split("\n");
  const sections = [];
  let current = { heading: "", lines: [] };

  for (const rawLine of lines) {
    const line = rawLine.trimEnd();
    if (/^#\s+/.test(line)) continue;
    const match = line.match(/^#{2,4}\s+(.+)$/);
    if (match) {
      if (current.heading || current.lines.length) {
        sections.push({ heading: cleanText(current.heading), body: current.lines.join("\n").trim() });
      }
      current = { heading: match[1], lines: [] };
      continue;
    }
    current.lines.push(line);
  }

  if (current.heading || current.lines.length) {
    sections.push({ heading: cleanText(current.heading), body: current.lines.join("\n").trim() });
  }
  return sections;
}

function parseKnowledgeBlocks(text) {
  const lines = text.split("\n");
  const blocks = [];
  let currentTitle = "";
  let currentLines = [];
  let introLines = [];

  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line) continue;
    if (/^#/.test(line)) continue;
    if (/^- \*\*授课时间\*\*：/.test(line) || /^- \*\*领域\*\*：/.test(line)) continue;

    const headingMatch = line.match(/^\*\*(.+?)\*\*$/);
    if (headingMatch) {
      if (currentTitle && currentLines.length) blocks.push(makeKnowledgeEntry(currentTitle, currentLines));
      currentTitle = headingMatch[1];
      currentLines = [];
      continue;
    }

    if (!currentTitle) {
      introLines.push(line.replace(/^[*-]\s*/, ""));
      continue;
    }

    currentLines.push(line.replace(/^[*-]\s*/, ""));
  }

  if (introLines.length) blocks.unshift(makeKnowledgeEntry("业务概述", introLines));
  if (currentTitle && currentLines.length) blocks.push(makeKnowledgeEntry(currentTitle, currentLines));

  return blocks.filter((item) => item.content.length >= MIN_CHUNK_LEN);
}

function parseTranscriptItems(text) {
  const lines = text.split("\n");
  const items = [];
  let currentTitle = "";
  let currentLines = [];

  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line) continue;
    const titleMatch = line.match(/^\*\*(.+?)\*\*$/);
    if (titleMatch) {
      if (currentTitle && currentLines.length) {
        items.push({ title: cleanText(currentTitle), content: cleanConversationText(currentLines.join(" ")) });
      }
      currentTitle = titleMatch[1];
      currentLines = [];
      continue;
    }
    if (currentTitle) currentLines.push(line);
  }

  if (currentTitle && currentLines.length) {
    items.push({ title: cleanText(currentTitle), content: cleanConversationText(currentLines.join(" ")) });
  }

  return items.filter((item) => item.content.length >= MIN_CHUNK_LEN && !isNoiseTurn(item.content));
}

function parseConceptItems(text) {
  return text
    .split("\n")
    .map((line) => line.trim())
    .map((line) => line.match(/^- \*\*(.+?)\*\*[:：]\s*(.+)$/))
    .filter(Boolean)
    .map((match) => ({
      title: cleanText(match[1]),
      content: cleanText(match[2]),
    }))
    .filter((item) => item.content.length >= MIN_CHUNK_LEN);
}

function parseQuoteItems(text) {
  return text
    .split("\n")
    .map((line) => line.trim())
    .map((line) => line.match(/^- [“"](.*?)[”"]。?$/))
    .filter(Boolean)
    .map((match, index) => ({
      title: `对话金句${index + 1}`,
      content: cleanText(match[1]),
    }))
    .filter((item) => item.content.length >= 6);
}

function makeKnowledgeEntry(title, lines) {
  const content = cleanText(lines.join("\n"));
  const signals = analyzeConversationSignals(collectSignalSentences(content));
  return {
    title: cleanText(title),
    content,
    statementType: inferStatementType(signals),
    intentType: inferIntentType(signals),
    certainty: inferCertainty(signals),
    decisionStatus: inferDecisionStatus(signals),
    knowledgeTypes: inferConversationKnowledgeTypes(content),
  };
}

function dedupeEntries(entries) {
  const seen = new Set();
  return entries.filter((item) => {
    const key = `${item.title}||${item.content}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function cleanConversationText(text) {
  return cleanText(
    text
      .replace(/^[，。；、\s]+/, "")
      .replace(/[啊嗯哦呃]+(?=[，。；、\s]|$)/g, " ")
      .replace(/\bAI\b/gi, "AI")
  );
}

function isNoiseTurn(text) {
  if (text.length <= 3) return true;
  if (/^(好|嗯|行|哦|啊|对|好的|知道了|收到)$/.test(text)) return true;
  if (/^[\W_。、，；：？！…\s]+$/.test(text)) return true;
  if (/(点个外卖|跟我妈说一声|我不回去了|你在哪儿呢|先到这吧|哇塞|尝尝刘总的茶)/.test(text)) return true;
  return false;
}

function extractConversationTitle(markdown, filePath) {
  const firstLine = markdown.replace(/\r\n/g, "\n").split("\n").map((line) => line.trim()).find(Boolean);
  const cleaned = firstLine ? cleanText(firstLine.replace(/^#+\s*/, "")) : "";
  return cleaned || path.basename(filePath, path.extname(filePath));
}

function collectSignalSentences(text) {
  return splitSentences(text)
    .map((sentence) => cleanConversationSentence(sentence))
    .filter((sentence) => sentence && sentence.length >= 8 && !isLowValueSentence(sentence));
}

function splitSentences(text) {
  return text.split(/(?<=[。！？；])/).map((item) => item.trim()).filter(Boolean);
}

function cleanConversationSentence(text) {
  return cleanText(
    text
      .replace(/^(就是|其实|那个|这个|然后|所以|那就|我们来看看)/, "")
      .replace(/(哈哈哈|なるほど|哇塞|哦行下一个|天猫精灵)/g, " ")
  );
}

function isLowValueSentence(text) {
  if (text.length < 8) return true;
  if (/^(好|行|嗯|对|好的|知道了)/.test(text)) return true;
  if (/(我记一下|我搜一下|到时候我改一下|先到这吧)$/.test(text)) return true;
  if (/(点个外卖|我不回去了|你在哪儿呢|尝尝刘总的茶|哈哈哈)/.test(text)) return true;
  if (/^[0-9\s~\-]+$/.test(text)) return true;
  return false;
}

function isMeaningfulSegment(segment) {
  const meaningfulTurns = segment.filter((turn) => !isLowValueSentence(turn.content));
  if (meaningfulTurns.length < 2) return false;
  const joined = meaningfulTurns.map((turn) => turn.content).join(" ");
  return /([一-龥]{4,}|[0-9]+[%万米人套项])/.test(joined);
}

function inferConversationInsightTitle(sourceTitle, sentences) {
  const candidates = [
    { pattern: /一站式|全流程|交钥匙/, title: "一站式与全流程价值" },
    { pattern: /中试|放大|量产|小试/, title: "中试放大与量产衔接" },
    { pattern: /自动换卷|裁切|克重|全检|成品率|合格率/, title: "设备能力与质量成本优势" },
    { pattern: /培训|操作人员|陪产/, title: "培训与交付落地能力" },
    { pattern: /知识库|AI|文案|痛点|素材/, title: "AI知识库理解业务的缺口" },
    { pattern: /客户|药企|学校|CRO|院校/, title: "客户对象与业务场景判断" },
  ];

  const haystack = sentences.join(" ");
  const matched = candidates.find((item) => item.pattern.test(haystack));
  return matched ? `${sourceTitle}｜${matched.title}` : `${sourceTitle}｜讨论片段`;
}

function selectSentencesByPattern(sentences, pattern, limit) {
  return sentences.filter((item) => pattern.test(item)).slice(0, limit);
}

function inferConversationKnowledgeTypes(text) {
  const types = ["fact"];
  if (/(痛点|问题|卡在|失败|浪费|不放心|推诿|风险)/.test(text)) types.push("pain_point");
  if (/(提供|支持|培训|解决|做到|覆盖|全流程|一站式|试机)/.test(text)) types.push("solution");
  if (/(文案|表达|话术|置顶视频|讲故事)/.test(text)) types.push("expression_rule");
  if (/(不要|不能|筛选|别那样做)/.test(text)) types.push("compliance_rule");
  return mergeKnowledgeTypes(types);
}

function analyzeConversationSignals(sentences) {
  return {
    focus: sentences.slice(0, 2),
    intent: selectSentencesByPattern(sentences, /(想|希望|打算|准备|要做|优化|加进|放到|讲|表达|展示)/, 2),
    proposal: selectSentencesByPattern(sentences, /(建议|可以|应该|不如|先|再|提供|做到|支持|负责|加进)/, 2),
    concern: selectSentencesByPattern(sentences, /(不能|不要|别|不放心|推诿|浪费|失败|风险|跟不上|筛选|卡在)/, 2),
    confirmed: selectSentencesByPattern(sentences, /(这是重点|这个重点|先按照|确认|核心就是|就按这个|就这么定|定下来|咱们主要是|目的一个是|先出五个版本|这几个点全检是重点)/, 2),
    evidence: selectSentencesByPattern(sentences, /([0-9]+[%万米人套项]|D 级|C 级|GMP|博士|专利|90%|60%|12000平米|30余年)/, 2),
  };
}

function inferStatementType(signals) {
  if (signals.confirmed.length && !signals.concern.length) return "decision";
  if (signals.proposal.length) return "proposal";
  if (signals.concern.length) return "objection";
  if (signals.evidence.length) return "evidence";
  return "discussion";
}

function inferIntentType(signals) {
  if (signals.proposal.length) return "solution_design";
  if (signals.concern.length) return "risk_identification";
  if (signals.intent.length) return "goal_alignment";
  if (signals.evidence.length) return "fact_grounding";
  return "discussion";
}

function inferCertainty(signals) {
  if (signals.confirmed.length && signals.evidence.length) return "high";
  if (signals.proposal.length || signals.concern.length || signals.evidence.length || signals.confirmed.length) return "medium";
  return "low";
}

function inferDecisionStatus(signals) {
  if (signals.confirmed.length && !signals.concern.length) return "confirmed";
  if (signals.confirmed.length && signals.concern.length) return "tentative";
  if (signals.proposal.length) return "tentative";
  if (signals.concern.length) return "discussing";
  return "discussing";
}

function buildConversationGroupId(relativePath, timeRange) {
  const safePath = relativePath.replace(/[\\/.\s]+/g, "-");
  const safeRange = timeRange.replace(/[:]+/g, "-");
  return `conv-${safePath}-${safeRange}`;
}

function buildHeadingPath(relativePath, title) {
  const dirName = path.dirname(relativePath);
  if (dirName === "doc") return ["资料知识", title];
  const folders = dirName.split(path.sep).filter(Boolean).slice(1);
  return ["资料知识", ...folders, title];
}

function createChunk(fields) {
  return {
    id: `kb-${String(++chunkCounter).padStart(4, "0")}`,
    title: cleanText(fields.title || "未命名知识"),
    headingPath: Array.isArray(fields.headingPath) ? fields.headingPath.map((item) => cleanText(item)).filter(Boolean) : [],
    content: cleanText(fields.content || ""),
    triggerCondition: cleanText(fields.triggerCondition || ""),
    usageRule: cleanText(fields.usageRule || ""),
    sourceModule: cleanText(fields.sourceModule || ""),
    confidence: cleanText(fields.confidence || ""),
    topics: mergeList(fields.topics),
    businessStages: mergeList(fields.businessStages),
    scenes: mergeList(fields.scenes),
    knowledgeTypes: mergeKnowledgeTypes(fields.knowledgeTypes),
    publicLevel: normalizePublicLevel(fields.publicLevel) || "public",
    keywords: mergeList(fields.keywords),
    sourceType: cleanText(fields.sourceType || "document"),
    sourceFile: cleanText(fields.sourceFile || ""),
    sourceFormat: cleanText(fields.sourceFormat || ""),
    sourceLabel: cleanText(fields.sourceLabel || ""),
    sourceTimeRange: cleanText(fields.sourceTimeRange || ""),
    conceptType: cleanText(fields.conceptType || ""),
    executionSite: cleanText(fields.executionSite || ""),
    pilotContext: cleanText(fields.pilotContext || ""),
    businessScenario: cleanText(fields.businessScenario || ""),
    conversationSpeakers: mergeList(fields.conversationSpeakers),
    conversationGroupId: cleanText(fields.conversationGroupId || ""),
    statementType: cleanText(fields.statementType || ""),
    intentType: cleanText(fields.intentType || ""),
    certainty: cleanText(fields.certainty || ""),
    decisionStatus: cleanText(fields.decisionStatus || ""),
  };
}

function cleanText(text) {
  return String(text || "")
    .replace(/\*\*/g, "")
    .replace(/`/g, "")
    .replace(/\|/g, " | ")
    .replace(/\bCIP\b/gi, "自动清洗站")
    .replace(/CIP系统/gi, "自动清洗站")
    .replace(/CIP清洗/gi, "自动清洗站")
    .replace(/自动清洗站\s*清洗站/g, "自动清洗站")
    .replace(/自动清洗站\s*自动清洗站/g, "自动清洗站")
    .replace(/\s+/g, " ")
    .replace(/\s*\n\s*/g, "\n")
    .trim();
}

function splitContent(text, maxLen) {
  if (text.length <= maxLen) return [text];

  const parts = [];
  const segments = text.split("\n");
  let current = "";

  for (const segment of segments) {
    const next = current ? `${current}\n${segment}` : segment;
    if (next.length <= maxLen) {
      current = next;
      continue;
    }

    if (current) parts.push(current);
    if (segment.length <= maxLen) {
      current = segment;
      continue;
    }

    const sentences = segment.split(/(?<=[。！？；])/);
    let sentenceBuffer = "";
    for (const sentence of sentences) {
      const sentenceNext = sentenceBuffer ? `${sentenceBuffer}${sentence}` : sentence;
      if (sentenceNext.length <= maxLen) {
        sentenceBuffer = sentenceNext;
      } else {
        if (sentenceBuffer) parts.push(sentenceBuffer.trim());
        sentenceBuffer = sentence;
      }
    }
    current = sentenceBuffer.trim();
  }

  if (current) parts.push(current);
  return parts.filter(Boolean);
}

function mergeList(items) {
  return [...new Set((Array.isArray(items) ? items : [items]).map((item) => cleanText(item)).filter(Boolean))];
}

function mergeKnowledgeTypes(items) {
  const allowed = new Set(["fact", "pain_point", "solution", "evidence", "expression_rule", "compliance_rule"]);
  return mergeList(items).filter((item) => allowed.has(item));
}

function normalizePublicLevel(value) {
  const normalized = cleanText(value || "").toLowerCase();
  if (["public", "restricted", "internal"].includes(normalized)) return normalized;
  return "";
}

function inferTopics(text) {
  const topicMap = {
    pilot: ["中试", "放大", "共享基地", "中试基地", "转化平台"],
    equipment: ["设备", "涂布机", "切片机", "清洗站", "生产线", "包装机", "自动化"],
    material: ["包材", "压花膜", "TPU", "弹力布", "外包装袋", "辅料"],
    hospital: ["院内制剂", "备案", "药剂科", "医院"],
    strength: ["华胜", "客户", "出口", "院士", "中医科学院", "专利", "30余年"],
    onestop: ["一站式", "交钥匙", "全链路", "解决方案", "代工", "技术服务"],
    training: ["培训", "实践基地", "操作工人", "陪产", "维保"],
    tdts: ["透皮", "经皮", "贴剂", "凝胶膏", "热熔胶", "溶剂胶"],
  };

  return Object.entries(topicMap)
    .filter(([, keywords]) => keywords.some((keyword) => text.includes(keyword)))
    .map(([topic]) => topic);
}

function inferBusinessStages(text) {
  const stageMap = {
    brand: ["定位", "品牌", "客户基础", "出口", "专利", "30余年"],
    consultation: ["需求", "客户", "咨询", "适合", "能不能", "经验"],
    pilot_service: ["中试", "小试", "放大", "转化", "试验"],
    equipment_selection: ["设备", "选型", "产能", "工艺路线", "生产线"],
    compliance: ["GMP", "备案", "核查", "申报", "验证", "合规"],
    delivery: ["交付", "投产", "上线", "陪产", "培训"],
    aftersales: ["维保", "售后", "培训", "驻厂", "维护"],
    content: ["内容", "表达", "用词", "禁用", "合规红线", "素材", "文案", "知识库", "AI"],
  };

  return Object.entries(stageMap)
    .filter(([, keywords]) => keywords.some((keyword) => text.includes(keyword)))
    .map(([stage]) => stage);
}

function inferScenes(text) {
  const sceneMap = {
    equipment_video: ["设备", "运行", "自动化", "切片机", "涂布机", "生产线"],
    technical_talk: ["技术", "工艺", "参数", "处方", "放大"],
    base_showcase: ["基地", "车间", "洁净", "实验室", "平台"],
    client_visit: ["客户", "来访", "参观", "接待", "合作"],
    hospital_consultation: ["院内制剂", "医院", "备案", "药剂科"],
    sales_qa: ["选型", "供应商", "成本", "解决方案", "试机", "客户痛点"],
    aftersales_qa: ["维保", "培训", "驻厂", "陪产", "维护"],
    industry_content: ["行业", "趋势", "市场", "赛道", "政策", "宣传视频", "讲故事"],
  };

  return Object.entries(sceneMap)
    .filter(([, keywords]) => keywords.some((keyword) => text.includes(keyword)))
    .map(([scene]) => scene);
}

function inferConceptType(text) {
  if (text.includes("试机") || text.includes("试运行")) return "action";
  if (/(中试|放大|中试基地|中试车间)/.test(text)) return "stage";
  return "";
}

function inferExecutionSite(text) {
  if (/(中试基地|中试车间|中试平台)/.test(text)) return "pilot_base";
  if (/(制造车间|生产车间|设备车间|车间试机|现场试机)/.test(text)) return "production_workshop";
  if (/(客户现场|客户来厂|来厂试机|参观展示区)/.test(text)) return "client_site";
  if (text.includes("实验室")) return "lab";
  return "";
}

function inferPilotContext(text) {
  if (/(中试基地|中试车间|中试平台)/.test(text)) return "explicit_pilot";
  if (text.includes("试机")) return "non_pilot_default";
  return "";
}

function inferBusinessScenario(text) {
  if (text.includes("试机") || text.includes("试运行")) return "trial_run_validation";
  if (/(中试|放大)/.test(text)) return "pilot_scaleup";
  if (/(备案|申报|审评)/.test(text)) return "filing_support";
  return "";
}

function inferKnowledgeTypes(text) {
  const types = [];
  if (/[0-9]/.test(text) || text.includes("客户") || text.includes("院士") || text.includes("专利")) types.push("fact");
  if (text.includes("痛点") || text.includes("失败") || text.includes("问题") || text.includes("风险") || text.includes("浪费")) types.push("pain_point");
  if (text.includes("服务") || text.includes("解决方案") || text.includes("支持") || text.includes("可承接") || text.includes("提供")) types.push("solution");
  if (text.includes("禁用") || text.includes("合规红线") || text.includes("不说") || text.includes("不要") || text.includes("筛选")) types.push("compliance_rule");
  if (text.includes("建议") || text.includes("话术") || text.includes("表达") || text.includes("文案") || text.includes("讲故事")) types.push("expression_rule");
  return mergeKnowledgeTypes(types);
}

function inferPublicLevel(text) {
  if (text.includes("内部使用") || text.includes("请勿外传")) return "internal";
  if (text.includes("禁用") || text.includes("合规红线") || text.includes("谨慎")) return "restricted";
  return "public";
}

function extractKeywords(text) {
  const candidates = [
    "中试", "放大", "基地", "设备", "清洗", "GMP", "备案", "院内制剂", "贴剂", "凝胶膏",
    "热熔胶", "溶剂胶", "包材", "交钥匙", "一站式", "客户", "出口", "培训", "维保", "自动化",
    "成品率", "合格率", "试机", "量产", "CRO", "学校", "药企", "知识库", "AI", "文案",
  ];

  return candidates.filter((keyword) => text.includes(keyword));
}
