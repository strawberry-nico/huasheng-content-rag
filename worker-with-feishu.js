import { getKnowledgeSource, retrieveKnowledge } from "./knowledge-base.js";

const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type",
};

const CONTENT_TYPE_NAMES = {
  oral: "口播脚本",
  title: "标题+封面文案",
  article: "图文正文",
  comment: "评论区置顶话术",
  moments: "朋友圈文案",
};

const DEFAULT_GENERATION_TOP_K = 3;

const COMPANY = {
  slogan: "创百年华胜 造世界精品",
  history: "30余年",
  positioning: "聚焦外用制剂领域，从小试到中试到规模量产，提供全链路交钥匙服务",
  pilotBase: "12000平米开放式共享贴剂中试基地，具备C级、D级洁净区条件",
};

const TOPICS = {
  pilot: {
    label: "中试服务",
    desc: "12000平开放式共享基地，从小试到量产全链路",
    hooks: {
      pharma_all: ["90%的新药项目不是死在临床，是死在中试", "为什么中试失败的锅，总是研发部背？"],
      tdds_pro: ["凝胶膏剂中试放大，黏度每升高10%，失败率增加35%", "同样的处方，实验室95分，中试直接不及格"],
      executive: ["你熬了3个月的处方，放大到100公斤直接分层", "中试失败的锅，为什么总是研发部背？"],
      management: ["自建中试车间1个亿投资，外包只要1/10", "项目延期6个月，隐性成本可能比中试费用还高"],
      procurement: ["最低价中标的中试供应商，最后反而多花30%", "选择中试供应商的5个隐形风险点"],
      compliance: ["中试批记录这3个问题，是现场核查的高频雷区", "从实验室到中试，数据链断裂的常见陷阱"],
      executive_top: ["透皮给药赛道正在复制PD-1的故事，窗口期还剩多久？", "谁掌握了中试能力，谁就卡住了外用制剂的咽喉"],
    },
  },
  equipment: {
    label: "制药设备",
    desc: "贴剂/凝胶膏剂生产线，每天节省21170元",
    hooks: {
      pharma_all: ["买设备便宜30万，运维贵100万，这笔账怎么算？", "为什么头部药企开始把进口设备换成国产？"],
      tdds_pro: ["凝胶膏剂设备选错，换型成本够买半条生产线", "同样的产能，为什么大厂只要一半的设备？"],
      executive: ["选型时只看价格，结果产能利用率只有40%", "贴剂生产线这3个参数，决定了你的换型时间"],
      management: ["一条贴剂生产线300万，产能利用率只有40%，你算过账吗？", "买设备便宜30万，运维贵100万，这笔账怎么算？"],
      procurement: ["设备供应商评估的5个隐形维度，价格表上看不到", "进口VS国产：什么时候选进口，什么时候选国产？"],
      compliance: ["设备DQ/IQ/OQ验证不过，项目延期3个月", "设备变更没备案，现场核查被开缺陷"],
      executive_top: ["设备国产化浪潮下，谁掌握了核心工艺？", "从设备供应商到整体解决方案，行业格局正在重塑"],
    },
  },
  material: {
    label: "包材产品",
    desc: "压花膜/TPU弹力布/外包装袋，全系列配套",
    hooks: {
      pharma_all: ["包材选错，贴剂还没出厂就翘边", "为什么大厂都有自己的包材标准？"],
      tdds_pro: ["换了3家包材供应商，终于找到了不翘边的秘诀", "包材和中试工艺不匹配，量产时问题爆发"],
      executive: ["换了3家包材供应商，终于找到了不翘边的秘诀", "包材和中试工艺不匹配，量产时问题爆发"],
      management: ["包材成本占贴剂总成本的多少？你可能算少了", "包材供应商评估，除了价格还要看什么？"],
      procurement: ["包材采购的隐性成本，比单价高3倍", "如何评估包材供应商的真实产能？"],
    },
  },
  hospital: {
    label: "院内制剂",
    desc: "医院药剂科备案/GMP合规/全程技术服务",
    hooks: {
      pharma_all: ["院内制剂备案，为什么80%的卡在这一步？", "医院自己做贴剂，比买成品省多少钱？"],
      executive: ["院内制剂中试，和商业化生产有什么不同？", "备案资料准备，最容易被忽略的3个细节"],
      management: ["院内制剂项目的ROI怎么算？", "医院备案流程，平均需要多少天？"],
      procurement: ["院内制剂服务供应商怎么选？", "价格最低的供应商，合规风险最高？"],
      compliance: ["院内制剂GMP要求，比想象更严格", "现场核查时，这5个问题必问"],
    },
  },
  strength: {
    label: "企业实力",
    desc: "30余年/370+药企/22%百强占比/出口20+国",
    hooks: {
      pharma_all: ["370多家药企选择的设备供应商，有什么不一样？", "22%的医药百强，都在用这家设备"],
      executive: ["华胜设备用了5年，和进口设备对比如何？", "从雄县机械厂到行业龙头，30年只做一件事"],
      management: ["选择设备供应商，看历史还是看规模？", "30年设备厂家，比新厂贵但值"],
      procurement: ["供应商评估，历史业绩占多少权重？", "为什么大客户都选择华胜？"],
      compliance: ["华胜设备的验证包，审计一次过的秘诀", "30年品质沉淀，合规体系如何建立？"],
      executive_top: ["从雄县机械厂到行业龙头，30余年只做一件事", "出口20多个国家，中国设备的国际认可度"],
    },
  },
  onestop: {
    label: "一站式采购",
    desc: "设备+耗材+辅料+包材+中试，交钥匙工程",
    hooks: {
      pharma_all: ["找10家供应商，不如找1家做全链路的", "一站式采购，能省多少管理成本？"],
      executive: ["分项采购vs一站式，哪个更适合你的项目？", "一站式服务的隐性价值，怎么算？"],
      management: ["交钥匙工程vs分项采购，哪个更适合你？", "全链路服务，出了问题找谁？"],
      procurement: ["供应商数量从10家减到1家，采购成本不降反升？", "一站式服务的隐性价值，怎么算？"],
    },
  },
  training: {
    label: "人才培训",
    desc: "研究生实践基地/设备操作/工艺培训",
    hooks: {
      pharma_all: ["设备买回去不会用？培训比设备更重要", "研究生实践基地，产学研怎么结合？"],
      tdds_pro: ["设备操作培训，为什么一定要有实操环节？", "从实验室到量产，需要哪些技能培训？"],
      management: ["员工培训投入，多久能看到回报？", "校企合作，能解人才短缺之困吗？"],
    },
  },
  tdts: {
    label: "经皮知识",
    desc: "透皮给药技术/行业趋势/专业解读",
    hooks: {
      pharma_all: ["2025年透皮给药迎来爆发，但90%的人没看懂政策信号", "贴剂 VS 凝胶膏剂，哪个才是未来？"],
      tdds_pro: ["为什么你的贴剂贴到患者身上就翘边？", "凝胶膏剂黏度越高，反而越容易失败"],
      executive: ["透皮给药中试，最容易踩的3个坑", "从实验室到临床，透皮贴剂要过几道关？"],
      management: ["透皮给药市场，现在是入场的好时机吗？", "中试外包vs自建，透皮项目怎么选？"],
    },
  },
};

const PLATFORMS = {
  douyin: {
    id: "douyin",
    label: "抖音",
    rhythm: "极快节奏",
    hookInterval: "每5秒",
    sentenceLength: "每句不超过12字",
    emotion: "极高",
    style: "口语化短句、情绪强烈、网络热词",
    structure: "3秒致命钩子 → 15秒痛点放大 → 10秒解决方案 → 5秒强引导",
    tactics: "利用完播率算法，前3秒必须制造认知冲突",
    contentTypes: ["oral", "title"],
    wordCount: { oral: "150-200字", title: "15-25字" },
  },
  shipinhao: {
    id: "shipinhao",
    label: "视频号",
    rhythm: "中等节奏",
    hookInterval: "每15秒",
    sentenceLength: "长短句结合",
    emotion: "中等",
    style: "专业但不刻板，有温度，用'我们'拉近距离",
    structure: "10秒问题引入 → 30秒深度分析 → 15秒案例佐证 → 5秒信任引导",
    tactics: "利用社交关系链，强调'同行都在看'",
    contentTypes: ["oral", "title", "article", "comment", "moments"],
    wordCount: { oral: "200-300字", article: "400-600字", comment: "50-80字", moments: "100-200字", title: "20-30字" },
  },
  xiaohongshu: {
    id: "xiaohongshu",
    label: "小红书",
    rhythm: "轻快节奏",
    hookInterval: "每10秒",
    sentenceLength: "活泼短句",
    emotion: "中高",
    style: "亲和、表情丰富、干货感、朋友口吻",
    structure: "5秒好奇钩子 → 20秒干货输出 → 10秒案例 → 5秒互动引导",
    tactics: "利用收藏心理，强调'建议收藏'、'整理好了'",
    contentTypes: ["article", "title", "comment"],
    wordCount: { article: "300-500字", comment: "40-60字", title: "15-25字" },
  },
  zhihu: {
    id: "zhihu",
    label: "知乎",
    rhythm: "慢节奏",
    hookInterval: "每段落",
    sentenceLength: "长句为主，逻辑严密",
    emotion: "低",
    style: "理性、数据支撑、专业术语",
    structure: "问题引入 → 背景分析 → 深度论证 → 数据佐证 → 总结建议",
    tactics: "利用SEO长尾流量，强调专业性和权威性",
    contentTypes: ["article", "title", "comment"],
    wordCount: { article: "600-1000字", comment: "80-120字", title: "25-40字" },
  },
};

export default {
  async fetch(request, env) {
    if (request.method === "OPTIONS") {
      return new Response(null, { headers: CORS_HEADERS });
    }

    if (request.method !== "POST") {
      return jsonResponse({ success: false, error: "Method not allowed" }, 405);
    }

    try {
      const url = new URL(request.url);
      const action = url.searchParams.get("action") || "generate";
      const body = await request.json();
      const payload = normalizePayload(body);

      if (action === "save") {
        requireEnv(env, ["FEISHU_APP_ID", "FEISHU_APP_SECRET", "FEISHU_BASE_TOKEN", "FEISHU_TABLE_ID"]);
        if (!payload.content?.trim()) {
          throw new Error("缺少可保存的内容");
        }

        const feishu = await saveToFeishu(payload, env);
        return jsonResponse({ success: true, feishu });
      }

      requireEnv(env, ["KIMI_API_KEY"]);
      validatePayload(payload);

      const knowledgeContext = await resolveKnowledgeContext(payload, env);
      const prompt = buildPrompt(payload, knowledgeContext);
      const content = await generateWithKimi(prompt, env);

      let feishu = { saved: false };
      if (hasFeishuConfig(env)) {
        try {
          feishu = await saveToFeishu({ ...payload, content, prompt }, env);
        } catch (error) {
          console.error("飞书保存失败:", error);
          feishu = { saved: false, error: error.message };
        }
      }

      return jsonResponse({
        success: true,
        content,
        promptPreview: prompt.slice(0, 500),
        knowledgeSource: knowledgeContext.source,
        feishu,
      });
    } catch (error) {
      return jsonResponse({ success: false, error: error.message || "服务异常" }, 500);
    }
  },
};

function normalizePayload(body) {
  return {
    topic: typeof body.topic === "string" ? body.topic.trim() : "",
    brief: typeof body.brief === "string" ? body.brief.trim() : "",
    platform: typeof body.platform === "string" ? body.platform.trim() : "",
    types: Array.isArray(body.types) ? body.types.filter((item) => typeof item === "string") : [],
    content: typeof body.content === "string" ? body.content : "",
    rawInput: typeof body.rawInput === "string" ? body.rawInput.trim() : "",
    prompt: typeof body.prompt === "string" ? body.prompt : "",
  };
}

function validatePayload(payload) {
  if (!TOPICS[payload.topic]) {
    throw new Error("主题参数无效");
  }
  if (!payload.brief) {
    throw new Error("请填写视频素材或表达重点");
  }
  if (!PLATFORMS[payload.platform]) {
    throw new Error("平台参数无效");
  }
  if (!payload.types.length) {
    throw new Error("至少选择一种内容类型");
  }

  const allowedTypes = new Set(PLATFORMS[payload.platform].contentTypes);
  for (const type of payload.types) {
    if (!allowedTypes.has(type)) {
      throw new Error(`内容类型不支持当前平台: ${type}`);
    }
  }
}

async function generateWithKimi(prompt, env) {
  const res = await fetch("https://api.moonshot.cn/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${env.KIMI_API_KEY}`,
    },
    body: JSON.stringify({
      model: env.KIMI_MODEL || "moonshot-v1-32k",
      messages: [{ role: "user", content: prompt }],
      temperature: 0.85,
    }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err?.error?.message || `Kimi 请求失败: HTTP ${res.status}`);
  }

  const data = await res.json();
  return data.choices?.[0]?.message?.content || "";
}

function buildPrompt(payload, knowledgeContext) {
  const topic = TOPICS[payload.topic];
  const platform = PLATFORMS[payload.platform];
  const selectedTypes = payload.types.map((type) => CONTENT_TYPE_NAMES[type] || type).join("、");
  const supplementalInput = payload.rawInput
    ? `【补充补充说明】\n${payload.rawInput}\n`
    : "";
  const scenePrompt = buildScenePrompt(payload.topic);
  const knowledgeSection = knowledgeContext.items
    .map((item, index) => {
      const path = item.headingPath?.length ? `\n路径：${item.headingPath.join(" > ")}` : "";
      const score = item.rerankScore != null ? `\n相关度：${item.rerankScore}` : "";
      return `资料${index + 1}｜${item.title}${path}${score}\n${item.content}`;
    })
    .join("\n\n");

  return `【角色设定】
你是华胜品牌内容团队一员，代表华胜对外发声。华胜始创于1993年，30余年来持续深耕外用制剂领域，业务覆盖设备、中试转化、工艺放大与配套服务。中试基地是华胜能力体系中的关键模块之一，但不等于华胜全部业务。

【华胜背景资料】（按相关性自然融入，不要每条都机械堆砌）
- 品牌口号：${COMPANY.slogan}
- 品牌积累：始创于1993年，${COMPANY.history}持续深耕外用制剂领域
- 中试基地：${COMPANY.pilotBase}
- 品牌定位：${COMPANY.positioning}

【内容生成参数】
主题：${topic.label}
视频素材/表达重点：${payload.brief}
平台：${platform.label}
内容类型：${selectedTypes}

【品牌层级要求】
- 默认主体是“华胜”这一整体品牌，不要默认把华胜写成只有中试基地
- 当素材明确展示基地、车间、实验室、中试服务时，可以突出“中试基地”这一能力模块
- 当素材更偏设备、包材、院内制剂、行业认知或一站式服务时，应回到华胜整体解决方案视角
- 可以写“华胜具备中试转化能力”，不要写成“华胜只做中试”

【平台风格要求 - 必须严格执行】
- 节奏：${platform.rhythm}，${platform.hookInterval}必须有一个钩子
- 句式：${platform.sentenceLength}
- 情绪强度：${platform.emotion}
- 语言风格：${platform.style}
- 结构模板：${platform.structure}
- 平台策略：${platform.tactics}

【总口吻原则】
- 优先服从当前平台的表达习惯，但整体保持专业、可信、克制的B端品牌气质
- 可以更口语化，但不要油腻、浮夸、喊口号、过度像泛流量营销号
- 可以有冲突感和传播感，但结论要落在专业判断、业务价值或解决方案上
- 不要为了迎合平台牺牲真实性和专业度

【字数要求】
${Object.entries(platform.wordCount)
  .map(([type, count]) => `- ${CONTENT_TYPE_NAMES[type] || type}：${count}`)
  .join("\n")}

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

${supplementalInput}
【动态知识库召回】
以下资料来自华胜知识库（来源：${knowledgeContext.source}），请优先使用与当前主题、素材描述、平台最相关的证据，不要为了凑数字而全部堆砌：
${knowledgeSection}

【知识使用规则】
- 优先使用上方已召回资料中的事实、场景和表述
- 若召回资料与硬编码背景资料冲突，以召回资料为准
- 若召回资料没有覆盖某个数字、认证级别、设备名称或能力点，就不要自行补全
- 引用数字时，优先解释业务意义，不要孤立堆数字

【本次主题子提示词】
${scenePrompt}

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
- 不要把“10秒问题引入”“30秒深度分析”这类时间结构直接写成标题
- 口播脚本可以自然分段，但分段标题必须像成品口播提纲，而不是时间标签
- 图文正文默认输出成自然正文；如确需小标题，只能用自然语言标题，不能用时间、步骤、模板名
- 标题+封面文案模块只输出标题和封面文案，不输出解释说明
- 评论区置顶话术、朋友圈文案不得大段重复正文原句
- 先写痛点，再给证据，再给价值，再给行动，不要一上来堆品牌荣誉
- 每个模块都要体现与素材和场景相关的关注点，而不是默认套受众模板
- 即使是抖音、小红书等平台，也要避免低质营销腔和过度夸张表达

【内容类型具体要求】
${buildContentRequirements(platform, payload.types)}

【输出格式】
严格按以下内容类型输出，每个模块用【】标注：
${payload.types.map((type) => `【${CONTENT_TYPE_NAMES[type] || type}】`).join("\n")}`;
}

function buildScenePrompt(topicId) {
  const prompts = {
    pilot: `- 本条内容以“中试转化能力”作为核心切口
- 重点讲中试如何连接研发与产业化，不只讲场地和设备
- 优先突出工艺放大、验证衔接、转移效率、项目承接能力
- 表达主体可以是“华胜的中试基地/中试平台”，但结论仍落回华胜整体能力`,
    equipment: `- 本条内容以“设备如何服务工艺放大和稳定量产”作为核心切口
- 不只罗列设备参数，要讲换型效率、验证友好、质量一致性和后续产业化价值
- 若素材不涉及清洗、换型或GMP清洁要求，不要主动展开在线清洗能力
- 表达主体优先是华胜设备能力，而不是把内容全部写成基地介绍`,
    material: `- 本条内容以“包材/耗材如何影响制剂稳定性和量产一致性”作为核心切口
- 重点讲适配性、配套性、稳定性和与工艺协同的价值
- 不要把包材内容硬写成设备介绍或中试基地宣传`,
    hospital: `- 本条内容以“院内制剂开发与备案落地支持”作为核心切口
- 重点讲备案配套、工艺验证、稳定制备、合规支持和协同服务
- 表达主体应是华胜的院内制剂服务能力，不要弱化为单纯设备销售`,
    strength: `- 本条内容以“华胜整体实力与长期积累”作为核心切口
- 重点讲长期深耕、技术沉淀、标准化能力、整体解决方案和产业协同
- 不要默认堆客户数、出口数、专利数等高波动数字，除非召回资料明确支持`,
    onestop: `- 本条内容以“整体解决方案协同价值”作为核心切口
- 重点讲设备、包材、中试、工艺与服务协同如何减少沟通损耗、缩短周期、降低系统性风险
- 避免把一站式理解成简单的产品打包销售`,
    training: `- 本条内容以“人才培训与能力转移”作为核心切口
- 重点讲实操培训、工艺理解、上手效率和团队能力建设
- 不要把培训内容写成泛泛企业宣传`,
    tdts: `- 本条内容以“行业认知 + 产业落地判断”作为核心切口
- 先建立行业认知，再落到中试转化、设备适配、产业落地或合规挑战
- 不要只谈趋势概念，最终要回到华胜能解决什么问题`,
  };

  return prompts[topicId] || `- 以华胜整体品牌视角组织内容
- 根据素材判断更适合突出设备能力、中试转化、工艺理解还是整体解决方案
- 不要把单一能力模块误写成华胜全部业务`;
}

function buildContentRequirements(platform, types) {
  const requirements = [];

  for (const type of types) {
    switch (type) {
      case "oral":
        requirements.push(`【口播脚本 - ${platform.label}版】
- 字数：${platform.wordCount.oral}
- 节奏：${platform.rhythm}，${platform.hookInterval}一个转折点
- 开头：必须从当前素材或表达重点里提炼一个强切口
- 中间：痛点放大+华胜解决方案，优先植入与召回资料一致的华胜背景和证据
- 结尾：围绕当前场景做自然引导，不要写成固定受众分层话术
- 句式：${platform.sentenceLength}
- 输出形态：直接给可读的成品口播，不要输出“第1段”“10秒开场”这类结构标签`);
        break;
      case "title":
        requirements.push(`【标题+封面文案 - ${platform.label}版】
- 数量：3个备选标题 + 1条封面文案
- 标题字数：${platform.wordCount.title}
- 标题要求：${getTitleRequirement(platform.id)}
- 封面文案：10-15字，强调冲突感或悬念
- 输出格式：先列3条标题，再单独列1条封面文案，不要写解释文字`);
        break;
      case "article":
        requirements.push(`【图文正文 - ${platform.label}版】
- 字数：${platform.wordCount.article}
- 结构：${platform.structure}
- 格式：${getArticleFormat(platform.id)}
- 植入：自然融入与召回资料一致的华胜背景，如30余年、基地能力、客户基础等
- 输出形态：正文成品优先，不要使用时间轴标题；如果分小标题，只能写自然语言标题，如“为什么运维成本才是真成本”`);
        break;
      case "comment":
        requirements.push(`【评论区置顶话术 - ${platform.label}版】
- 字数：${platform.wordCount.comment}
- 风格：${getCommentStyle(platform.id)}
- 类型：资料引流型/互动争议型/参观预约型
- 输出形态：一句或两句即可，不要改写整段正文`);
        break;
      case "moments":
        requirements.push(`【朋友圈文案 - ${platform.label}版】
- 字数：${platform.wordCount.moments}
- 结构：场景引入 → 反常识观点 → 华胜植入 → 分层引导
- 人设：华胜团队一员，专业但不高冷
- 配图建议：车间实拍/设备运行/客户案例/团队工作
- 输出形态：像销售或品牌团队会发的成品朋友圈，不要写成新闻稿`);
        break;
      default:
        break;
    }
  }

  return requirements.join("\n\n");
}

function getTitleRequirement(platformId) {
  if (platformId === "douyin") return '数字+冲突+情绪词，例如"90%死在这"、"血亏300万"';
  if (platformId === "xiaohongshu") return "emoji+干货词+收藏引导";
  if (platformId === "zhihu") return "专业术语+完整问题，SEO友好";
  return "专业洞察+价值承诺";
}

function getArticleFormat(platformId) {
  if (platformId === "xiaohongshu") return "多用 emoji 分段，清单式排版";
  if (platformId === "zhihu") return "标题层级清楚，必要时用数据和来源增强可信度";
  return "段落清晰，每段不超过3-5行";
}

function getCommentStyle(platformId) {
  if (platformId === "douyin") return '口语化、紧迫感，适合"评论区扣XX"';
  if (platformId === "xiaohongshu") return '亲和互动，适合"整理好了"';
  if (platformId === "zhihu") return "专业、价值延续、深度引导";
  return "有温度、建立信任";
}

async function resolveKnowledgeContext(payload, env) {
  if (env.RAG_API_URL) {
    try {
      const ragResponse = await fetch(`${env.RAG_API_URL.replace(/\/$/, "")}/retrieve`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          topic: payload.topic,
          platform: payload.platform,
          brief: payload.brief,
          rawInput: payload.rawInput,
          top_k: DEFAULT_GENERATION_TOP_K,
        }),
      });

      if (!ragResponse.ok) {
        throw new Error(`HTTP ${ragResponse.status}`);
      }

      const ragData = await ragResponse.json();
      if (Array.isArray(ragData.results) && ragData.results.length) {
        return {
          source: `RAG:${ragData.source || env.RAG_API_URL}`,
          items: ragData.results,
        };
      }
    } catch (error) {
      console.error("RAG 服务调用失败，回退本地检索:", error);
    }
  }

  return {
    source: `local:${getKnowledgeSource()}`,
    items: retrieveKnowledge({ ...payload, topK: DEFAULT_GENERATION_TOP_K }),
  };
}

async function saveToFeishu(data, env) {
  const { FEISHU_APP_ID, FEISHU_APP_SECRET, FEISHU_BASE_TOKEN, FEISHU_TABLE_ID } = env;

  const tokenRes = await fetch("https://open.feishu.cn/open-apis/auth/v3/app_access_token/internal", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      app_id: FEISHU_APP_ID,
      app_secret: FEISHU_APP_SECRET,
    }),
  });

  const tokenData = await tokenRes.json();
  if (tokenData.code !== 0) {
    throw new Error(`获取飞书token失败: ${tokenData.msg}`);
  }

  const accessToken = tokenData.app_access_token;
  const fields = {
    主题: mapTopic(data.topic),
    受众: "",
    平台: mapPlatform(data.platform),
    内容类型: data.types?.map((type) => mapContentType(type)).join("、") || "",
    原始需求: [data.brief, data.rawInput].filter(Boolean).join("\n\n"),
    生成内容: data.content,
    提示词: data.prompt?.slice(0, 500) || "",
    生成时间: Date.now(),
    状态: "已生成",
  };

  const url = `https://open.feishu.cn/open-apis/bitable/v1/apps/${FEISHU_BASE_TOKEN}/tables/${FEISHU_TABLE_ID}/records`;
  const recordRes = await fetch(url, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${accessToken}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ fields }),
  });

  const recordData = await recordRes.json();
  if (recordData.code !== 0) {
    console.error("飞书API错误详情:", JSON.stringify(recordData));
    throw new Error(`飞书API错误: ${recordData.msg || recordData.error?.message || JSON.stringify(recordData)}`);
  }

  return {
    saved: true,
    recordId: recordData.data?.record?.record_id,
  };
}

function hasFeishuConfig(env) {
  return Boolean(env.FEISHU_APP_ID && env.FEISHU_APP_SECRET && env.FEISHU_BASE_TOKEN && env.FEISHU_TABLE_ID);
}

function requireEnv(env, keys) {
  for (const key of keys) {
    if (!env[key]) {
      throw new Error(`${key} 未配置，请在 Cloudflare Worker 环境变量中设置`);
    }
  }
}

function jsonResponse(payload, status = 200) {
  return new Response(JSON.stringify(payload), {
    status,
    headers: {
      ...CORS_HEADERS,
      "Content-Type": "application/json",
    },
  });
}

function mapTopic(topic) {
  const map = {
    pilot: "🏭 中试服务",
    equipment: "⚙️ 制药设备",
    material: "📦 包材产品",
    hospital: "🏥 院内制剂",
    strength: "⭐ 企业实力",
    onestop: "🛒 一站式采购",
    training: "🎓 人才培训",
    tdts: "📚 经皮知识",
  };
  return map[topic] || topic;
}

function mapPlatform(platform) {
  const map = {
    douyin: "抖音",
    shipinhao: "视频号",
    xiaohongshu: "小红书",
    zhihu: "知乎",
  };
  return map[platform] || platform;
}

function mapContentType(type) {
  return CONTENT_TYPE_NAMES[type] || type;
}
