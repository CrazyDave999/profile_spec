"""Render four landscape SVG statecharts and their offline HTML reader."""
import _bootstrap
from html import escape
from pathlib import Path

ROOT = _bootstrap.ROOT
ASSETS = ROOT / "reports/2026-11-06_sd_systems_flow"
REPORT = ROOT / "reports/2026-11-06_sd_systems_flow.html"

COLORS = {
    "state": ("#f7fafc", "#7892a7"),
    "draft": ("#edf4ff", "#3974bf"),
    "control": ("#fff6e7", "#b47a24"),
    "verify": ("#eaf7f3", "#228978"),
    "commit": ("#f4effb", "#8663b0"),
    "miss": ("#fff0ed", "#c16959"),
}

# Cards describe components; they do not introduce extra execution states.
# Shapes use the shared hidden width h in the Qwen3-8B example.
MODULES = {
    "dflash": [
        ("M1", "Draft model", "基础", "L_d 层 Transformer · block 内双向 attention", "H[B,P,mh] + query IDs[B,q]", "→ Z[B,γ,h] → 共享 LM head → U[B,γ,V]"),
        ("M2", "Context converter · draft 内部", "基础", "Linear(mh,h) + RMSNorm", "H[B,P,mh] → C[B,P,h]", "每层将 C 投影为 prefix K/V；不预测未来 H"),
    ],
    "dspark": [
        ("M1", "Draft backbone", "沿用", "DFlash 式 L_d 层 Transformer + 共享 LM head", "H[B,P,mh] + query IDs[B,q]", "→ Z[B,γ,h]、基础 logits U[B,γ,V]"),
        ("M2", "Markov head · vanilla", "新增", "Embedding(V,r) + Linear(r,V)；r 常用 256", "prev IDs[B] → E_prev[B,r]", "→ bias[B,V]；U_i + bias → token[B]"),
        ("M3", "Confidence head", "新增", "Linear(h+r,1) + 温度校准 + sigmoid", "concat(Z[B,γ,h], E_prev[B,γ,r])", "→ c[B,γ]；cumprod 得前缀存活概率"),
    ],
    "jetspec": [
        ("M1", "Causal draft model", "改动", "L_d 层 Transformer · block 内 causal attention", "真实 H[B,P,mh] + query IDs[B,q]", "→ 各深度 logits U[B,γ,V] → top-k / 构树"),
        ("M2", "Context converter · 特征投影", "沿用", "Linear(mh,h) + RMSNorm；位于 draft 内部", "H[B,P,mh] → C[B,P,h] → 各层 prefix K/V", "仅投影已知特征；不是 JetSpec++ 的预测式 adapter"),
    ],
    "jetspecpp": [
        ("M1", "Draft model", "改训", "5 层 causal Transformer + 共享 LM head", "(H 或 Ĥ)[B,P,mh] + IDs[B,q]", "→ U[B,γ,V]；J0 / J1 复用同一模型"),
        ("M2", "Hidden-state adapter", "新增", "12 层 Transformer：五组深度 1/1/1/4/5", "真实 prefix H[B,P,5h] + IDs[B,Q]", "→ 预测 Ĥ[B,Q,5h]；h=4096 时 5h=20480"),
        ("M3", "Anchor decoder", "新增", "冻结的 target 尾部 2 层 + Norm + LM head", "最深层预测 ĥ[B,Q,h] + prefix KV", "→ 近似 target logits[B,Q,V]"),
    ],
}

MODEL_STATES = {
    "dflash": {"draft": ["M1", "M2"], "proposal": ["M1"]},
    "dspark": {"backbone": ["M1"], "markov": ["M2"], "plan": ["M3"]},
    "jetspec": {"draft": ["M1", "M2"]},
    "jetspecpp": {"foresight": ["M2", "M3"], "refine": ["M1", "M2", "M3"], "fallback": ["M1"]},
}


class Chart:
    def __init__(self, name, description):
        self.name, self.description = name, description
        self.prefix = name.lower().replace("++", "pp")
        self.back, self.edges, self.nodes, self.labels = [], [], [], []

    def text(self, x, y, lines, size=15, anchor="start", color="#314c63", weight=400, layer=None):
        target = self.labels if layer is None else layer
        target.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" fill="{color}" font-weight="{weight}">')
        for i, line in enumerate(lines):
            target.append(f'<tspan x="{x}" dy="{0 if i == 0 else 23}">{escape(line)}</tspan>')
        target.append("</text>")

    def region(self, x, y, w, h, title, subtitle=None):
        self.back.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="22" fill="#fbfcfe" stroke="#bdccd8" stroke-width="1.6" stroke-dasharray="7 5"/>')
        self.text(x + 20, y + 29, [title], size=18, weight=700, layer=self.back)
        if subtitle:
            self.text(x + 20, y + 53, [subtitle], size=14, color="#688096", layer=self.back)

    def state(self, key, x, y, w, h, title, lines, kind="state"):
        fill, stroke = COLORS[kind]
        self.nodes.append(f'<g id="{self.prefix}-{key}" class="state" data-kind="{kind}"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{fill}" stroke="{stroke}" stroke-width="1.8"/><path d="M{x + 1} {y + 43}H{x + w - 1}" stroke="{stroke}" stroke-opacity=".3"/>')
        self.text(x + 16, y + 29, [title], size=22, weight=700, color="#18364e", layer=self.nodes)
        self.text(x + 16, y + 69, lines, size=18, layer=self.nodes)
        badges = MODEL_STATES.get(self.prefix, {}).get(key, [])
        for i, badge in enumerate(badges):
            bx = x + w - 12 - (len(badges) - i) * 33
            self.nodes.append(f'<a href="#{self.prefix}-module-{badge.lower()}" aria-label="查看 {badge} 模块"><rect x="{bx}" y="{y+11}" width="28" height="22" rx="5" fill="#fff" stroke="{stroke}" stroke-opacity=".5"/><text x="{bx+14}" y="{y+27}" text-anchor="middle" font-size="12" font-weight="650" fill="{stroke}">{badge}</text></a>')
        self.nodes.append("</g>")

    def label(self, x, y, lines, color="#3f5b71", anchor="middle", size=15):
        # A white halo keeps conditions readable where several routes meet.
        self.labels.append('<g class="edge-label" paint-order="stroke" stroke="#fff" stroke-width="6" stroke-linejoin="round">')
        self.text(x, y, lines, size=size, anchor=anchor, color=color, weight=500)
        self.labels.append("</g>")

    def edge(self, d, label=None, xy=None, kind="normal", dashed=False):
        color = {"normal": "#607b91", "miss": "#bf6556", "commit": "#8663b0", "verify": "#228978", "draft": "#3974bf"}[kind]
        dash = ' stroke-dasharray="7 5"' if dashed else ""
        self.edges.append(f'<path class="transition {kind}" d="{d}" fill="none" stroke="{color}" stroke-width="2.1"{dash} marker-end="url(#{self.prefix}-arrow-{kind})"/>')
        if label:
            self.label(*xy, label, color=color)

    def initial(self, x, y):
        self.nodes.append(f'<circle cx="{x}" cy="{y}" r="9" fill="#213f57"/>')

    def final(self, x, y):
        self.nodes.append(f'<g class="final"><circle cx="{x}" cy="{y}" r="15" fill="#fff" stroke="#213f57" stroke-width="2"/><circle cx="{x}" cy="{y}" r="9" fill="#213f57"/></g>')

    def choice(self, x, y, label=None):
        self.nodes.append(f'<path d="M{x} {y-19} L{x+19} {y} L{x} {y+19} L{x-19} {y}Z" fill="#fff9ed" stroke="#b47a24" stroke-width="1.8"/>')
        if label:
            self.label(x, y - 33, [label])

    def bar(self, x, y1, y2, title=None):
        self.nodes.append(f'<rect x="{x-3}" y="{y1}" width="6" height="{y2-y1}" rx="3" fill="#284a63"/>')
        if title:
            self.label(x, y1 - 13, [title], size=12)

    def note(self, x, y, lines):
        self.text(x, y, lines, size=15, color="#698095")

    def modules(self):
        parts = []
        self.text(50, 973, ["模块速览 · M 标记对应状态中执行的模型"], size=17, weight=700, layer=parts)
        self.text(1550, 973, ["B 批量 · P prefix · q/Q query · γ 预测位置 · h hidden · m 特征层数 · V 词表"], size=14, anchor="end", color="#698095", layer=parts)
        modules = MODULES[self.prefix]
        gap = 16
        width = (1500 - gap * (len(modules) - 1)) / len(modules)
        for i, (key, name, status, structure, inputs, outputs) in enumerate(modules):
            x, y = 50 + i * (width + gap), 990
            parts.append(f'<g class="module-card" id="{self.prefix}-module-{key.lower()}"><rect x="{x}" y="{y}" width="{width}" height="140" rx="12" fill="#f8fafc" stroke="#cbd8e2"/><rect x="{x+15}" y="{y+14}" width="33" height="24" rx="5" fill="#e5eef8"/>')
            self.text(x+31.5, y+32, [key], size=14, weight=700, anchor="middle", color="#3974bf", layer=parts)
            self.text(x+59, y+32, [name], size=20, weight=700, color="#18364e", layer=parts)
            self.text(x+width-15, y+31, [status], size=14, anchor="end", color="#698095", layer=parts)
            self.text(x+16, y+62, [structure], size=17, weight=500, layer=parts)
            self.text(x+16, y+88, [inputs, outputs], size=17, color="#4b6479", layer=parts)
            parts.append("</g>")
        self.text(50, 1153, ["形状以共享 h 的 Qwen3-8B 式模型为例：h=4096；L_d、m 依 checkpoint；γ 为选取后的预测位置数。共享 LM head = Linear(h,V)。"], size=14, color="#698095", layer=parts)
        return "".join(parts)

    def render(self):
        defs = []
        for kind, color in [("normal", "#607b91"), ("miss", "#bf6556"), ("commit", "#8663b0"), ("verify", "#228978"), ("draft", "#3974bf")]:
            defs.append(f'<marker id="{self.prefix}-arrow-{kind}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M1 1L9 5L1 9Z" fill="{color}"/></marker>')
        legend = []
        for i, (kind, name) in enumerate([("draft", "draft / 预测计算"), ("control", "控制 / 调度"), ("verify", "target verify"), ("commit", "状态提交"), ("miss", "fallback")]):
            fill, stroke = COLORS[kind]
            x = 50 + i * 205
            legend.append(f'<rect x="{x}" y="919" width="12" height="12" rx="3" fill="{fill}" stroke="{stroke}"/><text x="{x+21}" y="930" font-size="14" fill="#698095">{name}</text>')
        return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1600 1170" width="1600" height="1170" role="img" aria-label="{escape(self.name)} 状态机">
<title>{escape(self.name)} · 横版状态机</title><desc>{escape(self.description)}</desc>
<defs>{''.join(defs)}</defs><style>text{{font-family:"Noto Sans CJK SC","Microsoft YaHei",system-ui,sans-serif}}.state:hover rect{{filter:brightness(.98)}}.edge-label{{pointer-events:none}}</style>
<rect width="1600" height="1170" fill="#fff"/>{''.join(self.back)}{''.join(self.edges)}{''.join(self.nodes)}{''.join(self.labels)}
<path d="M50 896H1550" stroke="#e0e8ee"/>{''.join(legend)}<text x="1545" y="930" text-anchor="end" font-size="14" fill="#698095">箭头：事件 / [条件] → 动作　　● 初态　◎ 终态</text>
{self.modules()}
</svg>'''


def dflash():
    c = Chart("DFlash", "单卡基本状态机。一次 draft forward 后验证，接受部分或全部候选，再提交状态并进入下一轮。")
    c.initial(45, 200)
    c.state("prefill", 90, 145, 240, 116, "Prefill 执行中", ["do：target 处理 prompt", "产出 KV / H、anchor"], "verify")
    c.state("ready", 440, 145, 280, 116, "Context ready", ["真实 prefix KV / H 已确认", "anchor 已知，等待生成草稿"])
    c.state("draft", 845, 145, 280, 116, "Draft 执行中", ["do：并行 forward × 1", "anchor + MASK → logits"], "draft")
    c.state("proposal", 1250, 145, 280, 116, "候选就绪", ["entry：argmax / sampling", "持有完整线性候选 block"], "control")
    c.state("verify", 1250, 440, 280, 116, "Verify 执行中", ["do：target causal extend", "输入 anchor + 当前候选"], "verify")
    c.choice(1080, 498, "target 返回")
    c.state("commit", 665, 440, 285, 139, "Commit 执行中", ["保留接受前缀的 KV / H", "输出 correction / bonus", "丢弃其余候选与临时状态"], "commit")
    c.choice(500, 510, "是否结束")
    c.final(500, 755)
    c.edge("M54 200H90")
    c.edge("M330 200H440", ["prefill 完成"], (386, 181))
    c.edge("M720 200H845", ["开始本轮"], (782, 181))
    c.edge("M1125 200H1250", ["forward 完成"], (1188, 181))
    c.edge("M1390 261V440", ["候选就绪", "提交 target"], (1460, 346), "verify")
    c.edge("M1250 498H1099", ["logits 就绪"], (1170, 479), "verify")
    c.edge("M1080 479V375H810V440", ["[首个拒绝位置 k+1]", "接受前 k 个；产生 correction"], (933, 340), "commit")
    c.edge("M1080 517V650H810V579", ["[γ 个候选全部接受]", "接受整块；产生 bonus"], (946, 685), "commit")
    c.edge("M665 510H519", ["状态提交完成"], (594, 490), "commit")
    c.edge("M500 491V261", ["[继续]", "新 anchor / 新 H"], (420, 370), "commit")
    c.edge("M500 529V740", ["[EOS / token budget]"], (398, 668))
    c.note(71, 780, ["GPU 常驻：target KV、draft prefix KV、真实 target 特征 H。", "每个 block 一次 draft forward；这里没有反复 denoise / Jacobi 的状态回环。"])
    c.note(1060, 800, ["单轮关键依赖：draft 完成 → target verify。", "verify 结果更新 context 后，下一轮才重新 draft。"])
    return c


def dspark():
    c = Chart("DSpark", "包含并行 draft 与历史预算的并发区域；Markov 采样有位置内循环，验证后回到新 context。")
    c.region(370, 85, 940, 475, "ROUND PREPARATION · 两个并发区域", "GPU 生成当前候选；CPU 用历史预测准备总预算。黑竖条表示 fork / join。")
    c.back.append('<path d="M390 330H1290" stroke="#cbd7e1" stroke-dasharray="6 5"/>')
    c.text(394, 160, ["GPU / 当前 block"], size=12, color="#688096", layer=c.back)
    c.text(470, 370, ["CPU / 历史预算"], size=12, color="#688096", layer=c.back)
    c.initial(35, 345)
    c.state("ready", 80, 285, 240, 139, "Context ready", ["prefill 或上轮提交已完成", "真实 H / KV 与 anchor 已知", "历史 confidence 可用"])
    c.bar(410, 200, 445, "FORK")
    c.state("backbone", 465, 188, 255, 116, "Backbone", ["do：并行 forward × 1", "产出 hᵢ、基础 logits Uᵢ"], "draft")
    c.state("markov", 850, 188, 320, 116, "Markov 采样", ["do：Uᵢ + bias(prev token) → dᵢ", "更新 prev token、位置 i"], "control")
    c.state("budget", 495, 398, 555, 116, "Budget planning", ["历史 confidence + 本机 SPS 表 → 搜索 K", "保存总预算；不读取当前未来候选来回看选 K"], "control")
    c.bar(1250, 245, 455, "JOIN")
    c.state("plan", 1355, 290, 205, 139, "Verify 计划", ["hᵢ / prev → cᵢ", "cumprod / 分配 K", "各请求长度 ℓᵣ 确定"], "control")
    c.state("verify", 1120, 660, 300, 139, "Verify 执行中", ["do：compact / ragged target", "每请求只处理所选前缀", "输入量 = 请求数 + Σℓᵣ"], "verify")
    c.state("commit", 675, 660, 300, 139, "Commit 执行中", ["接受连续前缀 + correction", "提交真实 KV / H", "未验证 / 被拒绝后缀都丢弃"], "commit")
    c.choice(520, 730, "是否结束")
    c.final(520, 845)
    c.edge("M44 345H80", ["prefill 完成"], (167, 258))
    c.edge("M320 345H410", ["开始新轮"], (360, 324))
    c.edge("M410 245H465")
    c.edge("M410 445H495")
    c.edge("M720 245H850", ["hᵢ / Uᵢ 就绪"], (785, 225))
    c.edge("M940 188V166H1110V188", ["[i < γ] / 下一位置"], (1022, 154))
    c.edge("M1170 245H1250", ["[i = γ]"], (1210, 225))
    c.edge("M1050 455H1250", ["K ready"], (1150, 435))
    c.edge("M1250 350H1355", ["[两侧均就绪]"], (1300, 278))
    c.edge("M1457 429V610H1270V660", ["prefix 选择 / 压紧完成", "候选与长度交给 target"], (1370, 577), "verify")
    c.edge("M1120 730H975", ["target 返回 k、a"], (1047, 710), "commit")
    c.edge("M675 730H539", ["提交完成"], (608, 710), "commit")
    c.edge("M501 730H200V424", ["[继续] / 新 anchor，重新 draft"], (377, 647), "commit")
    c.edge("M520 749V830", ["[结束]"], (570, 805))
    c.note(65, 866, ["Static verify 分支：固定验证长度，可跳过 confidence / 预算调度。"])
    c.note(1035, 862, ["优化边界：串行采样、动态前缀压紧。", "缩短 ℓ 后仍需同步缩小 graph padding。"])
    return c


def jetspec():
    c = Chart("JetSpec", "包含 CPU 构树循环、target tree verify，以及 gather 和 logical-map 两种 KV 提交分支。")
    c.region(1070, 95, 450, 235, "CPU CONTROL · 本地默认 builder", "top-k 有 D2H；树 / ancestor 元数据回到 GPU。")
    c.initial(35, 220)
    c.state("ready", 80, 165, 245, 116, "Context ready", ["真实 prefix KV / H", "当前 anchor 已确认"])
    c.state("draft", 420, 165, 280, 116, "Causal draft", ["do：并行 forward × 1", "得到各深度 logits"], "draft")
    c.state("topk", 795, 165, 205, 116, "Top-k ready", ["每深度候选与分数", "等待 tree builder"], "control")
    c.state("tree", 1100, 180, 385, 116, "Tree building", ["heap 扩展候选；构造 ancestor", "节点数 / 深度 / 宽度受预算限制"], "control")
    c.state("verify", 1190, 465, 300, 139, "Tree verify 执行中", ["do：N 节点 target forward", "每节点只看 prefix + 自身祖先", "KV 暂存在各节点槽位"], "verify")
    c.state("accept", 795, 465, 280, 139, "Path selection", ["从 root 找到接受路径", "产生 correction / bonus", "路径索引用于 KV / H 提交"], "control")
    c.choice(700, 535, "KV backend")
    c.state("gather", 405, 435, 215, 116, "KV gather", ["按路径整理有效 KV", "回收被拒绝分支"], "commit")
    c.state("logical", 405, 680, 260, 116, "逻辑映射提交", ["KV 保留原物理槽位", "仅更新有效逻辑映射"], "commit")
    c.state("commit", 80, 435, 245, 139, "Context committed", ["追加接受路径的真实 H", "输出 tokens 与新 anchor", "提交完成后进入下一轮"], "commit")
    c.final(200, 775)
    c.edge("M44 220H80", ["prefill 完成"], (169, 140))
    c.edge("M325 220H420", ["开始本轮"], (372, 200))
    c.edge("M700 220H795", ["forward 完成"], (748, 200))
    c.edge("M1000 220H1100", ["D2H"], (1050, 199))
    c.edge("M1195 180V61H1440V180", ["[可继续扩展] / heap 下一节点"], (1320, 42))
    c.edge("M1292 296V465", ["[预算到达 / 候选耗尽]", "H2D：tokens / parents / ancestor"], (1350, 376), "verify")
    c.edge("M1190 535H1075", ["target logits 就绪"], (1135, 444), "verify")
    c.edge("M795 535H719", ["path ready"], (756, 570))
    c.edge("M681 535H650V493H620", ["[gather]"], (653, 468), "commit")
    c.edge("M700 554V738H665", ["[no-gather]"], (763, 657), "commit")
    c.edge("M405 493H325", ["提交有效 KV"], (365, 472), "commit")
    c.edge("M405 738H350V555H325", ["提交有效映射"], (260, 707), "commit")
    c.edge("M200 435V281", ["[继续] / 新 anchor、H"], (318, 366), "commit")
    c.edge("M200 574V760", ["[EOS / budget]"], (116, 670))
    c.note(800, 775, ["树是当前轮多个可能前缀的表示；target 最终只提交其中一条路径。", "gather 是设备内缓存整理，不是 TP collective。", "可选 conditioned extension 会增加 draft forward，不在本基本状态机内。"])
    c.note(80, 864, ["优化边界：CPU 构树与同步、ancestor / mask、N 节点 target attention、路径 KV 提交。"])
    return c


def jetspecpp():
    c = Chart("JetSpec++", "带两个 GPU 并发区域和同步汇合；候选 refinement 自循环；命中和 miss 分支；首轮 admission 与终止路径。")
    c.region(365, 90, 1165, 450, "PIPELINED ROUND(t) · 两个 GPU 并发区域", "同一 READY 事件启动两侧；候选没准备好或真实结果没返回，汇合处都不能继续。")
    c.back.append('<path d="M385 310H1510" stroke="#cbd7e1" stroke-dasharray="6 5"/>')
    c.text(390, 177, ["TARGET GPU"], size=11, color="#688096", layer=c.back)
    c.text(480, 353, ["DRAFT GPU"], size=11, color="#688096", layer=c.back)
    c.initial(35, 350)
    c.state("ready", 80, 285, 235, 139, "Block(t) ready", ["当前草稿已确定", "首轮：admission draft", "稳态来自 hit 或 fallback"])
    c.bar(425, 225, 450, "FORK")
    c.state("verify", 480, 195, 290, 116, "Verify(t) 执行中", ["do：完整 target forward", "得到真实 k*、a*、Hₜ"], "verify")
    c.state("outcome", 920, 195, 290, 116, "Outcome ready", ["提交接受 tokens / KV / H", "发布 outcome + H 增量"], "commit")
    c.state("foresight", 480, 385, 265, 139, "Foresight", ["adapter：预测 Ĥₜ", "decoder：选 (k,a)", "构造多种假想 context"], "draft")
    c.state("refine", 840, 385, 300, 139, "R / J 迭代", ["J0：批量生成 t+1 初稿", "按 schedule 执行 R / J", "R 预测 Ĥₜ₊₁；J 更新 tokens"], "draft")
    c.state("cache", 1230, 385, 200, 116, "Cache ready", ["Cache[(k,a)]", "保存最终 t+1 草稿"], "control")
    c.bar(1475, 250, 445, "JOIN")
    c.state("lookup", 1110, 625, 310, 116, "Resolve outcome", ["使用真实 (k*,a*) 查询缓存", "候选已发布；真实结果已就绪"], "control")
    c.choice(985, 683, "是否命中")
    c.state("next", 600, 625, 280, 139, "Block(t+1) ready", ["持有选中的或补生成的草稿", "真实 H 用于后续状态更新", "推进轮次 t ← t+1"], "commit")
    c.state("fallback", 1100, 795, 320, 93, "Fallback drafting", ["真实 context + a* → 补一次 J0"], "miss")
    c.final(1550, 52)
    c.edge("M44 350H80", ["首轮启动完成"], (166, 263))
    c.edge("M315 350H425", ["启动本轮"], (365, 330))
    c.edge("M425 250H480")
    c.edge("M425 450H480")
    c.edge("M770 250H920", ["verify 返回"], (846, 230), "verify")
    c.edge("M1210 250H1475", ["[非 terminal] / 发布真实结果"], (1343, 231), "commit")
    c.edge("M1065 195V48H1534", ["[EOS / token budget] / pipeline 收尾"], (1316, 30))
    c.edge("M745 450H840", ["context 就绪"], (791, 430), "draft")
    c.edge("M900 385V346H1100V385", ["[还有 R / J 阶段] / 继续迭代"], (1000, 335), "draft")
    c.edge("M1140 450H1230", ["[阶段完成]"], (1186, 430), "draft")
    c.edge("M1430 445H1475", ["候选发布"], (1450, 485))
    c.edge("M1475 350H1560V683H1420", ["[两侧就绪]", "否则等候"], (1502, 587))
    c.edge("M1110 683H1004")
    c.edge("M966 683H880", ["[hit] / 取缓存行"], (922, 661), "commit")
    c.edge("M985 702V841H1100", ["[miss] / 无匹配候选"], (978, 777), "miss")
    c.edge("M1100 858H740V764", ["补 draft 完成"], (907, 880), "miss")
    c.edge("M600 695H200V424", ["下一轮：verify(t+1) 与准备 t+2 同时启动"], (378, 646), "commit")
    c.note(65, 805, ["J = draft forward；R = adapter + decoder + 修正。", "默认预备 4 种结果；J1-R1 = J0 → R1 → J1。"])
    c.note(65, 858, ["跨卡 / 进程：Target→Draft 传 outcome + 真实 H；反向传候选 token IDs 与键。", "双卡各用自己的 SM；单卡 MPS 的资源竞争不由本图假定消失。"])
    return c


INFO = [
    ("dflash", "DFlash", "一次并行 draft；verify 后提交真实前缀，再开始下一轮。", dflash,
     "model.py · dflash_generate", "../third_party/DFlash/dflash/model.py"),
    ("dspark", "DSpark", "Markov 位置内循环 + 历史预算并发准备；当前 confidence 分配 verify 名额。", dspark,
     "dspark_planner.py / dspark_draft.py", "../third_party/SGLang/python/sglang/srt/speculative/dspark_components/dspark_planner.py"),
    ("jetspec", "JetSpec", "树构建与路径选择；gather / logical-map 两条缓存提交路径。", jetspec,
     "engine.py · generate_tree", "../third_party/JetSpec/jetspec/engines/jetspec/engine.py"),
    ("jetspecpp", "JetSpec++", "两个 GPU 的并发状态、汇合等待、候选迭代与 hit / miss 恢复。", jetspecpp,
     "round.py / serving engine", "../third_party/JetSpec/jetspec/methods/jetspec_plus_plus/round.py"),
]


def main():
    ASSETS.mkdir(exist_ok=True)
    sections = []
    for i, (slug, title, subtitle, fn, source_name, source_url) in enumerate(INFO, 1):
        svg = fn().render()
        (ASSETS / f"{slug}_state_machine.svg").write_text(svg)
        sections.append(f'''<section id="{slug}" class="sheet"><header><div><div class="kicker">{i:02d} / SYSTEM STATE MACHINE</div><h1>{title}</h1><p>{subtitle}</p></div><span class="number">{i:02d}</span></header>
<div class="drawing">{svg}</div><footer><span>2026-11-06 · 逻辑状态机：边标注事件 / 条件与动作，节点合并了底层 kernel。H = 真实 target 特征；Ĥ = adapter 预测。</span><span><a href="{source_url}">{source_name}</a> · <a download href="2026-11-06_sd_systems_flow/{slug}_state_machine.svg">下载 SVG ↗</a></span></footer></section>''')
    nav = "".join(f'<a href="#{s}">{t}</a>' for s, t, *_ in INFO)
    html = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DFlash · DSpark · JetSpec · JetSpec++｜横版 SVG 状态机</title><style>
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:74px}body{margin:0;background:#edf2f6;color:#1b3b54;font:15px/1.5 "Noto Sans CJK SC","Microsoft YaHei",system-ui,sans-serif}.toolbar{position:sticky;top:0;z-index:3;display:flex;align-items:center;gap:20px;padding:12px 25px;background:#193b54;color:white}.brand{font-size:14px;font-weight:750;letter-spacing:.12em}nav{display:flex;gap:8px;flex:1}.toolbar a,.toolbar button{color:#fff;text-decoration:none;background:transparent;border:1px solid #ffffff30;border-radius:5px;padding:6px 13px;font:inherit;font-size:14px}.toolbar a:hover,.toolbar button:hover{background:#ffffff15}button{cursor:pointer}a{color:#2d658f;text-underline-offset:3px}a:focus-visible,button:focus-visible{outline:3px solid #d1aa55;outline-offset:3px}.sheet{max-width:1660px;min-width:1050px;margin:23px auto;padding:26px 28px 18px;background:white;border:1px solid #d6e1e9;border-radius:12px;box-shadow:0 6px 20px #16334908;scroll-margin-top:76px}.sheet header{display:flex;justify-content:space-between;gap:24px;padding:0 20px}.kicker{font-size:12px;letter-spacing:.14em;color:#6b8193;font-weight:700}h1{font-size:38px;line-height:1.2;letter-spacing:-.04em;margin:5px 0 7px}header p{font-size:16px;color:#667f93;margin:0}.number{color:#ccd8e1;font:36px/1.2 ui-monospace,monospace}.drawing{margin-top:8px}.drawing svg{width:100%;height:auto;display:block}.sheet footer{padding:13px 20px 0;border-top:1px solid #e1e8ee;display:flex;justify-content:space-between;gap:20px;font-size:12px;color:#6b8193}.sheet footer span:last-child{white-space:nowrap}.boundary{max-width:1660px;min-width:1050px;margin:0 auto 30px;padding:16px 22px;font-size:13px;color:#667f93;border:1px solid #d8e2e9;border-radius:8px;background:white}.boundary summary{cursor:pointer;color:#1b3b54;font-weight:650}.boundary p{margin:9px 0 0}.mobile{display:none}@media(max-width:1100px){.sheet,.boundary{width:1050px;margin-left:14px;margin-right:14px}.brand{display:none}.toolbar{gap:6px;padding:9px 12px}.toolbar a,.toolbar button{font-size:10px;padding:5px 8px}.mobile{display:block;margin:10px 15px;font-size:11px;color:#667f93}}@media print{@page{size:A3 landscape;margin:8mm}body{background:#fff;-webkit-print-color-adjust:exact;print-color-adjust:exact}.toolbar,.boundary,.mobile{display:none!important}.sheet{width:1600px;max-width:none;min-width:0;height:1310px;margin:0;padding:20px;border:0;border-radius:0;box-shadow:none;zoom:.77;break-after:page;page-break-after:always}.sheet:last-of-type{break-after:auto;page-break-after:auto}.sheet header{padding:0 20px}.drawing{margin-top:0}.sheet footer{font-size:10px}.sheet footer a{color:inherit;text-decoration:none}}
</style></head><body><div class="toolbar"><span class="brand">PARALLEL SD / STATECHARTS</span><nav aria-label="状态机导航">'''+nav+'''</nav><button type="button" onclick="window.print()">横向打印 / PDF</button></div><p class="mobile">每个系统独立一张横版 SVG；窄屏可横向滚动。</p>'''+"".join(sections)+'''
<details class="boundary"><summary>图示范围与源码版本</summary><p>只保留系统执行的逻辑状态、转换条件、分支、并发和回环；不按节点大小表达耗时。DFlash 指原始单步版本；DSpark 为 Markov + 动态 prefix scheduler；JetSpec 为本地默认树路径，no-gather 是可选 backend；JetSpec++ 为默认双卡 J1-R1 示例，首轮需 admission，候选迟到在汇合处等待，超时回退策略依 recipe。EOS 收尾可能还需排空在途通信。</p><p>JetSpec：673267e4faba71a63099c258f0802c2f3aa44f8e；DFlash：07ebd93db9f472af339b644bb70221ad8428328a；SGLang：27aedf7f93cc7ac3bff43e7150e42a3c0baa6957。该图不证明本机启用了全部路径，也不替代既有 BF16 AR / tree 数值差异的独立核查。</p><p>作者论文：<a href="https://arxiv.org/html/2602.06036v1">DFlash</a> · <a href="https://arxiv.org/html/2607.05147v1">DSpark</a> · <a href="https://arxiv.org/html/2606.18394v1">Parallel tree drafting</a> · <a href="https://arxiv.org/html/2610.19792v1">JetSpec++</a>。所有图形均内嵌于 HTML，可离线打开；独立 SVG 可另存编辑。</p></details></body></html>'''
    REPORT.write_text(html)
    from build_report_index import write_index
    write_index()
    print(REPORT)


if __name__ == "__main__":
    main()
