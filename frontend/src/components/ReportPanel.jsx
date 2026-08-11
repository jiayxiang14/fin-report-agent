import { AlertTriangle, CheckCircle2, ClipboardList, Database, Loader2, Target } from 'lucide-react'
import { Children } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { parseReport } from '../lib/parseReport'
import Panel from './Panel'

// 结论区块的颜色不是固定的——跟着Agent自己在<sentiment>标签里对<conclusion>的归纳走
// （generation.md：positive/negative/neutral三选一），前端只负责映射颜色，不重新判断。
// 模型没写这个标签（旧行为、或者解析失败）时退化成中性紫，不是报错。
const SENTIMENT_THEMES = {
  positive: {
    label: '结论',
    icon: Target,
    wrapperClass: 'border-l-4 border-emerald-500 bg-emerald-50',
    iconClass: 'bg-emerald-500 text-white',
    headingClass: 'text-emerald-700',
    markerClass: 'marker:text-emerald-500',
    numberClass: 'bg-emerald-100 text-emerald-700',
  },
  negative: {
    label: '结论',
    icon: Target,
    wrapperClass: 'border-l-4 border-rose-500 bg-rose-50',
    iconClass: 'bg-rose-500 text-white',
    headingClass: 'text-rose-700',
    markerClass: 'marker:text-rose-500',
    numberClass: 'bg-rose-100 text-rose-700',
  },
  neutral: {
    label: '结论',
    icon: Target,
    wrapperClass: 'border-l-4 border-violet-500 bg-violet-50',
    iconClass: 'bg-violet-500 text-white',
    headingClass: 'text-violet-700',
    markerClass: 'marker:text-violet-500',
    numberClass: 'bg-violet-100 text-violet-700',
  },
}

function resolveSentiment(raw) {
  // 用included而不是精确相等：模型偶尔会在标签内容里带标点或多余空白
  // （比如"positive。"），精确相等会把这种情况错判成没写sentiment，
  // 退化成中性紫——这个标签本来就是纯归纳词，只要包含关键词就该按其归类
  const normalized = (raw || '').trim().toLowerCase()
  if (normalized.includes('positive')) return 'positive'
  if (normalized.includes('negative')) return 'negative'
  return 'neutral'
}

const SECTIONS = {
  evidence: {
    label: '数据支撑',
    icon: Database,
    wrapperClass: 'border-l-4 border-blue-400 bg-blue-50/60',
    iconClass: 'bg-blue-500 text-white',
    headingClass: 'text-blue-700',
    markerClass: 'marker:text-blue-500',
    numberClass: 'bg-blue-100 text-blue-700',
  },
  flags: {
    label: '异常提示',
    icon: AlertTriangle,
    wrapperClass: 'border-l-4 border-amber-400 bg-amber-50/60',
    iconClass: 'bg-amber-500 text-white',
    headingClass: 'text-amber-700',
    markerClass: 'marker:text-amber-500',
    numberClass: 'bg-amber-100 text-amber-700',
  },
}

const DEFAULT_SECTION = {
  headingClass: 'text-gray-700',
  markerClass: 'marker:text-gray-400',
  numberClass: 'bg-gray-100 text-gray-700',
}

// 数字高亮：只认"明确带着金额/百分比/量级标记"的数字，而不是"看起来像数字就先
// 高亮，再一个个排除年份/日期/文件代号"。之前是后一种思路，每发现一类新的
// "数字但不是财务数字"（Q1/Q2 → 10-K/10-Q → 日期）就得补一条排除规则，属于
// 越堵越多的漏洞列表；这次发现"不足5天""5个交易日"这类纯计数依然会被裸高亮
// ——只排除"日期"这个子集是不够的，任何不带$/%/亿/万/BMK/千分位逗号/小数点的
// 裸整数都可能是计数、序号、日期数字。反过来想：这份简报里真正的财务数字
// 幾乎总是带$前缀、百分号、亿/万/B/M/K量级后缀、千分位逗号或小数点之一——直接
// 只匹配这几种明确形态，年份/日期/文件代号/交易日计数这些裸整数天然就不会
// 命中，不需要再为每一类新变体单独写排除函数。
//
// (?<![A-Za-z0-9]) 仍然保留：防止在字母/数字紧邻的位置从数字串中间重新起头
// 匹配（比如不会把"FY2025"这种拆开算），跟内容形态的判断是两回事。
const NUMBER_PATTERN =
  /(?<![A-Za-z0-9])(?:\$[\d,]+(?:\.\d+)?(?:亿|万|[BMK])?|[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?(?:%|亿|万|[BMK])?|[+-]?\d+(?:\.\d+)?(?:%|亿|万|[BMK])|[+-]?\d+\.\d+)/g

function highlightText(text, keyPrefix, numberClass) {
  const parts = []
  let lastIndex = 0
  let match
  let i = 0
  NUMBER_PATTERN.lastIndex = 0
  while ((match = NUMBER_PATTERN.exec(text)) !== null) {
    const token = match[0]
    if (match.index > lastIndex) parts.push(text.slice(lastIndex, match.index))
    parts.push(
      <span key={`${keyPrefix}-${i++}`} className={`rounded px-1 py-0.5 font-semibold ${numberClass}`}>
        {token}
      </span>
    )
    lastIndex = match.index + token.length
  }
  if (lastIndex < text.length) parts.push(text.slice(lastIndex))
  return parts.length > 0 ? parts : text
}

// 只处理直接是字符串的子节点，遇到已经渲染成元素的子节点（比如markdown里的
// <a>链接）原样跳过不递归——避免破坏已有的渲染结果，数字高亮主要覆盖的是
// 大部分数字出现的场景：纯文本段落里的数字
function highlightChildren(children, numberClass) {
  return Children.map(children, (child, idx) =>
    typeof child === 'string' ? highlightText(child, `num-${idx}`, numberClass) : child
  )
}

// Agent写的markdown文本（加粗/列表/标题）交给react-markdown渲染，不是原样吐出去——
// 用户实测发现"**研发投入**"这种加粗语法直接显示成了星号，说明简报里的markdown格式
// 之前完全没被解析过。子标题(###)和列表颜色跟着各自区块的主题色走（结论/数据支撑/
// 异常提示三个区块颜色不一样），不是所有区块共用一套灰色样式——这样同一个区块内部
// 如果有好几个子话题，用小标题分段之后能看得更清楚，不是一整段文字堆在一起。
function createMarkdownComponents({ headingClass, markerClass, numberClass }) {
  return {
    p: ({ children }) => (
      <p className="mt-2 leading-relaxed first:mt-0">{highlightChildren(children, numberClass)}</p>
    ),
    strong: ({ children }) => (
      <strong className="font-semibold text-gray-900">
        {highlightChildren(children, numberClass)}
      </strong>
    ),
    ul: ({ children }) => (
      <ul className={`mt-2 list-disc space-y-1.5 pl-5 ${markerClass}`}>{children}</ul>
    ),
    ol: ({ children }) => (
      <ol className={`mt-2 list-decimal space-y-1.5 pl-5 ${markerClass}`}>{children}</ol>
    ),
    li: ({ children }) => (
      <li className="leading-relaxed">{highlightChildren(children, numberClass)}</li>
    ),
    h1: ({ children }) => (
      <h4 className={`mt-4 border-b border-current/10 pb-1 text-sm font-semibold first:mt-0 ${headingClass}`}>
        {children}
      </h4>
    ),
    h2: ({ children }) => (
      <h4 className={`mt-4 border-b border-current/10 pb-1 text-sm font-semibold first:mt-0 ${headingClass}`}>
        {children}
      </h4>
    ),
    h3: ({ children }) => (
      <h4 className={`mt-4 border-b border-current/10 pb-1 text-sm font-semibold first:mt-0 ${headingClass}`}>
        {children}
      </h4>
    ),
    // Agent有时会用markdown表格呈现多列对比（比如各业务分部的营收/同比一起列出来），
    // 标准markdown不带表格语法，得靠remark-gfm这个插件解析，不然就是原样显示一堆|和-。
    // 表格本身可能比max-w-prose窄的正文宽，套一层overflow-x-auto让它在自己的框里横向
    // 滚动，不撑破卡片——跟FinancialsPanel/PeerComparisonPanel的表格是同一个处理方式。
    table: ({ children }) => (
      <div className="mt-3 overflow-x-auto rounded border border-current/10">
        <table className="w-full min-w-[420px] border-collapse text-sm">{children}</table>
      </div>
    ),
    thead: ({ children }) => <thead className="bg-current/5">{children}</thead>,
    tr: ({ children }) => <tr className="border-b border-current/10 last:border-b-0">{children}</tr>,
    th: ({ children }) => (
      <th className={`px-3 py-1.5 text-left text-xs font-semibold ${headingClass}`}>{children}</th>
    ),
    td: ({ children }) => (
      <td className="px-3 py-1.5">{highlightChildren(children, numberClass)}</td>
    ),
  }
}

const fallbackMarkdownComponents = createMarkdownComponents(DEFAULT_SECTION)

export default function ReportPanel({ result }) {
  return (
    <Panel
      icon={ClipboardList}
      iconClassName="bg-gradient-to-br from-indigo-500 to-violet-500 text-white"
      title="投研简报"
    >
      {!result && (
        <div className="flex items-center gap-2 text-sm text-gray-400">
          <Loader2 className="h-4 w-4 animate-spin" />
          等待Agent完成分析…
        </div>
      )}

      {result && !result.completed && (
        <div className="rounded-lg border border-amber-300 bg-amber-50 p-4 text-sm">
          <p className="flex items-center gap-1.5 font-medium text-amber-800">
            <AlertTriangle className="h-4 w-4" />
            本次分析未能正常完成（stop_reason: {result.stop_reason}，共 {result.turns_used} 轮）
          </p>
          {result.final_report && (
            <p className="mt-2 max-w-prose whitespace-pre-wrap text-gray-700">
              {result.final_report}
            </p>
          )}
        </div>
      )}

      {result &&
        result.completed &&
        (() => {
          const { preamble, sections, hasAnyTag } = parseReport(result.final_report)

          if (!hasAnyTag) {
            return (
              <div className="rounded-lg border border-gray-200 bg-white p-4 text-sm">
                <p className="text-xs text-gray-400">未能按标准的三段式格式解析，展示原始内容：</p>
                <div className="mt-2 max-w-prose">
                  <ReactMarkdown remarkPlugins={[remarkGfm]} components={fallbackMarkdownComponents}>
                    {result.final_report}
                  </ReactMarkdown>
                </div>
              </div>
            )
          }

          return (
            <div className="space-y-3">
              {preamble && <p className="max-w-prose text-xs text-gray-400">{preamble}</p>}

              {['conclusion', 'evidence', 'flags'].map((tag) => {
                if (!sections[tag]) return null
                const section =
                  tag === 'conclusion' ? SENTIMENT_THEMES[resolveSentiment(sections.sentiment)] : SECTIONS[tag]
                const { label, icon: Icon, wrapperClass, iconClass, headingClass } = section
                return (
                  <div key={tag} className={`rounded-lg p-4 shadow-sm ${wrapperClass}`}>
                    <div className="flex items-center gap-1.5">
                      <div
                        className={`flex h-5 w-5 items-center justify-center rounded ${iconClass}`}
                      >
                        <Icon className="h-3 w-3" />
                      </div>
                      <h3 className={`text-xs font-semibold uppercase tracking-wide ${headingClass}`}>
                        {label}
                      </h3>
                    </div>
                    <div className="mt-1 max-w-prose text-sm text-gray-800">
                      <ReactMarkdown
                        remarkPlugins={[remarkGfm]}
                        components={createMarkdownComponents(section)}
                      >
                        {sections[tag]}
                      </ReactMarkdown>
                    </div>
                  </div>
                )
              })}

              {/* 透明化Agent自己的把关行为：自我核查/Reflexion整改都是代码里
                  真实发生的事，不是Prompt里写了就默认相信——用户应该能看到
                  这次分析实际经过了哪些自我把关步骤，而不是无从验证。
                  reflexion_triggered只在深度分析里可能为true（普通分析
                  没有这道检查），false时不展示，避免暗示"本该触发但没触发"。 */}
              <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-gray-400">
                <span className="flex items-center gap-1">
                  {result.self_verification_triggered ? (
                    <>
                      <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500" />
                      已完成自我核查
                    </>
                  ) : (
                    '本次未做自我核查'
                  )}
                </span>
                {result.reflexion_triggered && (
                  <span className="flex items-center gap-1">
                    <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500" />
                    过程裁判反馈后已整改
                  </span>
                )}
              </div>
            </div>
          )
        })()}
    </Panel>
  )
}
