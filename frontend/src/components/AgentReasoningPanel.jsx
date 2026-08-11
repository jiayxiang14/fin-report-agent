import {
  Activity,
  Brain,
  CheckCircle2,
  ChevronRight,
  Compass,
  FileText,
  Loader2,
  ShieldCheck,
  Sparkles,
  TrendingUp,
  Users,
  XCircle,
} from 'lucide-react'
import { useState } from 'react'
import { formatElapsed } from '../lib/formatElapsed'
import Panel from './Panel'

const REASONING_PREVIEW_LENGTH = 160

const TOOL_LABELS = {
  get_financials: '核心财务数据',
  get_sector_position: '板块轮动位置',
  get_peer_comparison: '同行对比',
  get_filing_text: '财报原文',
  get_price_reaction: '财报发布后的股价反应',
  verify_number: '数字核实',
}

const TOOL_ICONS = {
  get_financials: TrendingUp,
  get_sector_position: Compass,
  get_peer_comparison: Users,
  get_filing_text: FileText,
  get_price_reaction: Activity,
  verify_number: ShieldCheck,
}

function toolLabel(name) {
  return TOOL_LABELS[name] || name
}

function toolIcon(name) {
  return TOOL_ICONS[name] || Sparkles
}

// 按 turn 分组，而不是把所有事件摊平成一条时间线——Agent Loop真正的运作方式是
// "一轮决策（可能同时发起好几个工具调用）→ 下一轮决策"，摊平展示看不出"这几个
// 工具调用是同一轮里一起决定的"，也看不出轮数本身是可变的（不同公司跑出来的轮数
// 不一样，这正是"这是真实的自主决策而不是写死的固定流程"的直接证据）。
function groupByTurn(log) {
  const groups = []
  for (const entry of log) {
    const last = groups[groups.length - 1]
    if (last && last.turn === entry.turn) {
      last.entries.push(entry)
    } else {
      groups.push({ turn: entry.turn, entries: [entry] })
    }
  }
  return groups
}

// 用工具图标拼一条"这次跑了哪些步骤"的流程图，跟下面逐条展开的推理日志是互补关系——
// 这一条给"一眼看全貌"，下面的时间线给"想看细节再展开读"。直接从 tool_finished 事件
// 里取，跟着SSE事件实时增长，不是等分析完了才一次性画出来。
function WorkflowStrip({ log }) {
  const steps = log.filter((entry) => entry.kind === 'tool_finished')
  if (steps.length === 0) return null

  return (
    <div className="mb-4 flex flex-wrap items-center gap-1.5 rounded-lg bg-violet-50/60 p-2.5">
      {steps.map((step, idx) => {
        const Icon = toolIcon(step.toolName)
        return (
          <div key={step.key} className="flex items-center gap-1.5">
            <div
              title={toolLabel(step.toolName)}
              className={`flex items-center gap-1 rounded-full px-2 py-1 text-xs font-medium ${
                step.isError
                  ? 'bg-red-100 text-red-600'
                  : 'bg-white text-violet-700 shadow-sm'
              }`}
            >
              <Icon className="h-3.5 w-3.5" />
              <span className="hidden sm:inline">{toolLabel(step.toolName)}</span>
            </div>
            {idx < steps.length - 1 && (
              <ChevronRight className="h-3.5 w-3.5 shrink-0 text-violet-300" />
            )}
          </div>
        )
      })}
    </div>
  )
}

// 模型最后一轮的文字往往就是完整的三段式简报本身（带<conclusion>等标签），
// 那份内容已经在右边的投研简报里格式化展示过一遍了，这里没必要再原样甩一整段
// 带标签的原始文本——截一小段预览，想看全文自己展开
function ReasoningText({ text }) {
  const [expanded, setExpanded] = useState(false)
  const isLong = text.length > REASONING_PREVIEW_LENGTH

  if (!isLong || expanded) {
    return (
      <span className="text-gray-700">
        {text}
        {isLong && (
          <button
            type="button"
            onClick={() => setExpanded(false)}
            className="ml-1.5 text-xs font-medium text-violet-500 hover:underline"
          >
            收起
          </button>
        )}
      </span>
    )
  }

  return (
    <span className="text-gray-700">
      {text.slice(0, REASONING_PREVIEW_LENGTH).replace(/<[^>]+>/g, '')}…
      <button
        type="button"
        onClick={() => setExpanded(true)}
        className="ml-1.5 text-xs font-medium text-violet-500 hover:underline"
      >
        展开全文
      </button>
    </span>
  )
}

function LogEntry({ entry }) {
  if (entry.kind === 'reasoning') {
    return (
      <div className="flex items-start gap-2">
        <Sparkles className="mt-0.5 h-4 w-4 shrink-0 text-violet-400" />
        <ReasoningText text={entry.text} />
      </div>
    )
  }
  if (entry.kind === 'tool_started') {
    return (
      <div className="flex items-start gap-2">
        <Loader2 className="mt-0.5 h-4 w-4 shrink-0 animate-spin text-sky-500" />
        <span className="text-gray-500">正在查询「{toolLabel(entry.toolName)}」…</span>
      </div>
    )
  }
  return (
    <div className="flex items-start gap-2">
      {entry.isError ? (
        <XCircle className="mt-0.5 h-4 w-4 shrink-0 text-red-500" />
      ) : (
        <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-green-500" />
      )}
      <span className={entry.isError ? 'text-red-500' : 'text-gray-500'}>
        「{toolLabel(entry.toolName)}」{entry.isError ? `失败：${entry.summary}` : '已完成'}
      </span>
    </div>
  )
}

export default function AgentReasoningPanel({ log, result, streamError, done, elapsedMs }) {
  const groups = groupByTurn(log)

  return (
    <Panel icon={Brain} iconClassName="bg-violet-100 text-violet-600" title="Agent 推理过程">
      <WorkflowStrip log={log} />

      <div>
        {groups.map((group, idx) => (
          <div key={group.turn} className="flex gap-3">
            <div className="flex flex-col items-center">
              <div className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-violet-600 text-xs font-semibold text-white">
                {group.turn + 1}
              </div>
              {(idx < groups.length - 1 || !done) && (
                <div className="mt-1 w-px flex-1 bg-violet-100" />
              )}
            </div>
            {/* 右边加一个跟左边"序号圆点+连接线"那一列（w-6 + gap-3 = 36px）
                同宽的 pr-9，让文字左右两边到卡片边缘的视觉间距对称，不是左边
                因为有图标看起来挤、右边因为没有东西看起来空 */}
            <div className="flex-1 space-y-1.5 pb-4 pr-9 text-sm">
              {group.entries.map((entry) => (
                <LogEntry key={entry.key} entry={entry} />
              ))}
            </div>
          </div>
        ))}

        {!done && (
          <div className="flex items-center gap-2 pl-9 text-sm text-gray-300">
            <Loader2 className="h-4 w-4 animate-spin" />
            分析中…{elapsedMs != null && `（已用时 ${formatElapsed(elapsedMs)}）`}
          </div>
        )}
      </div>

      {done && result && (
        <p className="mt-1 text-xs text-gray-400">
          {elapsedMs != null && `本次分析用时 ${formatElapsed(elapsedMs)}，`}
          Agent自主决定跑了 {result.turns_used} 轮工具调用/推理，具体轮数和每轮的
          工具调用由模型根据这家公司的实际情况动态决定，不是固定流程
        </p>
      )}

      {streamError && (
        <p className="mt-2 flex items-center gap-1.5 text-sm text-red-600">
          <XCircle className="h-4 w-4" />
          分析失败：{streamError}
        </p>
      )}
    </Panel>
  )
}
