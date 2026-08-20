import { Loader2, Search, Sparkles } from 'lucide-react'
import { lazy, Suspense, useEffect, useRef, useState } from 'react'
import AgentReasoningPanel from './components/AgentReasoningPanel'
import CompanyProfilePanel from './components/CompanyProfilePanel'
import FinancialsHistoryPanel from './components/FinancialsHistoryPanel'
import FinancialsPanel from './components/FinancialsPanel'
import PeerComparisonPanel from './components/PeerComparisonPanel'
import ReportPanel from './components/ReportPanel'
import SectorPanel from './components/SectorPanel'
import ThematicFlowPanel from './components/ThematicFlowPanel'
import { useAgentAnalysis } from './hooks/useAgentAnalysis'
import { useBestOfNAnalysis } from './hooks/useBestOfNAnalysis'

// 懒加载：这个面板只有点了"深度分析"才会用到，普通分析的用户永远用不上——
// 不像其它面板（FinancialsPanel/SectorPanel等）那样一提交ticker就必须同步
// 展示（CLAUDE.md的架构原则："同步展示结构化数据"），这个面板本来就是异步
// 出现在深度分析流程后段的，用React.lazy延迟到真正需要时才下载对应代码，
// 缩小首屏必须加载的bundle体积，不影响任何一个面板"该同步出现"的时机。
const CandidateComparisonPanel = lazy(() => import('./components/CandidateComparisonPanel'))

const BEST_OF_N_CANDIDATE_COUNT = 3

// 手绘的K线标记（不是lucide现成图标）：3根蜡烛，锐利直角，中间一根空心——
// 跟favicon.svg用的是同一个形状，只是这里省去了外圈背景色块（"Terminal"方向：
// 图标裸露不带色块底），品牌绿色号取自docs/color.png
function LogoMark({ className }) {
  return (
    <svg viewBox="0 0 32 32" fill="none" className={className} style={{ color: '#00AA6F' }}>
      <line x1="8" y1="4" x2="8" y2="10" stroke="currentColor" strokeWidth="1.6" />
      <rect x="5.5" y="10" width="5" height="10" fill="currentColor" />
      <line x1="8" y1="20" x2="8" y2="26" stroke="currentColor" strokeWidth="1.6" />
      <line x1="16" y1="2" x2="16" y2="8" stroke="currentColor" strokeWidth="1.6" />
      <rect x="13.5" y="8" width="5" height="6" fill="none" stroke="currentColor" strokeWidth="1.6" />
      <line x1="16" y1="14" x2="16" y2="19" stroke="currentColor" strokeWidth="1.6" />
      <line x1="24" y1="9" x2="24" y2="14" stroke="currentColor" strokeWidth="1.6" />
      <rect x="21.5" y="14" width="5" height="12" fill="currentColor" />
      <line x1="24" y1="26" x2="24" y2="30" stroke="currentColor" strokeWidth="1.6" />
    </svg>
  )
}

function App() {
  const [ticker, setTicker] = useState('')
  const [submittedTicker, setSubmittedTicker] = useState(null)
  // 普通分析和深度分析(Best-of-N)各自维护自己的SSE连接，mode只决定右边两个
  // 面板此刻展示哪一路的结果——两条流互不干扰，切换只是换了"看哪个"
  const [mode, setMode] = useState('normal')
  const agent = useAgentAnalysis()
  const bestOfN = useBestOfNAnalysis()

  // 断线重连：页面刷新后，agent/bestOfN两边的sessionStorage可能都存着一个
  // 还没确认失效的task（比如先跑了一次普通分析，跑完之后又跑了一次深度
  // 分析），只应该恢复"最后一次"那个——用各自存的startedAt比较，只接回
  // 更晚发起的那个，避免刷新之后同时接回两条互相无关的旧流。
  // resumeAttempted这个ref是必须的：StrictMode开发模式下这个effect会跑
  // 两遍，resume()内部会真的开一个新的EventSource连接，不加这道守卫会
  // 在开发模式下短暂开出两条重复订阅同一个task的连接（不会重复花钱，但
  // 会导致log临时出现重复条目）——ref跨越StrictMode的模拟卸载/重装依然
  // 保留，能挡住第二次真正执行。
  const resumeAttempted = useRef(false)
  useEffect(() => {
    if (resumeAttempted.current) return
    resumeAttempted.current = true

    const normalSession = agent.resume()
    const deepSession = bestOfN.resume()
    const winner =
      normalSession && deepSession
        ? normalSession.startedAt >= deepSession.startedAt
          ? { session: normalSession, mode: 'normal' }
          : { session: deepSession, mode: 'deep' }
        : normalSession
          ? { session: normalSession, mode: 'normal' }
          : deepSession
            ? { session: deepSession, mode: 'deep' }
            : null

    if (winner) {
      setTicker(winner.session.ticker)
      setSubmittedTicker(winner.session.ticker)
      setMode(winner.mode)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  function runAnalysis(nextMode) {
    const trimmed = ticker.trim().toUpperCase()
    if (!trimmed) return
    setSubmittedTicker(trimmed)
    setMode(nextMode)
    if (nextMode === 'deep') {
      bestOfN.start(trimmed)
    } else {
      agent.start(trimmed)
    }
  }

  function handleSubmit(e) {
    e.preventDefault()
    runAnalysis('normal')
  }

  function handleDeepAnalysis() {
    runAnalysis('deep')
  }

  const isDeep = mode === 'deep'
  const activeLog = isDeep ? bestOfN.selectedLog : agent.log
  const activeResult = isDeep ? (bestOfN.result ? bestOfN.result.selected : null) : agent.result
  const activeStreamError = isDeep ? bestOfN.streamError : agent.streamError
  const activeDone = isDeep ? bestOfN.done : agent.done
  const activeElapsedMs = isDeep ? bestOfN.elapsedMs : agent.elapsedMs

  return (
    <div className="min-h-screen bg-gradient-to-b from-indigo-50 via-gray-50 to-gray-50 text-gray-900">
      <div className="mx-auto max-w-6xl px-4 py-6">
        {/* 细长条navbar：标题和输入框同一行，把纵向空间尽量留给下面的dashboard */}
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-2">
            <LogoMark className="h-7 w-7 shrink-0" />
            {/* 标题用全大写+拉宽字距（"Terminal"方向），字体沿用系统默认无衬线
                （"Grotesk"方向：不额外引入自定义字体，只收紧标题字距/加粗） */}
            <h1 className="whitespace-nowrap text-sm font-bold uppercase tracking-wider text-gray-900">
              Fin Report Agent
              <span className="block text-[9px] font-medium tracking-widest text-gray-400">
                US Equity Research
              </span>
            </h1>
          </div>

          <form onSubmit={handleSubmit} className="flex min-w-[260px] flex-1 items-center gap-2.5">
            {/* 搜索栏用下划线代替边框方框（"Ledger"方向） */}
            <div className="relative flex-1">
              <Search className="pointer-events-none absolute left-0.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-gray-400" />
              <input
                value={ticker}
                onChange={(e) => setTicker(e.target.value)}
                placeholder="输入 ticker，如 AAPL"
                className="w-full border-0 border-b-[1.5px] border-gray-300 bg-transparent py-1.5 pl-6 pr-2 uppercase focus:border-[#00AA6F] focus:outline-none"
              />
            </div>
            {/* 两个按钮用圆角+品牌绿（"Signal"方向），"深度分析"保留原来的Sparkles图标 */}
            <button
              type="submit"
              className="shrink-0 rounded-xl bg-[#00AA6F] px-3 py-1 text-xs font-medium text-white shadow-sm transition hover:opacity-90"
            >
              分析
            </button>
            <button
              type="button"
              onClick={handleDeepAnalysis}
              title={`生成${BEST_OF_N_CANDIDATE_COUNT}份候选简报并择优展示，API调用量/花费约为普通分析的4倍以上（过程裁判判定某候选决策过程不合格时会触发一次整改重试，成本会更高）；${BEST_OF_N_CANDIDATE_COUNT}份候选并行生成，等待时间不是简单的4倍`}
              className="flex shrink-0 items-center gap-1 rounded-xl border border-[#00AA6F]/30 bg-white px-3 py-1 text-xs font-medium text-[#00AA6F] shadow-sm transition hover:bg-[#00AA6F]/5"
            >
              <Sparkles className="h-3 w-3" />
              深度分析
            </button>
          </form>
        </div>
        {/* 下面两行字用左侧绿色竖线+"›"引导符（"Terminal"方向） */}
        <div className="mt-2 border-l-2 border-[#00AA6F] pl-2.5">
          <p className="text-xs text-gray-400">
            <span className="text-[#00AA6F]">›</span> 输入一家美股公司的 ticker，结构化数据即刻呈现；随后 Agent 自主展开多轮推理，逐步生成完整的投研简报。
          </p>
          <p className="text-xs text-gray-400">
            「深度分析」会生成{BEST_OF_N_CANDIDATE_COUNT}份候选简报并用奖励打分选出最优的一份，耗时和成本更高。本工具仅做信息整理与解读，不构成投资建议。
          </p>
        </div>

        {submittedTicker && (
          <div className="mt-6 space-y-4">
            <CompanyProfilePanel ticker={submittedTicker} />
            <FinancialsPanel ticker={submittedTicker} />
            <FinancialsHistoryPanel ticker={submittedTicker} />

            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
              <SectorPanel ticker={submittedTicker} />
              <PeerComparisonPanel ticker={submittedTicker} />
            </div>

            <ThematicFlowPanel ticker={submittedTicker} />

            {isDeep && (
              <Suspense
                fallback={
                  <div className="flex items-center gap-2 rounded-xl border border-gray-200 bg-white p-4 text-xs text-gray-400 shadow-sm">
                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                    候选对比面板加载中…
                  </div>
                }
              >
                <CandidateComparisonPanel
                  candidates={bestOfN.candidates}
                  totalCount={BEST_OF_N_CANDIDATE_COUNT}
                  result={bestOfN.result}
                  done={bestOfN.done}
                  streamError={bestOfN.streamError}
                />
              </Suspense>
            )}

            <div className="grid grid-cols-1 gap-4 md:grid-cols-[1fr_2fr]">
              <AgentReasoningPanel
                log={activeLog}
                result={activeResult}
                streamError={activeStreamError}
                done={activeDone}
                elapsedMs={activeElapsedMs}
              />
              <ReportPanel result={activeResult} />
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default App
