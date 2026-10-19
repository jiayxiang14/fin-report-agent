import { useCallback, useEffect, useRef, useState } from 'react'
import { getSessionId } from '../lib/clientSession'
import { clearTaskSession, loadTaskSession, saveTaskSession } from '../lib/sessionTask'
import { useElapsedTimer } from './useElapsedTimer'

const SESSION_KEY = 'fin-report-agent:normal-task'

// 开连接的逻辑刻意不放在 useEffect(() => {...}, [ticker]) 里，而是暴露一个
// start(ticker) 函数由提交表单的事件处理函数直接调用。原因：React StrictMode
// 在开发模式下会对组件"挂载阶段"的 effect 故意跑两遍（setup→cleanup→setup），
// 而 AgentReasoningPanel 只有在用户第一次提交ticker后才第一次出现在树里——这
// 意味着如果开 EventSource 连接的代码放在 useEffect 里，页面刷新后的第一次分析
// 会真实发出两次 SSE 请求，也就是两次真实的、要花钱的 Agent Loop（LLM API调用），
// 不是无害的"多打一次只读GET"。事件处理函数（表单提交）不会被 StrictMode 重复
// 调用，所以把开连接这个有真实副作用/成本的操作放在这里才是对的。
//
// 断线重连：后端现在把"发起分析"（POST /start，真实触发LLM调用）和"订阅事件流"
// （GET /stream/{task_id}，纯读，可以随便重连）拆开了——start() 先POST拿到
// task_id、存进sessionStorage，再订阅这个task_id的流；resume() 只做后半段
// （读sessionStorage里存的task_id、直接订阅，不重新POST），给页面刷新后
// 调用，接的是同一次分析，不会重复花钱。
//
// 网络抖动自动重连：连接层失败（onerror，且还没收到done/error）不立刻判定失败，
// 按RECONNECT_DELAYS_MS退避重新订阅同一个task_id，全部失败才报"连接中断"。
// 不用浏览器EventSource自带的自动重连：后端每次订阅都从头全量重放事件，原生
// 重连会把已经展示过的事件再追加一遍，所以这里先close掉、自己重新开连接，
// 并且等新连接收到第一个事件时才清空旧状态再接收重放——重连期间（以及重连
// 最终失败时）用户之前已经看到的推理过程不会被提前清掉。
// 总共约31秒：网络抖动通常几秒内恢复；服务重启（部署/崩溃后自动恢复任务）
// 一般要十几秒，窗口太短的话用户还得手动刷新一次才能接回
const RECONNECT_DELAYS_MS = [1000, 2000, 4000, 8000, 16000]

const EMPTY_USAGE_TOTALS = {
  inputTokens: 0,
  outputTokens: 0,
  cacheCreationInputTokens: 0,
  cacheReadInputTokens: 0,
}

export function useAgentAnalysis() {
  const [log, setLog] = useState([])
  const [result, setResult] = useState(null)
  const [streamError, setStreamError] = useState(null)
  const [done, setDone] = useState(false)
  // 当前还没走完这一轮的增量文字——跟 log 里 kind:'reasoning' 的完整条目是
  // 两回事：那条要等这一轮响应完全结束才会追加，这个是逐字符/逐token实时长大的，
  // 那条出现的瞬间这个就清空（避免同一段话被展示两遍）。turn为null表示当前
  // 没有正在流式输出的文字。
  const [liveReasoning, setLiveReasoning] = useState({ turn: null, text: '' })
  // 实时累加的token用量——分析跑完之后应该改看 result.total_input_tokens 等
  // 字段（那才是最终权威值，跟这里独立算出来的累加值理论上应该一致，但没必要
  // 在跑完之后还维护两份），这份状态只在"还没done"期间有意义
  const [usageTotals, setUsageTotals] = useState(EMPTY_USAGE_TOTALS)
  // 失败是外部故障（LLM余额不足/服务宕机）且后端保存了进度时为true，前端
  // 据此显示"从断点重试"
  const [retryable, setRetryable] = useState(false)
  const [cancelled, setCancelled] = useState(false)
  const sourceRef = useRef(null)
  // 当前订阅的是哪个task，取消/重试要用
  const currentTaskRef = useRef(null)
  const retryTimerRef = useRef(null)
  const nextKey = useRef(0)
  const { elapsedMs, start: startTimer, stop: stopTimer } = useElapsedTimer()

  useEffect(() => {
    return () => {
      clearTimeout(retryTimerRef.current)
      sourceRef.current?.close()
      stopTimer()
    }
  }, [stopTimer])

  const attach = useCallback((ticker, taskId, startedAt) => {
    clearTimeout(retryTimerRef.current)
    retryTimerRef.current = null
    // 计时器只在发起/接回时启动一次，自动重连不重启——重连期间分析本身还在
    // 后端继续跑，用时应该连续累计
    startTimer(startedAt)
    let failures = 0

    currentTaskRef.current = { ticker, taskId, startedAt }

    function resetState() {
      setLog([])
      setResult(null)
      setStreamError(null)
      setRetryable(false)
      setCancelled(false)
      setDone(false)
      setLiveReasoning({ turn: null, text: '' })
      setUsageTotals(EMPTY_USAGE_TOTALS)
      nextKey.current = 0
    }

    function connect(isRetry) {
      sourceRef.current?.close()
      if (!isRetry) resetState()

      const source = new EventSource(`/api/analyze/stream/${taskId}`)
      sourceRef.current = source
      let finished = false
      let pendingReset = isRetry

      function appendLog(entry) {
        nextKey.current += 1
        setLog((prev) => [...prev, { key: nextKey.current, ...entry }])
      }

      source.onmessage = (e) => {
        if (sourceRef.current !== source) return
        if (pendingReset) {
          // 重连成功、后端开始从头重放：这时才清掉旧状态，并把失败计数归零
          // （之后再断一次，重新拥有完整的重试次数）
          pendingReset = false
          failures = 0
          resetState()
        }
        const event = JSON.parse(e.data)
        switch (event.type) {
          case 'reasoning_delta':
            setLiveReasoning((prev) =>
              prev.turn === event.turn
                ? { turn: event.turn, text: prev.text + event.text }
                : { turn: event.turn, text: event.text }
            )
            break
          case 'reasoning':
            appendLog({ kind: 'reasoning', turn: event.turn, text: event.text })
            // 这一轮的完整文字已经作为正式日志条目落地了，实时增量缓冲区不再
            // 需要——不清空的话，下一轮开始前会有一瞬间同时展示"上一轮的完整
            // 条目"和"还没被清掉的上一轮增量文字"，看起来像重复了一遍
            setLiveReasoning({ turn: null, text: '' })
            break
          case 'llm_usage':
            setUsageTotals((prev) => ({
              inputTokens: prev.inputTokens + event.input_tokens,
              outputTokens: prev.outputTokens + event.output_tokens,
              cacheCreationInputTokens: prev.cacheCreationInputTokens + event.cache_creation_input_tokens,
              cacheReadInputTokens: prev.cacheReadInputTokens + event.cache_read_input_tokens,
            }))
            break
          case 'tool_call_started':
            appendLog({ kind: 'tool_started', turn: event.turn, toolName: event.tool_name })
            break
          case 'resumed':
            // 服务重启后自动恢复 / 用户点了从断点重试：后端已经把被打断那一轮
            // 推出去的半截事件截掉了，这里只需要清掉可能残留的实时增量文字，
            // 并在日志里标出"从哪里接着跑"
            setLiveReasoning({ turn: null, text: '' })
            appendLog({ kind: 'resumed', turn: event.turn, reason: event.reason })
            break
          case 'tool_call_finished':
            appendLog({
              kind: 'tool_finished',
              turn: event.turn,
              toolName: event.tool_name,
              isError: event.is_error,
              summary: event.summary,
            })
            break
          case 'done':
            finished = true
            setResult(event.result)
            setDone(true)
            setLiveReasoning({ turn: null, text: '' })
            stopTimer(event.elapsed_ms)
            source.close()
            break
          case 'error':
            if (event.code === 'task_not_found') {
              // 存的task_id后端已经不认得了（过期被清理/进程重启过）——不是
              // 分析本身失败，是"这次真的接不回去了"，把本地记录清掉，不然
              // 下次刷新还会再尝试一次同样失败的重连
              clearTaskSession(SESSION_KEY)
            }
            finished = true
            if (event.code === 'cancelled') {
              setCancelled(true)
            } else {
              setStreamError(event.message)
              setRetryable(event.retryable === true)
            }
            setDone(true)
            setLiveReasoning({ turn: null, text: '' })
            stopTimer(event.elapsed_ms)
            source.close()
            break
          default:
            break
        }
      }

      source.onerror = () => {
        // 浏览器原生EventSource在连接层失败时触发这个回调。先close掉，不让
        // 浏览器自己重连（见文件顶部RECONNECT_DELAYS_MS的注释）
        source.close()
        // 已经收到done/error（连接随后关闭也会触发onerror），或者这条连接
        // 已经被更新的一次attach/重连取代了——都不是"连接意外中断"
        if (finished || sourceRef.current !== source) return
        if (failures < RECONNECT_DELAYS_MS.length) {
          retryTimerRef.current = setTimeout(() => connect(true), RECONNECT_DELAYS_MS[failures])
          failures += 1
          return
        }
        finished = true
        setStreamError('与服务器的连接中断')
        setDone(true)
        setLiveReasoning({ turn: null, text: '' })
        stopTimer()
      }
    }

    connect(false)
  }, [startTimer, stopTimer])

  const start = useCallback(
    async (ticker) => {
      // 先同步重置一遍可见状态，避免POST这个网络往返期间UI还闪一下上一次
      // 分析的旧结果——attach()里也会重置一遍，两次重置互不冲突
      setResult(null)
      setStreamError(null)
      setDone(false)

      const sessionId = getSessionId()
      const headers = sessionId ? { 'X-Session-Id': sessionId } : {}

      let response
      try {
        response = await fetch(`/api/analyze/${ticker}/start`, { method: 'POST', headers })
      } catch {
        setStreamError('发起分析失败：无法连接服务器')
        setDone(true)
        return
      }
      if (!response.ok) {
        let detail = '发起分析失败，请稍后重试'
        try {
          const body = await response.json()
          if (body.detail) detail = body.detail
        } catch {
          // 响应体不是JSON，用默认文案
        }
        setStreamError(detail)
        setDone(true)
        return
      }
      const { task_id: taskId } = await response.json()
      const startedAt = Date.now()
      saveTaskSession(SESSION_KEY, { ticker, taskId, startedAt })
      attach(ticker, taskId, startedAt)
    },
    [attach]
  )

  // 取消正在跑的分析。结果不在这里处理：后端会在事件流里推一个
  // code=cancelled 的终止事件，跟其它终止事件走同一条路径
  const cancel = useCallback(async () => {
    const current = currentTaskRef.current
    if (!current) return
    try {
      await fetch(`/api/analyze/tasks/${current.taskId}/cancel`, { method: 'POST' })
    } catch {
      // 网络失败：分析会照常跑完，没有别的可做
    }
  }, [])

  // 从断点重试：后端从最近的存档点接着跑、沿用同一个task_id，这里重新订阅
  // 同一条流即可（后端会先重放截断到存档点的历史事件）
  const retry = useCallback(async () => {
    const current = currentTaskRef.current
    if (!current) return
    const sessionId = getSessionId()
    const headers = sessionId ? { 'X-Session-Id': sessionId } : {}
    let response
    try {
      response = await fetch(`/api/analyze/tasks/${current.taskId}/retry`, { method: 'POST', headers })
    } catch {
      setStreamError('重试失败：无法连接服务器')
      return
    }
    if (!response.ok) {
      let detail = '重试失败，请重新发起分析'
      try {
        const body = await response.json()
        if (body.detail) detail = body.detail
      } catch {
        // 响应体不是JSON，用默认文案
      }
      setStreamError(detail)
      setRetryable(false)
      return
    }
    attach(current.ticker, current.taskId, current.startedAt)
  }, [attach])

  // 页面刷新后调用：如果sessionStorage里存过还没被判定失效的task，直接接回去，
  // 不重新POST（不触发新的LLM调用）。返回存过的session（{ticker, taskId,
  // startedAt}）给调用方决定要不要恢复"当前正在看哪个ticker"这部分UI状态，
  // 没有可恢复的东西时返回null。
  const resume = useCallback(() => {
    const session = loadTaskSession(SESSION_KEY)
    if (!session) return null
    attach(session.ticker, session.taskId, session.startedAt)
    return session
  }, [attach])

  return {
    log,
    result,
    streamError,
    retryable,
    cancelled,
    done,
    elapsedMs,
    liveReasoning,
    usageTotals,
    start,
    resume,
    cancel,
    retry,
  }
}
