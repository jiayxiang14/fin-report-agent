import { useCallback, useEffect, useRef, useState } from 'react'
import { getSessionId } from '../lib/clientSession'
import { clearTaskSession, loadTaskSession, saveTaskSession } from '../lib/sessionTask'
import { useElapsedTimer } from './useElapsedTimer'

const SESSION_KEY = 'fin-report-agent:deep-task'

// candidate_scored事件（跑的过程中实时到）不带简报正文——只有等done事件里
// 完整的 BestOfNResult.candidates 才有 final_report。这里统一成一个映射函数，
// done到达时用它把三份候选的正文补齐，前端才能真正对比着看，不是只看分数。
function mapDoneCandidate(raw) {
  return {
    index: raw.index,
    temperature: raw.temperature,
    totalScore: raw.total_score,
    ruleScore: raw.rule_score,
    llmScore: raw.llm_score,
    llmReason: raw.llm_reason,
    trajectoryScore: raw.trajectory_score,
    trajectoryReason: raw.trajectory_reason,
    reflexionTriggered: raw.reflexion_triggered,
    finalReport: raw.final_report,
    error: raw.error,
  }
}

// 跟 useAgentAnalysis 一样，把开 EventSource 连接放进 start() 而不是
// useEffect(() => {...}, [ticker])，理由也一样：StrictMode 开发模式下会对
// 挂载阶段的 effect 故意跑两遍，而这里的连接是真实会花钱的3次Agent Loop
// +3次裁判调用，不能被无意双开。
//
// 后端的 reasoning/tool_call_started/tool_call_finished 事件都带了
// candidate_index，这里按候选分桶记录，等 done 事件带来 selected_index 后，
// 把"胜出候选"自己的那一份日志挑出来喂给 AgentReasoningPanel——这样深度分析
// 模式下用户看到的推理过程就是真正被选中展示的那份简报是怎么一步步生成的，
// 而不是一个空面板。
//
// 断线重连：跟 useAgentAnalysis 同一套 start()/resume() 拆分，网络抖动自动
// 重连的策略也一样（退避重试、新连接收到第一个事件才清空旧状态），见那边的注释。
const RECONNECT_DELAYS_MS = [1000, 2000, 4000, 8000, 16000]

export function useBestOfNAnalysis() {
  const [candidates, setCandidates] = useState([])
  const [result, setResult] = useState(null)
  const [selectedLog, setSelectedLog] = useState([])
  const [streamError, setStreamError] = useState(null)
  const [done, setDone] = useState(false)
  const [cancelled, setCancelled] = useState(false)
  const sourceRef = useRef(null)
  const currentTaskRef = useRef(null)
  const retryTimerRef = useRef(null)
  const logsByCandidateRef = useRef({})
  const nextKeyRef = useRef(0)
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
    startTimer(startedAt)
    let failures = 0

    currentTaskRef.current = { taskId }

    function resetState() {
      setCandidates([])
      setResult(null)
      setSelectedLog([])
      setStreamError(null)
      setCancelled(false)
      setDone(false)
      logsByCandidateRef.current = {}
      nextKeyRef.current = 0
    }

    function connect(isRetry) {
      sourceRef.current?.close()
      if (!isRetry) resetState()

      const source = new EventSource(`/api/analyze/best-of-n/stream/${taskId}`)
      sourceRef.current = source
      let finished = false
      let pendingReset = isRetry

      function appendToCandidateLog(candidateIndex, entry) {
        nextKeyRef.current += 1
        const bucket = logsByCandidateRef.current[candidateIndex] || []
        bucket.push({ key: nextKeyRef.current, ...entry })
        logsByCandidateRef.current[candidateIndex] = bucket
      }

      source.onmessage = (e) => {
        if (sourceRef.current !== source) return
        if (pendingReset) {
          pendingReset = false
          failures = 0
          resetState()
        }
        const event = JSON.parse(e.data)
        switch (event.type) {
          case 'reasoning':
            appendToCandidateLog(event.candidate_index, {
              kind: 'reasoning',
              turn: event.turn,
              text: event.text,
            })
            break
          case 'tool_call_started':
            appendToCandidateLog(event.candidate_index, {
              kind: 'tool_started',
              turn: event.turn,
              toolName: event.tool_name,
            })
            break
          case 'tool_call_finished':
            appendToCandidateLog(event.candidate_index, {
              kind: 'tool_finished',
              turn: event.turn,
              toolName: event.tool_name,
              isError: event.is_error,
              summary: event.summary,
            })
            break
          case 'candidate_scored':
            setCandidates((prev) => [
              ...prev,
              {
                index: event.candidate_index,
                temperature: event.temperature,
                totalScore: event.total_score,
                ruleScore: event.rule_score,
                llmScore: event.llm_score,
                llmReason: event.llm_reason,
                trajectoryScore: event.trajectory_score,
                trajectoryReason: event.trajectory_reason,
                reflexionTriggered: event.reflexion_triggered,
                finalReport: null, // 跑的过程中先不带正文，done事件到达后统一补上
                error: null,
              },
            ])
            break
          case 'candidate_failed':
            // 单个候选失败（比如上游余额不足）不影响其它候选继续跑，这里先标记
            // 出来，done事件到达后会用权威的result.candidates覆盖一次
            setCandidates((prev) => [
              ...prev,
              {
                index: event.candidate_index,
                temperature: event.temperature,
                totalScore: null,
                ruleScore: null,
                llmScore: null,
                llmReason: null,
                trajectoryScore: null,
                trajectoryReason: null,
                reflexionTriggered: null,
                finalReport: null,
                error: event.error,
              },
            ])
            break
          case 'done':
            setResult(event.result)
            setCandidates(event.result.candidates.map(mapDoneCandidate))
            setSelectedLog(logsByCandidateRef.current[event.result.selected_index] || [])
            finished = true
            setDone(true)
            stopTimer(event.elapsed_ms)
            source.close()
            break
          case 'error':
            if (event.code === 'task_not_found') {
              clearTaskSession(SESSION_KEY)
            }
            finished = true
            // 深度分析不存检查点（见后端checkpoint.py），失败不提供断点重试
            if (event.code === 'cancelled') {
              setCancelled(true)
            } else {
              setStreamError(event.message)
            }
            setDone(true)
            stopTimer(event.elapsed_ms)
            source.close()
            break
          default:
            break
        }
      }

      source.onerror = () => {
        source.close()
        if (finished || sourceRef.current !== source) return
        if (failures < RECONNECT_DELAYS_MS.length) {
          retryTimerRef.current = setTimeout(() => connect(true), RECONNECT_DELAYS_MS[failures])
          failures += 1
          return
        }
        finished = true
        setStreamError('与服务器的连接中断')
        setDone(true)
        stopTimer()
      }
    }

    connect(false)
  }, [startTimer, stopTimer])

  const start = useCallback(
    async (ticker) => {
      setResult(null)
      setStreamError(null)
      setDone(false)

      const sessionId = getSessionId()
      const headers = sessionId ? { 'X-Session-Id': sessionId } : {}

      let response
      try {
        response = await fetch(`/api/analyze/${ticker}/best-of-n/start`, { method: 'POST', headers })
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

  const cancel = useCallback(async () => {
    const current = currentTaskRef.current
    if (!current) return
    try {
      await fetch(`/api/analyze/tasks/${current.taskId}/cancel`, { method: 'POST' })
    } catch {
      // 网络失败：分析会照常跑完，没有别的可做
    }
  }, [])

  const resume = useCallback(() => {
    const session = loadTaskSession(SESSION_KEY)
    if (!session) return null
    attach(session.ticker, session.taskId, session.startedAt)
    return session
  }, [attach])

  return { candidates, result, selectedLog, streamError, cancelled, done, elapsedMs, start, resume, cancel }
}
