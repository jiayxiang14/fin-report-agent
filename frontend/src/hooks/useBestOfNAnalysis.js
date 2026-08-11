import { useCallback, useEffect, useRef, useState } from 'react'
import { useElapsedTimer } from './useElapsedTimer'

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
export function useBestOfNAnalysis() {
  const [candidates, setCandidates] = useState([])
  const [result, setResult] = useState(null)
  const [selectedLog, setSelectedLog] = useState([])
  const [streamError, setStreamError] = useState(null)
  const [done, setDone] = useState(false)
  const sourceRef = useRef(null)
  const logsByCandidateRef = useRef({})
  const nextKeyRef = useRef(0)
  const { elapsedMs, start: startTimer, stop: stopTimer } = useElapsedTimer()

  useEffect(() => {
    return () => {
      sourceRef.current?.close()
      stopTimer()
    }
  }, [stopTimer])

  const start = useCallback((ticker) => {
    sourceRef.current?.close()
    setCandidates([])
    setResult(null)
    setSelectedLog([])
    setStreamError(null)
    setDone(false)
    logsByCandidateRef.current = {}
    nextKeyRef.current = 0
    startTimer()

    const source = new EventSource(`/api/analyze/${ticker}/best-of-n/stream`)
    sourceRef.current = source

    function appendToCandidateLog(candidateIndex, entry) {
      nextKeyRef.current += 1
      const bucket = logsByCandidateRef.current[candidateIndex] || []
      bucket.push({ key: nextKeyRef.current, ...entry })
      logsByCandidateRef.current[candidateIndex] = bucket
    }

    source.onmessage = (e) => {
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
          setDone(true)
          stopTimer()
          source.close()
          break
        case 'error':
          setStreamError(event.message)
          setDone(true)
          stopTimer()
          source.close()
          break
        default:
          break
      }
    }

    source.onerror = () => {
      setDone((current) => {
        if (!current) setStreamError('与服务器的连接中断')
        return true
      })
      stopTimer()
      source.close()
    }
  }, [startTimer, stopTimer])

  return { candidates, result, selectedLog, streamError, done, elapsedMs, start }
}
