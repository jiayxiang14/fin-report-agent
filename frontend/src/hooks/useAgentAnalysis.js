import { useCallback, useEffect, useRef, useState } from 'react'
import { useElapsedTimer } from './useElapsedTimer'

// 开连接的逻辑刻意不放在 useEffect(() => {...}, [ticker]) 里，而是暴露一个
// start(ticker) 函数由提交表单的事件处理函数直接调用。原因：React StrictMode
// 在开发模式下会对组件"挂载阶段"的 effect 故意跑两遍（setup→cleanup→setup），
// 而 AgentReasoningPanel 只有在用户第一次提交ticker后才第一次出现在树里——这
// 意味着如果开 EventSource 连接的代码放在 useEffect 里，页面刷新后的第一次分析
// 会真实发出两次 SSE 请求，也就是两次真实的、要花钱的 Agent Loop（LLM API调用），
// 不是无害的"多打一次只读GET"。事件处理函数（表单提交）不会被 StrictMode 重复
// 调用，所以把开连接这个有真实副作用/成本的操作放在这里才是对的。
export function useAgentAnalysis() {
  const [log, setLog] = useState([])
  const [result, setResult] = useState(null)
  const [streamError, setStreamError] = useState(null)
  const [done, setDone] = useState(false)
  const sourceRef = useRef(null)
  const nextKey = useRef(0)
  const { elapsedMs, start: startTimer, stop: stopTimer } = useElapsedTimer()

  useEffect(() => {
    return () => {
      sourceRef.current?.close()
      stopTimer()
    }
  }, [stopTimer])

  const start = useCallback((ticker) => {
    sourceRef.current?.close()
    setLog([])
    setResult(null)
    setStreamError(null)
    setDone(false)
    nextKey.current = 0
    startTimer()

    const source = new EventSource(`/api/analyze/${ticker}/stream`)
    sourceRef.current = source

    function appendLog(entry) {
      nextKey.current += 1
      setLog((prev) => [...prev, { key: nextKey.current, ...entry }])
    }

    source.onmessage = (e) => {
      const event = JSON.parse(e.data)
      switch (event.type) {
        case 'reasoning':
          appendLog({ kind: 'reasoning', turn: event.turn, text: event.text })
          break
        case 'tool_call_started':
          appendLog({ kind: 'tool_started', turn: event.turn, toolName: event.tool_name })
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
          setResult(event.result)
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
      // 浏览器原生EventSource在连接层失败时也会触发这个回调（不只是我们
      // 自己发的error事件），如果这时候还没收到 done/error，说明连接本身断了
      setDone((current) => {
        if (!current) setStreamError('与服务器的连接中断')
        return true
      })
      stopTimer()
      source.close()
    }
  }, [startTimer, stopTimer])

  return { log, result, streamError, done, elapsedMs, start }
}
