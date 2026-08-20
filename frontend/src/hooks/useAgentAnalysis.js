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

  const attach = useCallback((ticker, taskId) => {
    sourceRef.current?.close()
    setLog([])
    setResult(null)
    setStreamError(null)
    setDone(false)
    nextKey.current = 0
    startTimer()

    const source = new EventSource(`/api/analyze/stream/${taskId}`)
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
          if (event.code === 'task_not_found') {
            // 存的task_id后端已经不认得了（过期被清理/进程重启过）——不是
            // 分析本身失败，是"这次真的接不回去了"，把本地记录清掉，不然
            // 下次刷新还会再尝试一次同样失败的重连
            clearTaskSession(SESSION_KEY)
          }
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
      saveTaskSession(SESSION_KEY, { ticker, taskId, startedAt: Date.now() })
      attach(ticker, taskId)
    },
    [attach]
  )

  // 页面刷新后调用：如果sessionStorage里存过还没被判定失效的task，直接接回去，
  // 不重新POST（不触发新的LLM调用）。返回存过的session（{ticker, taskId,
  // startedAt}）给调用方决定要不要恢复"当前正在看哪个ticker"这部分UI状态，
  // 没有可恢复的东西时返回null。
  const resume = useCallback(() => {
    const session = loadTaskSession(SESSION_KEY)
    if (!session) return null
    attach(session.ticker, session.taskId)
    return session
  }, [attach])

  return { log, result, streamError, done, elapsedMs, start, resume }
}
