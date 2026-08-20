// 每个浏览器标签页一个session_id，随/start请求以X-Session-Id头发给后端，
// 让后端能识别"这个session是不是已经有一个分析正在进行"（防止双开连接
// 抢占本来就紧张的Alpha Vantage/Polygon配额，真实踩过StrictMode导致SSE
// 连接被意外开两次的坑——见 useAgentAnalysis.js 顶部注释）。
//
// 用sessionStorage而不是localStorage：按"一个标签页"这个粒度隔离，两个
// 标签页各自分析不同ticker不该互相挡住——跟 sessionTask.js 存task_id用
// 同一个理由。
//
// try/catch包一层：隐私模式/存储被禁用时sessionStorage可能不可用，这时
// 直接不发送这个header——后端对没有这个header的请求直接放行，不强制
// 要求，是一个可选的额外保护，不是硬性门槛。
const SESSION_ID_KEY = 'fin-report-agent:session-id'

export function getSessionId() {
  try {
    let sessionId = sessionStorage.getItem(SESSION_ID_KEY)
    if (!sessionId) {
      sessionId = crypto.randomUUID()
      sessionStorage.setItem(SESSION_ID_KEY, sessionId)
    }
    return sessionId
  } catch {
    return null
  }
}
