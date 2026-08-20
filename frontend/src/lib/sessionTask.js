// 断线重连用：把"当前正在跑/刚跑完的分析"的task_id存进sessionStorage，
// 刷新页面能接回同一次分析，不用重新触发一次真实花钱的LLM调用。用
// sessionStorage而不是localStorage——只在同一个浏览器会话内有效（关掉
// 标签页/窗口就清空），避免几天后重新打开浏览器还去接一个早就该结束的
// 旧任务；真正的过期兜底在后端（按交易日过期，跳过周末），这里只是本地
// 这一层更保守的选择。
//
// try/catch 包一层是因为隐私模式/存储被用户禁用时 sessionStorage 可能
// 不可用——断线重连是锦上添花的能力，不该让分析本身因为存储失败而崩掉，
// 静默降级成"这次没法恢复"就够了。

export function saveTaskSession(key, session) {
  try {
    sessionStorage.setItem(key, JSON.stringify(session))
  } catch {
    // 存储不可用，静默忽略
  }
}

export function loadTaskSession(key) {
  try {
    const raw = sessionStorage.getItem(key)
    return raw ? JSON.parse(raw) : null
  } catch {
    return null
  }
}

export function clearTaskSession(key) {
  try {
    sessionStorage.removeItem(key)
  } catch {
    // 存储不可用，静默忽略
  }
}
