import { useCallback, useRef, useState } from 'react'

// 200ms一跳：够让秒数看起来在平滑走动（不是每秒才跳一次的生硬感），又不会
// 频繁到没必要地拖慢渲染——这只是一个小数字的重渲染，不是什么重量级组件。
const TICK_MS = 200

// useAgentAnalysis/useBestOfNAnalysis共用的计时逻辑：start()记录起始时间点并
// 开始每200ms刷新一次elapsedMs，stop()停止刷新但保留最后算出来的那个值（不是
// 清零）——分析结束后前端要展示"本次用时X秒"，这个值就是停表那一刻的elapsedMs。
export function useElapsedTimer() {
  const [elapsedMs, setElapsedMs] = useState(0)
  const startedAtRef = useRef(null)
  const intervalRef = useRef(null)

  const start = useCallback(() => {
    if (intervalRef.current) clearInterval(intervalRef.current)
    startedAtRef.current = performance.now()
    setElapsedMs(0)
    intervalRef.current = setInterval(() => {
      setElapsedMs(performance.now() - startedAtRef.current)
    }, TICK_MS)
  }, [])

  const stop = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current)
      intervalRef.current = null
    }
    if (startedAtRef.current != null) {
      setElapsedMs(performance.now() - startedAtRef.current)
    }
  }, [])

  return { elapsedMs, start, stop }
}
