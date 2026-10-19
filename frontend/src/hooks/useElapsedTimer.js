import { useCallback, useRef, useState } from 'react'

// 200ms一跳：够让秒数看起来在平滑走动（不是每秒才跳一次的生硬感），又不会
// 频繁到没必要地拖慢渲染——这只是一个小数字的重渲染，不是什么重量级组件。
const TICK_MS = 200

// useAgentAnalysis/useBestOfNAnalysis共用的计时逻辑：start()记录起始时间点并
// 开始每200ms刷新一次elapsedMs，stop()停止刷新但保留最后算出来的那个值（不是
// 清零）——分析结束后前端要展示"本次用时X秒"，这个值就是停表那一刻的elapsedMs。
//
// start(startedAtEpochMs)可以传入分析真正的发起时间（存在sessionStorage里的
// Date.now()值）：刷新页面/断线重连接回同一个task时从发起时刻接着算，不从0
// 重新开始。所以这里用Date.now()而不是performance.now()——后者的零点是当前
// 页面加载时刻，刷新之后跟之前存下的时间点没法比较。
// stop(finalMs)可以传入后端算好的权威总用时（done/error事件的elapsed_ms），
// 接回一个早已跑完的task时，前端自己算出来的"发起到现在"会把跑完后的空闲
// 时间也算进去，不能用。
export function useElapsedTimer() {
  const [elapsedMs, setElapsedMs] = useState(0)
  const startedAtRef = useRef(null)
  const intervalRef = useRef(null)

  const start = useCallback((startedAtEpochMs) => {
    if (intervalRef.current) clearInterval(intervalRef.current)
    const now = Date.now()
    // 防御：sessionStorage里的值缺失/被篡改成未来时间时，退回到从现在开始算
    startedAtRef.current =
      typeof startedAtEpochMs === 'number' && startedAtEpochMs <= now ? startedAtEpochMs : now
    setElapsedMs(now - startedAtRef.current)
    intervalRef.current = setInterval(() => {
      setElapsedMs(Date.now() - startedAtRef.current)
    }, TICK_MS)
  }, [])

  const stop = useCallback((finalMs) => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current)
      intervalRef.current = null
    }
    if (typeof finalMs === 'number') {
      setElapsedMs(finalMs)
    } else if (startedAtRef.current != null) {
      setElapsedMs(Date.now() - startedAtRef.current)
    }
  }, [])

  return { elapsedMs, start, stop }
}
