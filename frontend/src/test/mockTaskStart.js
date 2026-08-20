// start() 现在先 POST /api/analyze/{ticker}/start（或 best-of-n 那个变体）
// 拿 task_id，再拿 task_id 去开 EventSource——这个mock只做测试需要的最小子集：
// 每次调用返回一个递增的 task_id（不是固定值，方便测试断言"两次start()确实
// 各自走完了完整流程"，不是复用了同一个连接）。
import { vi } from 'vitest'

export function installMockTaskStart() {
  let counter = 0
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => {
      counter += 1
      return {
        ok: true,
        json: async () => ({ task_id: `mock-task-${counter}` }),
      }
    })
  )
}
