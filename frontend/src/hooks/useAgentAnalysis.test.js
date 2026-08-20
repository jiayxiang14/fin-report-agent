import { act, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { installMockEventSource, MockEventSource } from '../test/mockEventSource'
import { installMockTaskStart } from '../test/mockTaskStart'
import { useAgentAnalysis } from './useAgentAnalysis'

// start() 现在是 async（先POST /start拿task_id，再订阅），所有调用都要用
// `await act(async () => { await ... })`包一层，不然EventSource还没真的
// 开出来就去读MockEventSource.latest()会拿到undefined。
async function startAnalysis(result, ticker) {
  await act(async () => {
    await result.current.start(ticker)
  })
}

describe('useAgentAnalysis', () => {
  beforeEach(() => {
    installMockEventSource()
    installMockTaskStart()
    sessionStorage.clear()
  })

  afterEach(() => {
    MockEventSource.reset()
    sessionStorage.clear()
  })

  it('POSTs /start (with a session_id header) then opens a connection to the task-id-based stream URL', async () => {
    const { result } = renderHook(() => useAgentAnalysis())
    await startAnalysis(result, 'AAPL')

    const [url, init] = global.fetch.mock.calls[0]
    expect(url).toBe('/api/analyze/AAPL/start')
    expect(init.method).toBe('POST')
    expect(init.headers['X-Session-Id']).toEqual(expect.any(String))
    expect(MockEventSource.latest().url).toBe('/api/analyze/stream/mock-task-1')
  })

  it('reuses the same session_id across repeated start() calls in the same tab', async () => {
    const { result } = renderHook(() => useAgentAnalysis())
    await startAnalysis(result, 'AAPL')
    await startAnalysis(result, 'MSFT')

    const firstSessionId = global.fetch.mock.calls[0][1].headers['X-Session-Id']
    const secondSessionId = global.fetch.mock.calls[1][1].headers['X-Session-Id']
    expect(firstSessionId).toBe(secondSessionId)
  })

  it('accumulates reasoning/tool_call events into an ordered log with stable increasing keys', async () => {
    const { result } = renderHook(() => useAgentAnalysis())
    await startAnalysis(result, 'AAPL')
    const source = MockEventSource.latest()

    act(() => source.emit({ type: 'reasoning', turn: 1, text: '先查财务数据' }))
    act(() =>
      source.emit({ type: 'tool_call_started', turn: 1, tool_name: 'get_financials' })
    )
    act(() =>
      source.emit({
        type: 'tool_call_finished',
        turn: 1,
        tool_name: 'get_financials',
        is_error: false,
        summary: '{"entity_name":"Apple Inc."}',
      })
    )

    expect(result.current.log).toHaveLength(3)
    expect(result.current.log.map((entry) => entry.kind)).toEqual([
      'reasoning',
      'tool_started',
      'tool_finished',
    ])
    expect(result.current.log[2].isError).toBe(false)
    // key必须严格递增，前端拿它当React list key，重复/乱序会导致渲染错位
    expect(result.current.log.map((entry) => entry.key)).toEqual([1, 2, 3])
  })

  it('sets result and done, and closes the connection, on a done event', async () => {
    const { result } = renderHook(() => useAgentAnalysis())
    await startAnalysis(result, 'AAPL')
    const source = MockEventSource.latest()

    const fakeResult = { ticker: 'AAPL', final_report: '<conclusion>买入</conclusion>' }
    act(() => source.emit({ type: 'done', result: fakeResult }))

    expect(result.current.result).toEqual(fakeResult)
    expect(result.current.done).toBe(true)
    expect(source.close).toHaveBeenCalledTimes(1)
  })

  it('surfaces a backend-sent error event as streamError and marks done', async () => {
    const { result } = renderHook(() => useAgentAnalysis())
    await startAnalysis(result, 'AAPL')
    const source = MockEventSource.latest()

    act(() => source.emit({ type: 'error', message: 'DeepSeek 余额不足' }))

    expect(result.current.streamError).toBe('DeepSeek 余额不足')
    expect(result.current.done).toBe(true)
    expect(source.close).toHaveBeenCalledTimes(1)
  })

  it('treats a raw connection failure (onerror, no prior done/error) as a stream error', async () => {
    const { result } = renderHook(() => useAgentAnalysis())
    await startAnalysis(result, 'AAPL')
    const source = MockEventSource.latest()

    act(() => source.triggerError())

    expect(result.current.done).toBe(true)
    expect(result.current.streamError).toBe('与服务器的连接中断')
  })

  it('does not overwrite an already-received backend error message with the generic onerror message', async () => {
    // 真实场景：后端先发了明确的error事件（比如"402余额不足"），浏览器紧接着
    // 因为连接关闭也触发一次onerror——不能让后面这个通用消息把前面具体的原因覆盖掉
    const { result } = renderHook(() => useAgentAnalysis())
    await startAnalysis(result, 'AAPL')
    const source = MockEventSource.latest()

    act(() => source.emit({ type: 'error', message: 'DeepSeek 余额不足' }))
    act(() => source.triggerError())

    expect(result.current.streamError).toBe('DeepSeek 余额不足')
  })

  it('closes the previous connection and resets state when start is called again for a new ticker', async () => {
    const { result } = renderHook(() => useAgentAnalysis())
    await startAnalysis(result, 'AAPL')
    const firstSource = MockEventSource.latest()
    act(() => firstSource.emit({ type: 'reasoning', turn: 1, text: 'x' }))
    expect(result.current.log).toHaveLength(1)

    await startAnalysis(result, 'MSFT')

    expect(firstSource.close).toHaveBeenCalledTimes(1)
    expect(global.fetch.mock.calls[1][0]).toBe('/api/analyze/MSFT/start')
    expect(MockEventSource.latest()).not.toBe(firstSource)
    expect(result.current.log).toEqual([]) // 旧ticker的日志不该串到新一轮分析里
    expect(result.current.done).toBe(false)
  })

  it('surfaces the backend detail message when /start responds with a non-2xx status (e.g. rate limited)', async () => {
    global.fetch = async () => ({
      ok: false,
      json: async () => ({ detail: '请求过于频繁，请稍后再试' }),
    })

    const { result } = renderHook(() => useAgentAnalysis())
    await startAnalysis(result, 'AAPL')

    expect(result.current.streamError).toBe('请求过于频繁，请稍后再试')
    expect(result.current.done).toBe(true)
    expect(MockEventSource.instances).toHaveLength(0) // 没拿到task_id，不该尝试开连接
  })

  it('resume() returns null and does not open a connection when nothing was saved', () => {
    sessionStorage.clear()
    const { result } = renderHook(() => useAgentAnalysis())

    const session = result.current.resume()

    expect(session).toBeNull()
    expect(MockEventSource.instances).toHaveLength(0)
  })

  it('resume() reconnects to a previously saved task_id without calling /start again', async () => {
    sessionStorage.setItem(
      'fin-report-agent:normal-task',
      JSON.stringify({ ticker: 'AAPL', taskId: 'saved-task-id', startedAt: 123 })
    )
    const { result } = renderHook(() => useAgentAnalysis())

    let session
    act(() => {
      session = result.current.resume()
    })

    expect(session).toEqual({ ticker: 'AAPL', taskId: 'saved-task-id', startedAt: 123 })
    expect(global.fetch).not.toHaveBeenCalled()
    expect(MockEventSource.latest().url).toBe('/api/analyze/stream/saved-task-id')
  })

  it('clears the saved session when the backend reports task_not_found on reconnect', async () => {
    sessionStorage.setItem(
      'fin-report-agent:normal-task',
      JSON.stringify({ ticker: 'AAPL', taskId: 'stale-task-id', startedAt: 123 })
    )
    const { result } = renderHook(() => useAgentAnalysis())
    act(() => result.current.resume())
    const source = MockEventSource.latest()

    act(() => source.emit({ type: 'error', code: 'task_not_found', message: '找不到了' }))

    expect(result.current.streamError).toBe('找不到了')
    expect(sessionStorage.getItem('fin-report-agent:normal-task')).toBeNull()
  })
})
