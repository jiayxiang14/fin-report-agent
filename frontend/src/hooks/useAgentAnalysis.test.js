import { act, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
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

  it('accumulates reasoning_delta events into liveReasoning, keyed by turn, without touching the ordered log', async () => {
    const { result } = renderHook(() => useAgentAnalysis())
    await startAnalysis(result, 'AAPL')
    const source = MockEventSource.latest()

    act(() => source.emit({ type: 'reasoning_delta', turn: 0, text: '先查' }))
    act(() => source.emit({ type: 'reasoning_delta', turn: 0, text: '一下财务数据' }))

    expect(result.current.liveReasoning).toEqual({ turn: 0, text: '先查一下财务数据' })
    expect(result.current.log).toHaveLength(0) // delta不是正式日志条目，不该出现在log里

    // 完整的 reasoning 事件到达后，这一轮的增量缓冲区应该被清空——不然会跟
    // 正式落地的 log 条目重复展示同一段话
    act(() => source.emit({ type: 'reasoning', turn: 0, text: '先查一下财务数据' }))
    expect(result.current.liveReasoning).toEqual({ turn: null, text: '' })
    expect(result.current.log).toHaveLength(1)
  })

  it('resets liveReasoning to empty when a new turn starts streaming', async () => {
    const { result } = renderHook(() => useAgentAnalysis())
    await startAnalysis(result, 'AAPL')
    const source = MockEventSource.latest()

    act(() => source.emit({ type: 'reasoning_delta', turn: 0, text: '第一轮的文字' }))
    act(() => source.emit({ type: 'reasoning_delta', turn: 1, text: '第二轮' }))

    expect(result.current.liveReasoning).toEqual({ turn: 1, text: '第二轮' })
  })

  it('accumulates llm_usage events into running token totals across turns', async () => {
    const { result } = renderHook(() => useAgentAnalysis())
    await startAnalysis(result, 'AAPL')
    const source = MockEventSource.latest()

    act(() =>
      source.emit({
        type: 'llm_usage',
        turn: 0,
        input_tokens: 1000,
        output_tokens: 100,
        cache_creation_input_tokens: 50,
        cache_read_input_tokens: 0,
      })
    )
    act(() =>
      source.emit({
        type: 'llm_usage',
        turn: 1,
        input_tokens: 2000,
        output_tokens: 200,
        cache_creation_input_tokens: 0,
        cache_read_input_tokens: 1000,
      })
    )

    expect(result.current.usageTotals).toEqual({
      inputTokens: 3000,
      outputTokens: 300,
      cacheCreationInputTokens: 50,
      cacheReadInputTokens: 1000,
    })
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

  describe('automatic reconnect after a connection-level failure', () => {
    beforeEach(() => {
      vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval', 'Date'] })
    })

    afterEach(() => {
      vi.useRealTimers()
    })

    it('does not fail immediately on onerror; reopens the same task stream after a backoff delay', async () => {
      const { result } = renderHook(() => useAgentAnalysis())
      await startAnalysis(result, 'AAPL')
      const first = MockEventSource.latest()

      act(() => first.triggerError())

      expect(first.close).toHaveBeenCalled() // 不让浏览器原生重连，否则重放会重复追加
      expect(result.current.done).toBe(false)
      expect(result.current.streamError).toBeNull()
      expect(MockEventSource.instances).toHaveLength(1)

      act(() => vi.advanceTimersByTime(1000))

      expect(MockEventSource.instances).toHaveLength(2)
      expect(MockEventSource.latest().url).toBe(first.url)
      expect(global.fetch).toHaveBeenCalledTimes(1) // 只重新订阅，不重新POST发起分析
    })

    it('reports a connection error only after every retry has failed', async () => {
      const { result } = renderHook(() => useAgentAnalysis())
      await startAnalysis(result, 'AAPL')

      for (const delay of [1000, 2000, 4000, 8000, 16000]) {
        act(() => MockEventSource.latest().triggerError())
        expect(result.current.done).toBe(false)
        act(() => vi.advanceTimersByTime(delay))
      }
      act(() => MockEventSource.latest().triggerError())

      expect(MockEventSource.instances).toHaveLength(6) // 首次 + 5次重试
      expect(result.current.done).toBe(true)
      expect(result.current.streamError).toBe('与服务器的连接中断')

      act(() => vi.advanceTimersByTime(60_000))
      expect(MockEventSource.instances).toHaveLength(6) // 放弃之后不再继续重试
    })

    it('keeps already-shown events visible while reconnecting, then replaces them with the replay without duplicates', async () => {
      const { result } = renderHook(() => useAgentAnalysis())
      await startAnalysis(result, 'AAPL')
      const first = MockEventSource.latest()
      act(() => first.emit({ type: 'reasoning', turn: 0, text: '第一步' }))

      act(() => first.triggerError())
      act(() => vi.advanceTimersByTime(1000))

      // 新连接还没收到任何事件：旧内容保留，不闪成空白
      expect(result.current.log.map((e) => e.text)).toEqual(['第一步'])

      const second = MockEventSource.latest()
      act(() => second.emit({ type: 'reasoning', turn: 0, text: '第一步' })) // 后端从头重放
      act(() => second.emit({ type: 'reasoning', turn: 1, text: '第二步' }))

      expect(result.current.log.map((e) => e.text)).toEqual(['第一步', '第二步'])
      expect(result.current.log.map((e) => e.key)).toEqual([1, 2])
    })

    it('restores the full retry budget once a reconnected stream delivers an event', async () => {
      const { result } = renderHook(() => useAgentAnalysis())
      await startAnalysis(result, 'AAPL')

      // 用掉两次重试后重连成功
      act(() => MockEventSource.latest().triggerError())
      act(() => vi.advanceTimersByTime(1000))
      act(() => MockEventSource.latest().triggerError())
      act(() => vi.advanceTimersByTime(2000))
      act(() => MockEventSource.latest().emit({ type: 'reasoning', turn: 0, text: 'x' }))

      // 再断一次：退避重新从1秒开始，而不是只剩最后一次4秒的重试
      act(() => MockEventSource.latest().triggerError())
      act(() => vi.advanceTimersByTime(1000))

      expect(MockEventSource.instances).toHaveLength(4)
      expect(result.current.done).toBe(false)
    })

    it('cancels a pending reconnect when the user starts a new analysis', async () => {
      const { result } = renderHook(() => useAgentAnalysis())
      await startAnalysis(result, 'AAPL')
      act(() => MockEventSource.latest().triggerError())

      await startAnalysis(result, 'MSFT')
      act(() => vi.advanceTimersByTime(10_000))

      expect(MockEventSource.instances.map((s) => s.url)).toEqual([
        '/api/analyze/stream/mock-task-1',
        '/api/analyze/stream/mock-task-2',
      ])
    })

    it('does not reconnect when the connection closes after a done event', async () => {
      const { result } = renderHook(() => useAgentAnalysis())
      await startAnalysis(result, 'AAPL')
      const source = MockEventSource.latest()

      act(() => source.emit({ type: 'done', result: { final_report: 'ok' } }))
      act(() => source.triggerError())
      act(() => vi.advanceTimersByTime(10_000))

      expect(MockEventSource.instances).toHaveLength(1)
      expect(result.current.streamError).toBeNull()
    })

    it('cancels a pending reconnect on unmount', async () => {
      const { result, unmount } = renderHook(() => useAgentAnalysis())
      await startAnalysis(result, 'AAPL')
      act(() => MockEventSource.latest().triggerError())

      unmount()
      vi.advanceTimersByTime(10_000)

      expect(MockEventSource.instances).toHaveLength(1)
    })
  })

  describe('elapsed time across refreshes', () => {
    beforeEach(() => {
      vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval', 'Date'] })
    })

    afterEach(() => {
      vi.useRealTimers()
    })

    it('resume() continues the timer from the saved start time instead of 0', () => {
      sessionStorage.setItem(
        'fin-report-agent:normal-task',
        JSON.stringify({ ticker: 'AAPL', taskId: 'saved-task-id', startedAt: Date.now() - 40_000 })
      )
      const { result } = renderHook(() => useAgentAnalysis())

      act(() => {
        result.current.resume()
      })

      expect(result.current.elapsedMs).toBe(40_000)
    })

    it('uses the server-reported elapsed_ms as the final duration, not the time since the task started', () => {
      // 30分钟前发起、只跑了2分钟就结束的task，刷新后接回
      sessionStorage.setItem(
        'fin-report-agent:normal-task',
        JSON.stringify({ ticker: 'AAPL', taskId: 'saved-task-id', startedAt: Date.now() - 30 * 60_000 })
      )
      const { result } = renderHook(() => useAgentAnalysis())
      act(() => {
        result.current.resume()
      })

      act(() => MockEventSource.latest().emit({ type: 'done', result: { final_report: 'ok' }, elapsed_ms: 120_000 }))

      expect(result.current.elapsedMs).toBe(120_000)
    })
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

  describe('cancel, retry and resumed events', () => {
    it('cancel() POSTs to the cancel endpoint of the current task', async () => {
      const { result } = renderHook(() => useAgentAnalysis())
      await startAnalysis(result, 'AAPL')

      await act(async () => {
        await result.current.cancel()
      })

      expect(global.fetch.mock.calls[1][0]).toBe('/api/analyze/tasks/mock-task-1/cancel')
      expect(global.fetch.mock.calls[1][1].method).toBe('POST')
    })

    it('a cancelled terminal event marks cancelled instead of a stream error', async () => {
      const { result } = renderHook(() => useAgentAnalysis())
      await startAnalysis(result, 'AAPL')

      act(() => MockEventSource.latest().emit({ type: 'error', code: 'cancelled', message: '分析已取消', elapsed_ms: 5000 }))

      expect(result.current.cancelled).toBe(true)
      expect(result.current.streamError).toBeNull()
      expect(result.current.done).toBe(true)
      expect(result.current.elapsedMs).toBe(5000)
    })

    it('exposes retryable only when the backend marks the failure as retryable', async () => {
      const { result } = renderHook(() => useAgentAnalysis())
      await startAnalysis(result, 'AAPL')

      act(() => MockEventSource.latest().emit({ type: 'error', message: '402 余额不足', retryable: true }))

      expect(result.current.retryable).toBe(true)
      expect(result.current.streamError).toBe('402 余额不足')
    })

    it('retry() POSTs the retry endpoint with the session header, then re-subscribes to the same task', async () => {
      const { result } = renderHook(() => useAgentAnalysis())
      await startAnalysis(result, 'AAPL')
      const first = MockEventSource.latest()
      act(() => first.emit({ type: 'error', message: '502', retryable: true }))

      await act(async () => {
        await result.current.retry()
      })

      const [url, init] = global.fetch.mock.calls[1]
      expect(url).toBe('/api/analyze/tasks/mock-task-1/retry')
      expect(init.headers['X-Session-Id']).toEqual(expect.any(String))
      expect(MockEventSource.instances).toHaveLength(2)
      expect(MockEventSource.latest().url).toBe(first.url)
      expect(result.current.streamError).toBeNull()
      expect(result.current.retryable).toBe(false)
      expect(result.current.done).toBe(false)
    })

    it('retry() surfaces the backend detail and drops the retry option when the retry is refused', async () => {
      const { result } = renderHook(() => useAgentAnalysis())
      await startAnalysis(result, 'AAPL')
      act(() => MockEventSource.latest().emit({ type: 'error', message: '502', retryable: true }))
      global.fetch.mockResolvedValueOnce({ ok: false, json: async () => ({ detail: '检查点已过期' }) })

      await act(async () => {
        await result.current.retry()
      })

      expect(result.current.streamError).toBe('检查点已过期')
      expect(result.current.retryable).toBe(false)
      expect(MockEventSource.instances).toHaveLength(1)
    })

    it('a resumed event clears leftover live text and adds a log entry marking where it continued', async () => {
      const { result } = renderHook(() => useAgentAnalysis())
      await startAnalysis(result, 'AAPL')
      const source = MockEventSource.latest()
      act(() => source.emit({ type: 'reasoning_delta', turn: 2, text: '半截' }))

      act(() => source.emit({ type: 'resumed', turn: 2, reason: 'server_restart', truncate_to: 5 }))

      expect(result.current.liveReasoning).toEqual({ turn: null, text: '' })
      expect(result.current.log.at(-1)).toMatchObject({ kind: 'resumed', turn: 2, reason: 'server_restart' })
    })
  })
})
