import { act, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { installMockEventSource, MockEventSource } from '../test/mockEventSource'
import { useAgentAnalysis } from './useAgentAnalysis'

describe('useAgentAnalysis', () => {
  beforeEach(() => {
    installMockEventSource()
  })

  afterEach(() => {
    MockEventSource.reset()
  })

  it('opens a connection to the right ticker-specific stream URL', () => {
    const { result } = renderHook(() => useAgentAnalysis())
    act(() => result.current.start('AAPL'))
    expect(MockEventSource.latest().url).toBe('/api/analyze/AAPL/stream')
  })

  it('accumulates reasoning/tool_call events into an ordered log with stable increasing keys', () => {
    const { result } = renderHook(() => useAgentAnalysis())
    act(() => result.current.start('AAPL'))
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

  it('sets result and done, and closes the connection, on a done event', () => {
    const { result } = renderHook(() => useAgentAnalysis())
    act(() => result.current.start('AAPL'))
    const source = MockEventSource.latest()

    const fakeResult = { ticker: 'AAPL', final_report: '<conclusion>买入</conclusion>' }
    act(() => source.emit({ type: 'done', result: fakeResult }))

    expect(result.current.result).toEqual(fakeResult)
    expect(result.current.done).toBe(true)
    expect(source.close).toHaveBeenCalledTimes(1)
  })

  it('surfaces a backend-sent error event as streamError and marks done', () => {
    const { result } = renderHook(() => useAgentAnalysis())
    act(() => result.current.start('AAPL'))
    const source = MockEventSource.latest()

    act(() => source.emit({ type: 'error', message: 'DeepSeek 余额不足' }))

    expect(result.current.streamError).toBe('DeepSeek 余额不足')
    expect(result.current.done).toBe(true)
    expect(source.close).toHaveBeenCalledTimes(1)
  })

  it('treats a raw connection failure (onerror, no prior done/error) as a stream error', () => {
    const { result } = renderHook(() => useAgentAnalysis())
    act(() => result.current.start('AAPL'))
    const source = MockEventSource.latest()

    act(() => source.triggerError())

    expect(result.current.done).toBe(true)
    expect(result.current.streamError).toBe('与服务器的连接中断')
  })

  it('does not overwrite an already-received backend error message with the generic onerror message', () => {
    // 真实场景：后端先发了明确的error事件（比如"402余额不足"），浏览器紧接着
    // 因为连接关闭也触发一次onerror——不能让后面这个通用消息把前面具体的原因覆盖掉
    const { result } = renderHook(() => useAgentAnalysis())
    act(() => result.current.start('AAPL'))
    const source = MockEventSource.latest()

    act(() => source.emit({ type: 'error', message: 'DeepSeek 余额不足' }))
    act(() => source.triggerError())

    expect(result.current.streamError).toBe('DeepSeek 余额不足')
  })

  it('closes the previous connection and resets state when start is called again for a new ticker', () => {
    const { result } = renderHook(() => useAgentAnalysis())
    act(() => result.current.start('AAPL'))
    const firstSource = MockEventSource.latest()
    act(() => firstSource.emit({ type: 'reasoning', turn: 1, text: 'x' }))
    expect(result.current.log).toHaveLength(1)

    act(() => result.current.start('MSFT'))

    expect(firstSource.close).toHaveBeenCalledTimes(1)
    expect(MockEventSource.latest().url).toBe('/api/analyze/MSFT/stream')
    expect(result.current.log).toEqual([]) // 旧ticker的日志不该串到新一轮分析里
    expect(result.current.done).toBe(false)
  })
})
