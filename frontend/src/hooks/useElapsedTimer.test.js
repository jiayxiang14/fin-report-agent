import { act, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { useElapsedTimer } from './useElapsedTimer'

describe('useElapsedTimer', () => {
  beforeEach(() => {
    // Date.now() 也要跟着假时钟走，不然setInterval的回调被fake timer
    // 推进触发了，但内部用来算elapsedMs的Date.now()读到的还是真实
    // wall-clock时间，两者对不上，断言会不稳定
    vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval', 'Date', 'performance'] })
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('starts at 0 before start() is called', () => {
    const { result } = renderHook(() => useElapsedTimer())
    expect(result.current.elapsedMs).toBe(0)
  })

  it('elapsedMs increases as time passes after start()', () => {
    const { result } = renderHook(() => useElapsedTimer())

    act(() => result.current.start())
    act(() => vi.advanceTimersByTime(600))

    expect(result.current.elapsedMs).toBeGreaterThanOrEqual(600)
  })

  it('stop() freezes elapsedMs instead of resetting or continuing to tick', () => {
    const { result } = renderHook(() => useElapsedTimer())

    act(() => result.current.start())
    act(() => vi.advanceTimersByTime(1000))
    act(() => result.current.stop())

    const frozen = result.current.elapsedMs
    act(() => vi.advanceTimersByTime(1000))

    expect(result.current.elapsedMs).toBe(frozen)
  })

  it('calling start() again resets elapsedMs back to 0', () => {
    const { result } = renderHook(() => useElapsedTimer())

    act(() => result.current.start())
    act(() => vi.advanceTimersByTime(1000))
    act(() => result.current.stop())

    act(() => result.current.start())

    expect(result.current.elapsedMs).toBe(0)
  })

  it('start(startedAtEpochMs) resumes counting from an earlier start time instead of 0', () => {
    const { result } = renderHook(() => useElapsedTimer())

    act(() => result.current.start(Date.now() - 5000))
    expect(result.current.elapsedMs).toBe(5000)

    act(() => vi.advanceTimersByTime(600))
    expect(result.current.elapsedMs).toBeGreaterThanOrEqual(5600)
  })

  it('start() ignores a start time in the future and counts from now', () => {
    const { result } = renderHook(() => useElapsedTimer())

    act(() => result.current.start(Date.now() + 60_000))

    expect(result.current.elapsedMs).toBe(0)
  })

  it('stop(finalMs) uses the authoritative value instead of the locally measured one', () => {
    const { result } = renderHook(() => useElapsedTimer())

    // 模拟刷新页面接回一个30分钟前发起、只跑了2分钟的task
    act(() => result.current.start(Date.now() - 30 * 60_000))
    act(() => result.current.stop(120_000))

    expect(result.current.elapsedMs).toBe(120_000)
  })
})
