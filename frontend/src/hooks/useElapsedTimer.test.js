import { act, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { useElapsedTimer } from './useElapsedTimer'

describe('useElapsedTimer', () => {
  beforeEach(() => {
    // performance.now() 也要跟着假时钟走，不然setInterval的回调被fake timer
    // 推进触发了，但内部用来算elapsedMs的performance.now()读到的还是真实
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
})
