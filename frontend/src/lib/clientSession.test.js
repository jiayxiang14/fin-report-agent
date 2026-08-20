import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { getSessionId } from './clientSession'

describe('getSessionId', () => {
  beforeEach(() => {
    sessionStorage.clear()
  })

  afterEach(() => {
    sessionStorage.clear()
  })

  it('generates a session id on first call and persists it in sessionStorage', () => {
    const sessionId = getSessionId()

    expect(sessionId).toEqual(expect.any(String))
    expect(sessionStorage.getItem('fin-report-agent:session-id')).toBe(sessionId)
  })

  it('returns the same session id on repeated calls within the same tab', () => {
    const first = getSessionId()
    const second = getSessionId()

    expect(second).toBe(first)
  })

  it('reads back a session id already saved by a previous call', () => {
    const saved = getSessionId()

    // 模拟同一个标签页里重新渲染/新的hook实例——不该生成一个新的
    const readBack = getSessionId()

    expect(readBack).toBe(saved)
  })
})
