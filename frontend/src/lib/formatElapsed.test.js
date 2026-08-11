import { describe, expect, it } from 'vitest'
import { formatElapsed } from './formatElapsed'

describe('formatElapsed', () => {
  it('formats sub-minute durations with one decimal place of seconds', () => {
    expect(formatElapsed(0)).toBe('0.0秒')
    expect(formatElapsed(1500)).toBe('1.5秒')
    expect(formatElapsed(59900)).toBe('59.9秒')
  })

  it('formats durations at or over a minute as 分/秒', () => {
    expect(formatElapsed(60000)).toBe('1分0秒')
    expect(formatElapsed(65000)).toBe('1分5秒')
    expect(formatElapsed(185300)).toBe('3分5秒')
  })
})
