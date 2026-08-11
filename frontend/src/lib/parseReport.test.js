import { describe, expect, it } from 'vitest'
import { parseReport } from './parseReport'

describe('parseReport', () => {
  it('returns empty structure for falsy input', () => {
    expect(parseReport('')).toEqual({ preamble: '', sections: {}, hasAnyTag: false })
    expect(parseReport(null)).toEqual({ preamble: '', sections: {}, hasAnyTag: false })
    expect(parseReport(undefined)).toEqual({ preamble: '', sections: {}, hasAnyTag: false })
  })

  it('extracts all three sections when the report follows the standard三段式 format', () => {
    const raw =
      '数据截至2026-08-01。\n' +
      '<conclusion>买入</conclusion>\n' +
      '<evidence>营收同比增长15%</evidence>\n' +
      '<flags>存货周转天数上升</flags>'

    const result = parseReport(raw)

    expect(result.hasAnyTag).toBe(true)
    expect(result.sections.conclusion).toBe('买入')
    expect(result.sections.evidence).toBe('营收同比增长15%')
    expect(result.sections.flags).toBe('存货周转天数上升')
    expect(result.preamble).toBe('数据截至2026-08-01。')
  })

  it('keeps preamble empty when <conclusion> is the very first thing in the text', () => {
    const raw = '<conclusion>买入</conclusion><evidence>e</evidence><flags>f</flags>'
    expect(parseReport(raw).preamble).toBe('')
  })

  it('handles partial-tag reports (model only wrote some sections) without crashing', () => {
    const raw = '<conclusion>买入</conclusion>'
    const result = parseReport(raw)
    expect(result.hasAnyTag).toBe(true)
    expect(result.sections).toEqual({ conclusion: '买入' })
    expect(result.sections.evidence).toBeUndefined()
  })

  it('reports hasAnyTag false when the model refused/truncated and produced no tags at all', () => {
    const raw = '很抱歉，我无法完成这项分析。'
    const result = parseReport(raw)
    expect(result.hasAnyTag).toBe(false)
    expect(result.sections).toEqual({})
  })

  it('trims whitespace inside each captured section', () => {
    const raw = '<conclusion>\n  买入  \n</conclusion>'
    expect(parseReport(raw).sections.conclusion).toBe('买入')
  })

  it('extracts the sentiment tag when present after </flags>', () => {
    const raw =
      '<conclusion>买入</conclusion><evidence>e</evidence><flags>f</flags><sentiment>positive</sentiment>'
    expect(parseReport(raw).sections.sentiment).toBe('positive')
  })

  it('leaves sentiment undefined when the model did not write the tag (backward compatible)', () => {
    const raw = '<conclusion>买入</conclusion><evidence>e</evidence><flags>f</flags>'
    expect(parseReport(raw).sections.sentiment).toBeUndefined()
  })

  it('uses non-greedy matching so multiple tags of the same kind do not collapse into one', () => {
    // 模型不应该重复输出标签，但如果发生了，非贪婪匹配应该只抓第一段，
    // 不应该从第一个<evidence>一路匹配到最后一个</evidence>
    const raw = '<evidence>第一段</evidence>无关文字<evidence>第二段</evidence>'
    expect(parseReport(raw).sections.evidence).toBe('第一段')
  })
})
