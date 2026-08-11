import { act, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'
import { installMockEventSource, MockEventSource } from '../test/mockEventSource'
import { useBestOfNAnalysis } from './useBestOfNAnalysis'

describe('useBestOfNAnalysis', () => {
  beforeEach(() => {
    installMockEventSource()
  })

  afterEach(() => {
    MockEventSource.reset()
  })

  it('opens a connection to the best-of-n specific stream URL', () => {
    const { result } = renderHook(() => useBestOfNAnalysis())
    act(() => result.current.start('AMZN'))
    expect(MockEventSource.latest().url).toBe('/api/analyze/AMZN/best-of-n/stream')
  })

  it('maps trajectory_score/trajectory_reason from the candidate_scored event', () => {
    const { result } = renderHook(() => useBestOfNAnalysis())
    act(() => result.current.start('AMZN'))
    const source = MockEventSource.latest()

    act(() =>
      source.emit({
        type: 'candidate_scored',
        candidate_index: 0,
        temperature: 0.3,
        total_score: 80.15,
        rule_score: {},
        llm_score: 90,
        llm_reason: '逻辑连贯',
        trajectory_score: 75,
        trajectory_reason: '信息收集充分',
      })
    )

    expect(result.current.candidates[0]).toMatchObject({
      trajectoryScore: 75,
      trajectoryReason: '信息收集充分',
    })
  })

  it('maps reflexion_triggered from the candidate_scored event', () => {
    const { result } = renderHook(() => useBestOfNAnalysis())
    act(() => result.current.start('AMZN'))
    const source = MockEventSource.latest()

    act(() =>
      source.emit({
        type: 'candidate_scored',
        candidate_index: 0,
        temperature: 0.3,
        total_score: 80.15,
        rule_score: {},
        llm_score: 90,
        llm_reason: '逻辑连贯',
        trajectory_score: 60,
        trajectory_reason: '信息收集不充分',
        reflexion_triggered: true,
      })
    )

    expect(result.current.candidates[0]).toMatchObject({ reflexionTriggered: true })
  })

  it('appends a candidate_scored event to candidates without a report body yet', () => {
    const { result } = renderHook(() => useBestOfNAnalysis())
    act(() => result.current.start('AMZN'))
    const source = MockEventSource.latest()

    act(() =>
      source.emit({
        type: 'candidate_scored',
        candidate_index: 0,
        temperature: 0.3,
        total_score: 80.15,
        rule_score: { traceability: 40, self_verification: 20, structure: 20, length: 10 },
        llm_score: 90,
        llm_reason: '逻辑连贯',
      })
    )

    expect(result.current.candidates).toHaveLength(1)
    expect(result.current.candidates[0]).toMatchObject({
      index: 0,
      totalScore: 80.15,
      finalReport: null, // 打分完成时后端还没带正文，正文要等done事件才补齐
    })
  })

  it('records a candidate_failed candidate with its error and no score, without dropping other candidates', () => {
    const { result } = renderHook(() => useBestOfNAnalysis())
    act(() => result.current.start('AMZN'))
    const source = MockEventSource.latest()

    act(() =>
      source.emit({
        type: 'candidate_scored',
        candidate_index: 0,
        temperature: 0.3,
        total_score: 75.0,
        rule_score: {},
        llm_score: null,
        llm_reason: null,
      })
    )
    act(() =>
      source.emit({
        type: 'candidate_failed',
        candidate_index: 1,
        temperature: 0.6,
        error: 'Insufficient Balance',
      })
    )

    expect(result.current.candidates).toHaveLength(2)
    expect(result.current.candidates[1]).toMatchObject({
      index: 1,
      totalScore: null,
      error: 'Insufficient Balance',
    })
    // 候选0之前已经打完分的结果不该被候选1的失败影响
    expect(result.current.candidates[0].totalScore).toBe(75.0)
  })

  it('buckets reasoning/tool events per candidate_index so they do not mix across candidates', () => {
    const { result } = renderHook(() => useBestOfNAnalysis())
    act(() => result.current.start('AMZN'))
    const source = MockEventSource.latest()

    act(() => source.emit({ type: 'reasoning', candidate_index: 0, turn: 1, text: '候选0的思考' }))
    act(() => source.emit({ type: 'reasoning', candidate_index: 1, turn: 1, text: '候选1的思考' }))
    act(() =>
      source.emit({
        type: 'done',
        result: {
          selected_index: 0,
          candidates: [
            { index: 0, temperature: 0.3, total_score: 80, rule_score: {}, llm_score: 90, llm_reason: 'ok', final_report: 'R0', error: null },
            { index: 1, temperature: 0.6, total_score: 70, rule_score: {}, llm_score: 80, llm_reason: 'ok', final_report: 'R1', error: null },
          ],
        },
      })
    )

    // done事件到达后，selectedLog应该只挑出被选中候选(index 0)自己的日志，
    // 不是候选1的、也不是两者混在一起
    expect(result.current.selectedLog).toHaveLength(1)
    expect(result.current.selectedLog[0].text).toBe('候选0的思考')
  })

  it('backfills full report text and authoritative scores from the done event, overwriting the running-state candidates list', () => {
    const { result } = renderHook(() => useBestOfNAnalysis())
    act(() => result.current.start('AMZN'))
    const source = MockEventSource.latest()

    act(() =>
      source.emit({
        type: 'candidate_scored',
        candidate_index: 0,
        temperature: 0.3,
        total_score: 80,
        rule_score: {},
        llm_score: 90,
        llm_reason: 'ok',
      })
    )
    expect(result.current.candidates[0].finalReport).toBeNull()

    act(() =>
      source.emit({
        type: 'done',
        result: {
          selected_index: 0,
          candidates: [
            {
              index: 0,
              temperature: 0.3,
              total_score: 80,
              rule_score: {},
              llm_score: 90,
              llm_reason: 'ok',
              trajectory_score: 70,
              trajectory_reason: '过程还行',
              final_report: '<conclusion>买入</conclusion>',
              error: null,
            },
          ],
        },
      })
    )

    expect(result.current.candidates[0].finalReport).toBe('<conclusion>买入</conclusion>')
    expect(result.current.candidates[0].trajectoryScore).toBe(70)
    expect(result.current.candidates[0].trajectoryReason).toBe('过程还行')
    expect(result.current.done).toBe(true)
    expect(source.close).toHaveBeenCalledTimes(1)
  })

  it('falls back to an empty selectedLog when the selected candidate never logged any events', () => {
    const { result } = renderHook(() => useBestOfNAnalysis())
    act(() => result.current.start('AMZN'))
    const source = MockEventSource.latest()

    act(() =>
      source.emit({
        type: 'done',
        result: {
          selected_index: 2,
          candidates: [{ index: 2, temperature: 1.0, total_score: 60, rule_score: {}, llm_score: null, llm_reason: null, final_report: 'R', error: null }],
        },
      })
    )

    expect(result.current.selectedLog).toEqual([])
  })

  it('resets candidates/selectedLog/result when start is called again', () => {
    const { result } = renderHook(() => useBestOfNAnalysis())
    act(() => result.current.start('AMZN'))
    const firstSource = MockEventSource.latest()
    act(() => firstSource.emit({ type: 'candidate_scored', candidate_index: 0, temperature: 0.3, total_score: 80, rule_score: {}, llm_score: null, llm_reason: null }))
    expect(result.current.candidates).toHaveLength(1)

    act(() => result.current.start('KO'))

    expect(firstSource.close).toHaveBeenCalledTimes(1)
    expect(result.current.candidates).toEqual([])
    expect(result.current.selectedLog).toEqual([])
    expect(result.current.result).toBeNull()
  })
})
