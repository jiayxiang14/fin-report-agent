import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import CandidateComparisonPanel from './CandidateComparisonPanel'

function candidate(overrides = {}) {
  return {
    index: 0,
    temperature: 0.3,
    totalScore: 80.15,
    ruleScore: { traceability: 40, self_verification: 20, structure: 20, length: 10 },
    llmScore: 90,
    llmReason: '逻辑连贯',
    trajectoryScore: 75,
    trajectoryReason: '信息收集充分',
    reflexionTriggered: false,
    finalReport: '<conclusion>买入</conclusion>',
    error: null,
    ...overrides,
  }
}

describe('CandidateComparisonPanel', () => {
  it('shows the first slot as running and the rest as pending before any candidate has resolved', () => {
    render(<CandidateComparisonPanel candidates={[]} totalCount={3} result={null} done={false} streamError={null} />)

    expect(screen.getByText('生成中…')).toBeInTheDocument()
    expect(screen.getAllByText('等待中…')).toHaveLength(2)
  })

  it('marks the next unresolved slot as running while earlier ones show their scores', () => {
    render(
      <CandidateComparisonPanel
        candidates={[candidate({ index: 0 })]}
        totalCount={3}
        result={null}
        done={false}
        streamError={null}
      />
    )

    expect(screen.getByText('80.2')).toBeInTheDocument() // 候选0已经出分
    expect(screen.getByText('生成中…')).toBeInTheDocument() // 候选1在跑
    expect(screen.getByText('等待中…')).toBeInTheDocument() // 候选2还没轮到
  })

  it('badges the selected candidate once the done result names its index', () => {
    render(
      <CandidateComparisonPanel
        candidates={[candidate({ index: 0 }), candidate({ index: 1, totalScore: 70 })]}
        totalCount={2}
        result={{ selected_index: 1 }}
        done={true}
        streamError={null}
      />
    )

    expect(screen.getByText('已选中')).toBeInTheDocument()
    // 候选1（索引1，展示成"候选 2"）应该是被选中的那个
    expect(screen.getByText('候选 2').closest('div.rounded-lg')).toHaveTextContent('已选中')
  })

  it('renders a failed candidate with its error message and no score, without hiding the other candidates', () => {
    render(
      <CandidateComparisonPanel
        candidates={[candidate({ index: 0, error: 'Insufficient Balance', totalScore: null })]}
        totalCount={1}
        result={null}
        done={true}
        streamError={null}
      />
    )

    expect(screen.getByText(/这个候选运行失败/)).toBeInTheDocument()
    expect(screen.getByText(/Insufficient Balance/)).toBeInTheDocument()
    expect(screen.queryByText('80.2')).not.toBeInTheDocument()
  })

  it('toggles the full report text on click without showing it by default', async () => {
    render(
      <CandidateComparisonPanel
        candidates={[candidate()]}
        totalCount={1}
        result={{ selected_index: 0 }}
        done={true}
        streamError={null}
      />
    )

    expect(screen.queryByText('<conclusion>买入</conclusion>')).not.toBeInTheDocument()

    await userEvent.click(screen.getByText('查看这份候选的完整简报'))
    expect(screen.getByText('<conclusion>买入</conclusion>')).toBeInTheDocument()

    await userEvent.click(screen.getByText('收起正文'))
    expect(screen.queryByText('<conclusion>买入</conclusion>')).not.toBeInTheDocument()
  })

  it('shows the outcome judge as "未打分" when that judge call failed', () => {
    render(
      <CandidateComparisonPanel
        candidates={[candidate({ llmScore: null, llmReason: null })]}
        totalCount={1}
        result={null}
        done={false}
        streamError={null}
      />
    )

    expect(screen.getByText('未打分')).toBeInTheDocument()
  })

  it('shows the trajectory judge score and reason separately from the outcome judge', () => {
    render(
      <CandidateComparisonPanel
        candidates={[candidate({ trajectoryScore: 82, trajectoryReason: '决策路径高效' })]}
        totalCount={1}
        result={null}
        done={false}
        streamError={null}
      />
    )

    expect(screen.getByText('82.0分')).toBeInTheDocument()
    expect(screen.getByText('决策路径高效')).toBeInTheDocument()
  })

  it('shows the trajectory judge as "未打分" when that judge call failed', () => {
    render(
      <CandidateComparisonPanel
        candidates={[candidate({ trajectoryScore: null, trajectoryReason: null })]}
        totalCount={1}
        result={null}
        done={false}
        streamError={null}
      />
    )

    expect(screen.getByText('未打分')).toBeInTheDocument()
  })

  it('shows a "已整改" badge only when reflexion actually fired for that candidate', () => {
    const { rerender } = render(
      <CandidateComparisonPanel
        candidates={[candidate({ reflexionTriggered: true })]}
        totalCount={1}
        result={null}
        done={false}
        streamError={null}
      />
    )
    expect(screen.getByText('已整改')).toBeInTheDocument()

    rerender(
      <CandidateComparisonPanel
        candidates={[candidate({ reflexionTriggered: false })]}
        totalCount={1}
        result={null}
        done={false}
        streamError={null}
      />
    )
    expect(screen.queryByText('已整改')).not.toBeInTheDocument()
  })

  it('renders the stream-level error banner separately from per-candidate failures', () => {
    render(
      <CandidateComparisonPanel candidates={[]} totalCount={3} result={null} done={true} streamError="与服务器的连接中断" />
    )

    expect(screen.getByText(/深度分析失败：与服务器的连接中断/)).toBeInTheDocument()
  })
})
