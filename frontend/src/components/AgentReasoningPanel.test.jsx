import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import AgentReasoningPanel from './AgentReasoningPanel'

describe('AgentReasoningPanel elapsed time display', () => {
  it('shows a live-updating elapsed time next to the loading indicator while running', () => {
    render(<AgentReasoningPanel log={[]} result={null} streamError={null} done={false} elapsedMs={12300} />)
    expect(screen.getByText(/分析中/)).toHaveTextContent('分析中…（已用时 12.3秒）')
  })

  it('does not show an elapsed time note while running if elapsedMs is not provided', () => {
    render(<AgentReasoningPanel log={[]} result={null} streamError={null} done={false} elapsedMs={null} />)
    expect(screen.getByText('分析中…')).toBeInTheDocument()
  })

  it('shows the final elapsed time alongside the turns-used note once done', () => {
    const result = { turns_used: 4 }
    render(<AgentReasoningPanel log={[]} result={result} streamError={null} done={true} elapsedMs={23456} />)
    expect(screen.getByText(/本次分析用时/)).toHaveTextContent('本次分析用时 23.5秒，Agent自主决定跑了 4 轮工具调用/推理')
  })

  it('falls back to the turns-used note alone when elapsedMs is not provided', () => {
    const result = { turns_used: 4 }
    render(<AgentReasoningPanel log={[]} result={result} streamError={null} done={true} elapsedMs={null} />)
    const note = screen.getByText(/Agent自主决定跑了/)
    expect(note.textContent.startsWith('本次分析用时')).toBe(false)
  })

  it('formats a long-running analysis as 分/秒 instead of a big seconds number', () => {
    render(<AgentReasoningPanel log={[]} result={null} streamError={null} done={false} elapsedMs={125000} />)
    expect(screen.getByText(/分析中/)).toHaveTextContent('分析中…（已用时 2分5秒）')
  })
})

describe('AgentReasoningPanel token-level streaming and usage display', () => {
  it('shows the in-progress liveReasoning text while running', () => {
    render(
      <AgentReasoningPanel
        log={[]}
        result={null}
        streamError={null}
        done={false}
        elapsedMs={null}
        liveReasoning={{ turn: 0, text: '正在写到一半的句子' }}
      />
    )
    expect(screen.getByText('正在写到一半的句子')).toBeInTheDocument()
  })

  it('does not show liveReasoning text once done (the final report supersedes it)', () => {
    render(
      <AgentReasoningPanel
        log={[]}
        result={{ turns_used: 1 }}
        streamError={null}
        done={true}
        elapsedMs={null}
        liveReasoning={{ turn: 0, text: '不该再出现的半句话' }}
      />
    )
    expect(screen.queryByText('不该再出现的半句话')).not.toBeInTheDocument()
  })

  it('shows running token totals (no cost) next to the loading indicator', () => {
    render(
      <AgentReasoningPanel
        log={[]}
        result={null}
        streamError={null}
        done={false}
        elapsedMs={null}
        usageTotals={{ inputTokens: 1500, outputTokens: 250, cacheCreationInputTokens: 0, cacheReadInputTokens: 0 }}
      />
    )
    expect(screen.getByText(/已用 1,500 in \/ 250 out tokens/)).toBeInTheDocument()
  })

  it('shows the final token usage and estimated cost once done', () => {
    const result = {
      turns_used: 3,
      total_input_tokens: 12000,
      total_output_tokens: 1500,
      total_cache_read_input_tokens: 8000,
      estimated_cost_usd: 0.0567,
    }
    render(<AgentReasoningPanel log={[]} result={result} streamError={null} done={true} elapsedMs={null} />)
    expect(screen.getByText(/共消耗/)).toHaveTextContent(
      '共消耗 12,000 in / 1,500 out tokens（其中 8,000 命中缓存），预估花费 $0.0567'
    )
  })

  it('shows "cost unknown" instead of silently showing $0 when the model has no price table entry', () => {
    const result = {
      turns_used: 1,
      total_input_tokens: 100,
      total_output_tokens: 20,
      total_cache_read_input_tokens: 0,
      estimated_cost_usd: null,
    }
    render(<AgentReasoningPanel log={[]} result={result} streamError={null} done={true} elapsedMs={null} />)
    expect(screen.getByText(/共消耗/)).toHaveTextContent('预估成本未知（当前模型不在价格表里）')
  })
})

describe('AgentReasoningPanel task controls', () => {
  it('shows a stop button while running that calls onCancel', () => {
    const onCancel = vi.fn()
    render(<AgentReasoningPanel log={[]} result={null} streamError={null} done={false} onCancel={onCancel} />)

    fireEvent.click(screen.getByRole('button', { name: /停止分析/ }))

    expect(onCancel).toHaveBeenCalledTimes(1)
  })

  it('hides the stop button once done', () => {
    render(<AgentReasoningPanel log={[]} result={{ turns_used: 1 }} streamError={null} done={true} onCancel={vi.fn()} />)
    expect(screen.queryByRole('button', { name: /停止分析/ })).not.toBeInTheDocument()
  })

  it('offers a retry button only for retryable failures', () => {
    const onRetry = vi.fn()
    const { rerender } = render(
      <AgentReasoningPanel log={[]} result={null} streamError="502" retryable={false} done={true} onRetry={onRetry} />
    )
    expect(screen.queryByRole('button', { name: /从断点重试/ })).not.toBeInTheDocument()

    rerender(<AgentReasoningPanel log={[]} result={null} streamError="502" retryable={true} done={true} onRetry={onRetry} />)
    fireEvent.click(screen.getByRole('button', { name: /从断点重试/ }))

    expect(onRetry).toHaveBeenCalledTimes(1)
  })

  it('shows a neutral stopped note instead of a failure for a cancelled analysis', () => {
    render(<AgentReasoningPanel log={[]} result={null} streamError={null} cancelled={true} done={true} elapsedMs={8000} />)

    expect(screen.getByText(/已停止分析/)).toHaveTextContent('已停止分析（运行了 8.0秒）')
    expect(screen.queryByText(/分析失败/)).not.toBeInTheDocument()
  })

  it('renders a resumed log entry with the 1-indexed turn it continued from', () => {
    const log = [{ key: 1, kind: 'resumed', turn: 2, reason: 'server_restart' }]
    render(<AgentReasoningPanel log={log} result={null} streamError={null} done={false} />)

    expect(screen.getByText('服务重启后自动恢复，从第 3 轮继续')).toBeInTheDocument()
  })
})
