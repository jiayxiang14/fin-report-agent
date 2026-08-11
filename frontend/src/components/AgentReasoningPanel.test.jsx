import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
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
