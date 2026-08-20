import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import ReportPanel from './ReportPanel'

function resultWith(finalReport, sentiment) {
  return {
    completed: true,
    stop_reason: 'end_turn',
    turns_used: 3,
    final_report:
      `数据截至2026-08-01。\n<conclusion>${finalReport}</conclusion>` +
      `<evidence>营收同比增长15%，达到416.16B</evidence><flags>无明显异常</flags>` +
      (sentiment ? `<sentiment>${sentiment}</sentiment>` : ''),
  }
}

describe('ReportPanel', () => {
  it('renders a green conclusion box when sentiment is positive', () => {
    const { container } = render(<ReportPanel result={resultWith('公司表现强劲', 'positive')} />)
    const conclusionBox = screen.getByText('结论').closest('div.rounded-lg')
    expect(conclusionBox.className).toContain('border-emerald-500')
    expect(container.textContent).toContain('公司表现强劲')
  })

  it('renders a red conclusion box when sentiment is negative', () => {
    render(<ReportPanel result={resultWith('业绩明显恶化', 'negative')} />)
    const conclusionBox = screen.getByText('结论').closest('div.rounded-lg')
    expect(conclusionBox.className).toContain('border-rose-500')
  })

  it('falls back to the neutral violet theme when the model did not emit a sentiment tag', () => {
    render(<ReportPanel result={resultWith('喜忧参半', null)} />)
    const conclusionBox = screen.getByText('结论').closest('div.rounded-lg')
    expect(conclusionBox.className).toContain('border-violet-500')
  })

  it('highlights numeric tokens but not 4-digit years', () => {
    render(<ReportPanel result={resultWith('结论', 'neutral')} />)
    // "15%"和"416.16B"应该被包进带底色的<span>里
    expect(screen.getByText('15%').tagName).toBe('SPAN')
    expect(screen.getByText('416.16B').tagName).toBe('SPAN')
  })

  it('does not split the digit out of Q1/Q2/FY2025-style letter+number labels', () => {
    const result = {
      completed: true,
      stop_reason: 'end_turn',
      turns_used: 3,
      final_report:
        '<conclusion>结论</conclusion>' +
        '<evidence>e</evidence>' +
        '<flags>FY2025 Q1和Q2营收环比增长8%，H1整体符合预期</flags>',
    }
    const { container } = render(<ReportPanel result={result} />)

    // "8%"应该被高亮，但"Q1"/"Q2"/"FY2025"/"H1"不应该被拆开——字母后面紧跟的数字
    // 不应该单独出现在<span>里，只有"8%"这一个高亮span
    const highlighted = [...container.querySelectorAll('span.font-semibold')].map((el) => el.textContent)
    expect(highlighted).toEqual(['8%'])
    expect(container.textContent).toContain('FY2025 Q1和Q2营收环比增长8%，H1整体符合预期')
  })

  it('does not split the leading digits out of 10-K/10-Q SEC filing type references', () => {
    const result = {
      completed: true,
      stop_reason: 'end_turn',
      turns_used: 3,
      final_report:
        '<conclusion>结论</conclusion>' +
        '<evidence>e</evidence>' +
        '<flags>根据最新10-K和10-Q披露，营收增长12%</flags>',
    }
    const { container } = render(<ReportPanel result={result} />)

    const highlighted = [...container.querySelectorAll('span.font-semibold')].map((el) => el.textContent)
    expect(highlighted).toEqual(['12%'])
    expect(container.textContent).toContain('根据最新10-K和10-Q披露，营收增长12%')
  })

  it('does not highlight the month/day numbers in Chinese-style or ISO dates', () => {
    const result = {
      completed: true,
      stop_reason: 'end_turn',
      turns_used: 3,
      final_report:
        '<conclusion>结论</conclusion>' +
        '<evidence>e</evidence>' +
        '<flags>公司已于2026-08-01（即2026年8月7日前）发布财报，净利润增长20%</flags>',
    }
    const { container } = render(<ReportPanel result={result} />)

    // "2026-08-01"（ISO日期）和"2026年8月7日"（中文日期）里的年/月/日
    // 都不应该被拆出任何数字高亮，只有"20%"应该被高亮
    const highlighted = [...container.querySelectorAll('span.font-semibold')].map((el) => el.textContent)
    expect(highlighted).toEqual(['20%'])
    expect(container.textContent).toContain('公司已于2026-08-01（即2026年8月7日前）发布财报，净利润增长20%')
  })

  it('does not highlight bare counts like "5天"/"12月" that carry no financial marker', () => {
    const result = {
      completed: true,
      stop_reason: 'end_turn',
      turns_used: 3,
      final_report:
        '<conclusion>结论</conclusion>' +
        '<evidence>e</evidence>' +
        '<flags>反应窗口后交易日不足5天，预计12月完成，交易金额约1亿美元，成交量为20日均量的3.43倍</flags>',
    }
    const { container } = render(<ReportPanel result={result} />)

    // "5"（不足5天）、"12"（12月）都是没有任何金额/百分比/量级标记的裸数字，
    // 不应该被高亮；"1亿"（带量级后缀）和"3.43"（带小数点）应该被高亮
    const highlighted = [...container.querySelectorAll('span.font-semibold')].map((el) => el.textContent)
    expect(highlighted).toEqual(['1亿', '3.43'])
    expect(container.textContent).toContain('反应窗口后交易日不足5天，预计12月完成，交易金额约1亿美元，成交量为20日均量的3.43倍')
  })

  it('highlights comma-grouped thousands numbers without requiring a % or magnitude suffix', () => {
    const result = {
      completed: true,
      stop_reason: 'end_turn',
      turns_used: 3,
      final_report:
        '<conclusion>结论</conclusion>' +
        '<evidence>商业剩余履约义务 $6,780亿，另有1,337名新增员工</evidence>' +
        '<flags>f</flags>',
    }
    const { container } = render(<ReportPanel result={result} />)

    const highlighted = [...container.querySelectorAll('span.font-semibold')].map((el) => el.textContent)
    expect(highlighted).toEqual(['$6,780亿', '1,337'])
  })

  it('treats a sentiment tag with trailing punctuation as its matched keyword, not neutral fallback', () => {
    const result = {
      completed: true,
      stop_reason: 'end_turn',
      turns_used: 3,
      final_report:
        '<conclusion>业绩强劲</conclusion><evidence>e</evidence><flags>f</flags><sentiment>positive。</sentiment>',
    }
    render(<ReportPanel result={result} />)
    const conclusionBox = screen.getByText('结论').closest('div.rounded-lg')
    expect(conclusionBox.className).toContain('border-emerald-500')
  })

  it('shows the self-verification badge when the backend reports it was triggered', () => {
    render(<ReportPanel result={{ ...resultWith('结论', 'neutral'), self_verification_triggered: true }} />)
    expect(screen.getByText('已完成自我核查')).toBeInTheDocument()
  })

  it('shows a muted note when self-verification was not triggered', () => {
    render(<ReportPanel result={{ ...resultWith('结论', 'neutral'), self_verification_triggered: false }} />)
    expect(screen.getByText('本次未做自我核查')).toBeInTheDocument()
  })

  it('shows the verification-mismatch badge only when it was actually triggered', () => {
    const { rerender } = render(
      <ReportPanel
        result={{
          ...resultWith('结论', 'neutral'),
          verification_mismatch_triggered: true,
          verification_mismatch_resolved: true,
        }}
      />
    )
    expect(screen.getByText('核查发现数字不一致后已修正')).toBeInTheDocument()

    rerender(
      <ReportPanel result={{ ...resultWith('结论', 'neutral'), verification_mismatch_triggered: false }} />
    )
    expect(screen.queryByText('核查发现数字不一致后已修正')).not.toBeInTheDocument()
  })

  it('shows an amber warning instead when verification-mismatch triggered but was never confirmed resolved', () => {
    render(
      <ReportPanel
        result={{
          ...resultWith('结论', 'neutral'),
          verification_mismatch_triggered: true,
          verification_mismatch_resolved: false,
        }}
      />
    )
    expect(screen.getByText('核查发现数字不一致，未能确认已修正')).toBeInTheDocument()
    expect(screen.queryByText('核查发现数字不一致后已修正')).not.toBeInTheDocument()
  })

  it('shows the structure-gate badge only when it was actually triggered', () => {
    const { rerender } = render(
      <ReportPanel
        result={{ ...resultWith('结论', 'neutral'), structure_gate_triggered: true, structure_gate_resolved: true }}
      />
    )
    expect(screen.getByText('结构缺失后已补全')).toBeInTheDocument()

    rerender(<ReportPanel result={{ ...resultWith('结论', 'neutral'), structure_gate_triggered: false }} />)
    expect(screen.queryByText('结构缺失后已补全')).not.toBeInTheDocument()
  })

  it('shows an amber warning instead when structure gate triggered but was never confirmed resolved', () => {
    render(
      <ReportPanel
        result={{ ...resultWith('结论', 'neutral'), structure_gate_triggered: true, structure_gate_resolved: false }}
      />
    )
    expect(screen.getByText('检测到结构缺失，未能确认已补全')).toBeInTheDocument()
    expect(screen.queryByText('结构缺失后已补全')).not.toBeInTheDocument()
  })

  it('shows the tool-coverage-gate badge only when it was actually triggered', () => {
    const { rerender } = render(
      <ReportPanel
        result={{
          ...resultWith('结论', 'neutral'),
          tool_coverage_gate_triggered: true,
          tool_coverage_gate_resolved: true,
        }}
      />
    )
    expect(screen.getByText('补查基础数据后已定稿')).toBeInTheDocument()

    rerender(<ReportPanel result={{ ...resultWith('结论', 'neutral'), tool_coverage_gate_triggered: false }} />)
    expect(screen.queryByText('补查基础数据后已定稿')).not.toBeInTheDocument()
  })

  it('shows an amber warning instead when tool-coverage gate triggered but was never confirmed resolved', () => {
    render(
      <ReportPanel
        result={{
          ...resultWith('结论', 'neutral'),
          tool_coverage_gate_triggered: true,
          tool_coverage_gate_resolved: false,
        }}
      />
    )
    expect(screen.getByText('检测到未查基础数据，未能确认已补查')).toBeInTheDocument()
    expect(screen.queryByText('补查基础数据后已定稿')).not.toBeInTheDocument()
  })

  it('shows the sentiment-consistency-gate badge only when it was actually triggered', () => {
    const { rerender } = render(
      <ReportPanel
        result={{
          ...resultWith('结论', 'neutral'),
          sentiment_consistency_gate_triggered: true,
          sentiment_consistency_gate_resolved: true,
        }}
      />
    )
    expect(screen.getByText('情绪判断与市场反应有落差已要求说明')).toBeInTheDocument()

    rerender(
      <ReportPanel result={{ ...resultWith('结论', 'neutral'), sentiment_consistency_gate_triggered: false }} />
    )
    expect(screen.queryByText('情绪判断与市场反应有落差已要求说明')).not.toBeInTheDocument()
  })

  it('shows an amber warning instead when sentiment-consistency gate triggered but was never confirmed resolved', () => {
    render(
      <ReportPanel
        result={{
          ...resultWith('结论', 'neutral'),
          sentiment_consistency_gate_triggered: true,
          sentiment_consistency_gate_resolved: false,
        }}
      />
    )
    expect(screen.getByText('情绪判断与市场反应有落差，要求说明后仍未处理')).toBeInTheDocument()
    expect(screen.queryByText('情绪判断与市场反应有落差已要求说明')).not.toBeInTheDocument()
  })

  it('shows the traceability-gate badge only when it was actually triggered', () => {
    const { rerender } = render(
      <ReportPanel
        result={{
          ...resultWith('结论', 'neutral'),
          traceability_gate_triggered: true,
          traceability_gate_resolved: true,
        }}
      />
    )
    expect(screen.getByText('数字可追溯率偏低已要求补充说明')).toBeInTheDocument()

    rerender(
      <ReportPanel result={{ ...resultWith('结论', 'neutral'), traceability_gate_triggered: false }} />
    )
    expect(screen.queryByText('数字可追溯率偏低已要求补充说明')).not.toBeInTheDocument()
  })

  it('shows an amber warning instead when traceability gate triggered but was never confirmed resolved', () => {
    render(
      <ReportPanel
        result={{
          ...resultWith('结论', 'neutral'),
          traceability_gate_triggered: true,
          traceability_gate_resolved: false,
        }}
      />
    )
    expect(screen.getByText('数字可追溯率偏低，要求补充说明后仍未改善')).toBeInTheDocument()
    expect(screen.queryByText('数字可追溯率偏低已要求补充说明')).not.toBeInTheDocument()
  })

  it('shows the reflexion badge only when it was actually triggered', () => {
    const { rerender } = render(
      <ReportPanel result={{ ...resultWith('结论', 'neutral'), reflexion_triggered: true }} />
    )
    expect(screen.getByText('过程裁判反馈后已整改')).toBeInTheDocument()

    rerender(<ReportPanel result={{ ...resultWith('结论', 'neutral'), reflexion_triggered: false }} />)
    expect(screen.queryByText('过程裁判反馈后已整改')).not.toBeInTheDocument()
  })
})
