import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import ThematicFlowPanel from './ThematicFlowPanel'

function jsonResponse(body) {
  return Promise.resolve({
    ok: true,
    json: () => Promise.resolve(body),
  })
}

function theme(overrides = {}) {
  return {
    theme_name: '半导体',
    chain_position: '上游',
    constituent_tickers: ['SOXX'],
    rs_ratio: 101.2,
    rs_momentum: 100.5,
    quadrant: 'leading',
    history: [],
    ...overrides,
  }
}

describe('ThematicFlowPanel', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn())
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('fetches the ticker-scoped endpoint when a ticker is provided', async () => {
    fetch.mockReturnValue(
      jsonResponse({
        ticker: 'NVDA',
        matched_themes: [],
        sic_matched_themes: [],
        benchmark: 'SPY',
        as_of: '2026-08-05',
        themes: [],
        note: '',
      })
    )

    render(<ThematicFlowPanel ticker="NVDA" />)

    await waitFor(() => expect(fetch).toHaveBeenCalledWith('/api/thematic-flow?ticker=NVDA'))
  })

  it('fetches the unscoped endpoint when no ticker is provided', async () => {
    fetch.mockReturnValue(
      jsonResponse({
        ticker: null,
        matched_themes: [],
        sic_matched_themes: [],
        benchmark: 'SPY',
        as_of: '2026-08-05',
        themes: [theme()],
        note: '',
      })
    )

    render(<ThematicFlowPanel ticker={null} />)

    await waitFor(() => expect(fetch).toHaveBeenCalledWith('/api/thematic-flow'))
    // 没有ticker就不存在"无关"这个判断，应该直接展示网格，不出现收起提示
    expect(await screen.findByText('半导体')).toBeInTheDocument()
  })

  it('collapses into a one-line prompt when the ticker matches none of the themes', async () => {
    fetch.mockReturnValue(
      jsonResponse({
        ticker: 'TWST',
        matched_themes: [],
        sic_matched_themes: [],
        benchmark: 'SPY',
        as_of: '2026-08-05',
        themes: [theme()],
        note: '',
      })
    )

    render(<ThematicFlowPanel ticker="TWST" />)

    expect(await screen.findByText(/不在预设的AI算力产业链主题库里/)).toBeInTheDocument()
    // 不该强行铺满卡片制造"看起来都相关"的错觉——这是之前TWST真实测出来的bug
    expect(screen.queryByText('半导体')).not.toBeInTheDocument()
  })

  it('expands the collapsed panel on click and shows a collapse-back control', async () => {
    fetch.mockReturnValue(
      jsonResponse({
        ticker: 'TWST',
        matched_themes: [],
        sic_matched_themes: [],
        benchmark: 'SPY',
        as_of: '2026-08-05',
        themes: [theme()],
        note: '',
      })
    )

    render(<ThematicFlowPanel ticker="TWST" />)
    const expandButton = await screen.findByText(/不在预设的AI算力产业链主题库里/)

    await userEvent.click(expandButton)

    expect(await screen.findByText('半导体')).toBeInTheDocument()
    expect(screen.getByText(/收起 · 仅供参考大盘AI板块行情/)).toBeInTheDocument()
  })

  it('shows the grid directly (no collapse) when the ticker does match a theme', async () => {
    fetch.mockReturnValue(
      jsonResponse({
        ticker: 'NVDA',
        matched_themes: ['AI芯片/GPU', '半导体'],
        sic_matched_themes: ['半导体'],
        benchmark: 'SPY',
        as_of: '2026-08-05',
        themes: [theme({ theme_name: 'AI芯片/GPU' }), theme({ theme_name: '半导体' })],
        note: '',
      })
    )

    render(<ThematicFlowPanel ticker="NVDA" />)

    expect(await screen.findByText('AI芯片/GPU')).toBeInTheDocument()
    expect(screen.getByText('半导体')).toBeInTheDocument()
  })

  it('distinguishes basket matches ("当前公司") from SIC-only matches ("行业分类识别")', async () => {
    fetch.mockReturnValue(
      jsonResponse({
        ticker: 'NVDA',
        matched_themes: ['AI芯片/GPU', '半导体'],
        sic_matched_themes: ['半导体'], // 半导体只靠SIC识别，AI芯片/GPU是篮子成分股
        benchmark: 'SPY',
        as_of: '2026-08-05',
        themes: [theme({ theme_name: 'AI芯片/GPU' }), theme({ theme_name: '半导体' })],
        note: '',
      })
    )

    render(<ThematicFlowPanel ticker="NVDA" />)
    await screen.findByText('AI芯片/GPU')

    expect(screen.getByText('当前公司')).toBeInTheDocument()
    expect(screen.getByText('行业分类识别')).toBeInTheDocument()
  })

  it('renders the error message when the fetch fails', async () => {
    fetch.mockReturnValue(
      Promise.resolve({ ok: false, status: 502, json: () => Promise.resolve({ detail: '上游服务不可用' }) })
    )

    render(<ThematicFlowPanel ticker="NVDA" />)

    expect(await screen.findByText('上游服务不可用')).toBeInTheDocument()
  })
})
