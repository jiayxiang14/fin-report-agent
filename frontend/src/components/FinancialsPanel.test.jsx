import { render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import FinancialsPanel from './FinancialsPanel'

function jsonResponse(body) {
  return Promise.resolve({ ok: true, json: () => Promise.resolve(body) })
}

describe('FinancialsPanel', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn())
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('renders without crashing and shows a directional yoy badge', async () => {
    fetch.mockReturnValue(
      jsonResponse({
        ticker: 'AAPL',
        cik: '0000320193',
        entity_name: 'Apple Inc.',
        source: 'SEC EDGAR companyfacts',
        retrieved_at: '2026-08-07T00:00:00Z',
        metrics: {
          revenue: {
            label: '营业收入',
            unit: 'USD',
            latest_annual: { end: '2025-09-27', val: 416161000000, fy: 2025, fp: 'FY', form: '10-K', yoy_change_pct: 6.43 },
            latest_quarterly: null,
          },
          operating_cash_flow: {
            label: '经营活动现金流',
            unit: 'USD',
            latest_annual: { end: '2025-09-27', val: 111482000000, fy: 2025, fp: 'FY', form: '10-K', yoy_change_pct: -5.73 },
            latest_quarterly: null,
          },
        },
      })
    )

    render(<FinancialsPanel ticker="AAPL" />)

    await waitFor(() => screen.getByText('Apple Inc.（AAPL，CIK 0000320193）'))

    expect(screen.getByText(/\+6\.43%/)).toBeInTheDocument()
    expect(screen.getByText(/-5\.73%/)).toBeInTheDocument()
  })
})
