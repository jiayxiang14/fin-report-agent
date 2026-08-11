import { render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import PeerComparisonPanel from './PeerComparisonPanel'

function jsonResponse(body) {
  return Promise.resolve({ ok: true, json: () => Promise.resolve(body) })
}

describe('PeerComparisonPanel', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn())
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('renders the peer table (scatter chart + table both driven by the same fetch)', async () => {
    fetch.mockImplementation((url) => {
      if (url.startsWith('/api/peer-comparison/')) {
        return jsonResponse({
          ticker: 'AAPL',
          sector_etf: 'XLK',
          sector_name: 'Technology',
          peers: [
            {
              ticker: 'MSFT',
              entity_name: 'Microsoft Corp.',
              revenue: 282000000000,
              revenue_yoy_pct: 12.1,
              net_income: 96000000000,
              net_income_yoy_pct: -3.2,
            },
          ],
        })
      }
      return jsonResponse({
        ticker: 'AAPL',
        cik: '0000320193',
        entity_name: 'Apple Inc.',
        metrics: {
          revenue: { latest_annual: { val: 416161000000 } },
          net_income: { latest_annual: { val: 112010000000 } },
        },
      })
    })

    render(<PeerComparisonPanel ticker="AAPL" />)

    await waitFor(() => screen.getByText('MSFT'))

    expect(screen.getByText('Microsoft Corp.')).toBeInTheDocument()
    expect(screen.getByText(/\+12\.1%/)).toBeInTheDocument()
    expect(screen.getByText(/-3\.2%/)).toBeInTheDocument()
  })
})
