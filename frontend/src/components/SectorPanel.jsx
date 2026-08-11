import { Compass } from 'lucide-react'
import { useEffect, useState } from 'react'
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import Panel from './Panel'

const QUADRANT_LABELS = {
  leading: '领先（Leading）',
  weakening: '转弱（Weakening）',
  lagging: '落后（Lagging）',
  improving: '改善（Improving）',
}

const QUADRANT_COLORS = {
  leading: 'bg-green-100 text-green-700',
  weakening: 'bg-yellow-100 text-yellow-700',
  lagging: 'bg-red-100 text-red-700',
  improving: 'bg-blue-100 text-blue-700',
}

export default function SectorPanel({ ticker }) {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (!ticker) return
    let cancelled = false
    setLoading(true)
    setError(null)
    setData(null)

    fetch(`/api/sector-position/${ticker}`)
      .then(async (res) => {
        if (!res.ok) {
          const body = await res.json().catch(() => ({}))
          throw new Error(body.detail || `请求失败：${res.status}`)
        }
        return res.json()
      })
      .then((json) => {
        if (!cancelled) setData(json)
      })
      .catch((err) => {
        if (!cancelled) setError(err.message)
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })

    return () => {
      cancelled = true
    }
  }, [ticker])

  if (!ticker) return null

  const matched = data?.sectors?.find((s) => s.sector_etf === data.matched_sector_etf)

  return (
    <Panel icon={Compass} iconClassName="bg-sky-100 text-sky-600" title="板块轮动位置">
      {loading && <p className="text-sm text-gray-400">加载中…</p>}
      {error && <p className="text-sm text-red-600">{error}</p>}

      {data && (
        <>
          {data.note && <p className="text-sm text-amber-600">{data.note}</p>}

          {matched && (
            <>
              <div className="flex items-center gap-2">
                <span className="text-lg font-medium">{matched.sector_name}</span>
                <span
                  className={`rounded px-2 py-0.5 text-xs font-medium ${QUADRANT_COLORS[matched.quadrant] || 'bg-gray-100 text-gray-700'}`}
                >
                  {QUADRANT_LABELS[matched.quadrant] || matched.quadrant}
                </span>
              </div>
              <p className="mt-1 text-xs text-gray-400">
                相对 {data.benchmark} · RS-Ratio {matched.rs_ratio} · RS-Momentum{' '}
                {matched.rs_momentum} · 数据截至 {data.as_of}
              </p>

              {matched.history.length > 0 && (
                <div className="mt-3 h-40">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={matched.history}>
                      <XAxis dataKey="date" hide />
                      <YAxis domain={['auto', 'auto']} width={36} tick={{ fontSize: 11 }} />
                      <Tooltip
                        labelFormatter={(label) => `日期：${label}`}
                        formatter={(value, name) => [
                          value,
                          name === 'rs_ratio' ? 'RS-Ratio' : 'RS-Momentum',
                        ]}
                      />
                      <Line
                        type="monotone"
                        dataKey="rs_ratio"
                        stroke="#0ea5e9"
                        dot={false}
                        strokeWidth={2}
                      />
                      <Line
                        type="monotone"
                        dataKey="rs_momentum"
                        stroke="#f97316"
                        dot={false}
                        strokeWidth={2}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}
            </>
          )}

          <div className="mt-4 overflow-x-auto">
            <table className="w-full min-w-[420px] border-collapse text-xs">
              <thead>
                <tr className="border-b border-gray-200 text-left text-gray-500">
                  <th className="py-1.5">板块</th>
                  <th className="py-1.5">RS-Ratio</th>
                  <th className="py-1.5">RS-Momentum</th>
                  <th className="py-1.5">象限</th>
                </tr>
              </thead>
              <tbody>
                {data.sectors.map((s) => (
                  <tr
                    key={s.sector_etf}
                    className={`border-b border-gray-100 ${s.sector_etf === data.matched_sector_etf ? 'bg-sky-50 font-medium' : ''}`}
                  >
                    <td className="py-1.5">
                      {s.sector_name}（{s.sector_etf}）
                    </td>
                    <td className="py-1.5">{s.rs_ratio}</td>
                    <td className="py-1.5">{s.rs_momentum}</td>
                    <td className="py-1.5">
                      <span
                        className={`rounded px-1.5 py-0.5 ${QUADRANT_COLORS[s.quadrant] || 'bg-gray-100 text-gray-700'}`}
                      >
                        {QUADRANT_LABELS[s.quadrant] || s.quadrant}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </Panel>
  )
}
