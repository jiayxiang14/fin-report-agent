import { History } from 'lucide-react'
import { useEffect, useState } from 'react'
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import Panel from './Panel'

// 折线图重点看的是多年趋势，不是逐年精确数值——绝对值用K/M/B缩写，跟坐标轴/悬浮框
// 都共用同一套格式化，避免大公司财务数字（十亿美元级别）把坐标轴挤得全是长数字
function formatCompact(value) {
  if (value == null) return '—'
  const abs = Math.abs(value)
  if (abs >= 1e9) return `${(value / 1e9).toFixed(1)}B`
  if (abs >= 1e6) return `${(value / 1e6).toFixed(1)}M`
  if (abs >= 1e3) return `${(value / 1e3).toFixed(1)}K`
  return `${value}`
}

function formatYear(end) {
  return end ? end.slice(0, 4) : ''
}

// 只挑这几个最能反映"公司在赚钱、赚的是不是真现金"的科目做趋势图，不是把
// history 里全部指标都铺出来——呼应用户最初关心的"净利润好看但现金流转负"
// 这类背离，营收看规模、净利润看账面盈利、经营现金流+自由现金流看真实现金创造
const CHART_METRICS = [
  { key: 'revenue', color: '#0ea5e9' },
  { key: 'net_income', color: '#8b5cf6' },
  { key: 'operating_cash_flow', color: '#10b981' },
  { key: 'free_cash_flow', color: '#f59e0b' },
]

function MetricHistoryChart({ label, points, color }) {
  const data = points.map((p) => ({ ...p, year: formatYear(p.end) }))
  return (
    <div className="rounded-lg border border-gray-100 bg-gray-50 p-3">
      <p className="text-xs font-medium text-gray-500">{label}</p>
      <div className="mt-1 h-32">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ left: -20, right: 8 }}>
            <XAxis dataKey="year" tick={{ fontSize: 10 }} />
            <YAxis tick={{ fontSize: 10 }} tickFormatter={formatCompact} width={44} />
            <Tooltip
              labelFormatter={(_label, payload) => `截至 ${payload?.[0]?.payload?.end ?? ''}`}
              formatter={(value, _name, entry) => [
                `${formatCompact(value)}${entry.payload.yoy_change_pct != null ? `（同比 ${entry.payload.yoy_change_pct > 0 ? '+' : ''}${entry.payload.yoy_change_pct}%）` : ''}`,
                label,
              ]}
            />
            <Line type="monotone" dataKey="val" stroke={color} dot={{ r: 2 }} strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

export default function FinancialsHistoryPanel({ ticker }) {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (!ticker) return
    let cancelled = false
    setLoading(true)
    setError(null)
    setData(null)

    fetch(`/api/financials-history/${ticker}`)
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

  const charts = data
    ? CHART_METRICS.map(({ key, color }) => ({ key, color, metric: data.history[key] })).filter(
        (c) => c.metric && c.metric.points.length > 0
      )
    : []

  return (
    <Panel icon={History} iconClassName="bg-violet-100 text-violet-600" title="多年财务趋势">
      {loading && <p className="text-sm text-gray-400">加载中…</p>}
      {error && <p className="text-sm text-red-600">{error}</p>}

      {data && charts.length === 0 && (
        <p className="text-sm text-gray-400">没有足够的多年历史数据可供展示</p>
      )}

      {data && charts.length > 0 && (
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          {charts.map(({ key, color, metric }) => (
            <MetricHistoryChart key={key} label={metric.label} points={metric.points} color={color} />
          ))}
        </div>
      )}
    </Panel>
  )
}
