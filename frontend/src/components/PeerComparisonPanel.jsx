import { ArrowDown, ArrowUp, Users } from 'lucide-react'
import { useEffect, useState } from 'react'
import { Cell, LabelList, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from 'recharts'
import Panel from './Panel'

const POSITIVE_COLOR = '#16a34a'
const NEGATIVE_COLOR = '#dc2626'
const MAIN_COLOR = '#f59e0b'
const PEER_COLOR = '#94a3b8'

// 原来用 toLocaleString() 打印完整数字（比如"416,161,000,000"），5列挤在对半分的
// grid 里横向严重不够用；改成K/M/B缩写，跟FinancialsHistoryPanel是同一套格式化
function formatCompact(value) {
  if (value == null) return '—'
  const abs = Math.abs(value)
  if (abs >= 1e9) return `${(value / 1e9).toFixed(1)}B`
  if (abs >= 1e6) return `${(value / 1e6).toFixed(1)}M`
  if (abs >= 1e3) return `${(value / 1e3).toFixed(1)}K`
  return `${value}`
}

// 箭头图标+颜色双重标识涨跌方向，不是只靠颜色——跟FinancialsPanel是同一套"活力绿"方案
function YoyBadge({ value }) {
  if (value == null) return null
  const isPositive = value >= 0
  const Icon = isPositive ? ArrowUp : ArrowDown
  return (
    <span
      className="inline-flex items-center gap-0.5 text-xs font-medium"
      style={{ color: isPositive ? POSITIVE_COLOR : NEGATIVE_COLOR }}
    >
      <Icon className="h-2.5 w-2.5" />
      {isPositive ? '+' : ''}
      {value}%
    </span>
  )
}

function ScatterTooltip({ active, payload }) {
  if (!active || !payload?.length) return null
  const d = payload[0].payload
  return (
    <div className="rounded-md border border-gray-200 bg-white px-2.5 py-1.5 text-xs shadow-sm">
      <div className="font-medium text-gray-700">{d.label}</div>
      <div className="text-gray-500">营收 {formatCompact(d.revenue)}</div>
      <div className="text-gray-500">净利润 {formatCompact(d.net_income)}</div>
    </div>
  )
}

async function fetchJson(url) {
  const res = await fetch(url)
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body.detail || `请求失败：${res.status}`)
  }
  return res.json()
}

export default function PeerComparisonPanel({ ticker }) {
  const [peerData, setPeerData] = useState(null)
  const [mainFinancials, setMainFinancials] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (!ticker) return
    let cancelled = false
    setLoading(true)
    setError(null)
    setPeerData(null)
    setMainFinancials(null)

    // 同行对比图想同时画上这只ticker自己的营收/净利润做参照，所以这里额外拉一次
    // /api/financials——financials本身有6小时缓存，多这一次请求成本很低，
    // 换来的是组件之间保持独立自取数据（不需要从父组件props传递）
    Promise.all([
      fetchJson(`/api/peer-comparison/${ticker}`),
      fetchJson(`/api/financials/${ticker}`).catch(() => null),
    ])
      .then(([peers, financials]) => {
        if (cancelled) return
        setPeerData(peers)
        setMainFinancials(financials)
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

  const mainRevenue = mainFinancials?.metrics?.revenue?.latest_annual?.val ?? null
  const mainNetIncome = mainFinancials?.metrics?.net_income?.latest_annual?.val ?? null

  // 散点图能看出柱状图看不出来的东西：谁体量大但没那么赚钱——横轴营收看规模，
  // 纵轴净利润看盈利效率，两个维度一起看比单一维度的柱状图信息量更大
  const scatterData = peerData
    ? [
        ...(mainRevenue != null && mainNetIncome != null
          ? [{ label: `${ticker}（本公司）`, revenue: mainRevenue, net_income: mainNetIncome, isMain: true }]
          : []),
        ...peerData.peers
          .filter((p) => p.revenue != null && p.net_income != null)
          .map((p) => ({ label: p.ticker, revenue: p.revenue, net_income: p.net_income, isMain: false })),
      ]
    : []

  return (
    <Panel icon={Users} iconClassName="bg-amber-100 text-amber-600" title="同行对比">
      {loading && <p className="text-sm text-gray-400">加载中…</p>}
      {error && <p className="text-sm text-red-600">{error}</p>}

      {peerData && peerData.note && <p className="text-sm text-amber-600">{peerData.note}</p>}

      {peerData && !peerData.note && peerData.peers.length === 0 && (
        <p className="text-sm text-gray-400">同板块的其他公司暂时没有可用的对比数据</p>
      )}

      {scatterData.length > 0 && (
        <div className="mt-1 h-52">
          <p className="mb-1 text-xs font-medium text-gray-400">营收 × 净利润</p>
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 14, right: 16, bottom: 4, left: -12 }}>
              <XAxis
                type="number"
                dataKey="revenue"
                name="营收"
                tick={{ fontSize: 10 }}
                tickFormatter={formatCompact}
              />
              <YAxis
                type="number"
                dataKey="net_income"
                name="净利润"
                tick={{ fontSize: 10 }}
                tickFormatter={formatCompact}
                width={44}
              />
              <Tooltip cursor={{ strokeDasharray: '3 3' }} content={<ScatterTooltip />} />
              <Scatter data={scatterData}>
                {scatterData.map((entry) => (
                  <Cell key={entry.label} fill={entry.isMain ? MAIN_COLOR : PEER_COLOR} />
                ))}
                <LabelList dataKey="label" position="top" style={{ fontSize: 10, fill: '#6b7280' }} />
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      )}

      {peerData && peerData.peers.length > 0 && (
        // "公司/营收/净利润"3列，把"数值+同比"合并进同一个单元格（数值在上，
        // 同比小字带箭头在下方），数值用K/M/B缩写，不再挤占列宽
        <div className="mt-3 overflow-x-auto">
          <table className="w-full min-w-[360px] border-collapse text-sm">
            <thead>
              <tr className="border-b border-gray-200 text-left text-gray-500">
                <th className="py-2.5">公司</th>
                <th className="py-2.5">营收</th>
                <th className="py-2.5">净利润</th>
              </tr>
            </thead>
            <tbody>
              {peerData.peers.map((p) => (
                <tr key={p.ticker} className="border-b border-gray-100">
                  <td className="py-3">
                    <div className="font-medium text-gray-700">{p.ticker}</div>
                    <div className="text-xs text-gray-400">{p.entity_name}</div>
                  </td>
                  <td className="py-3">
                    <div>{formatCompact(p.revenue)}</div>
                    <YoyBadge value={p.revenue_yoy_pct} />
                  </td>
                  <td className="py-3">
                    <div>{formatCompact(p.net_income)}</div>
                    <YoyBadge value={p.net_income_yoy_pct} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Panel>
  )
}
