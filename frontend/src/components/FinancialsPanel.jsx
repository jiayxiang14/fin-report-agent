import { ArrowDown, ArrowUp, TrendingUp } from 'lucide-react'
import { useEffect, useState } from 'react'
import {
  Bar,
  BarChart,
  Cell,
  LabelList,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import Panel from './Panel'

// 表格里涨跌数字的绿色，色号取自 docs/color.png（品牌绿），红色沿用之前定的
// rose-600——这两个数字跟下面柱状图自己用的配色是分开的两套，不是同一套常量：
// 表格数字只是普通文字，柱状图的"浅色柱体+深色数字"是照抄市值/市盈率StatCard的
// emerald-100/emerald-600、rose-100/rose-600配色，两者色号来源不同，故意不合并
const TABLE_POSITIVE_COLOR = '#00AA6F'
const TABLE_NEGATIVE_COLOR = '#dc2626'

// 柱状图专用配色：跟CompanyProfilePanel里"市值"(emerald-100/600)和"市盈率"
// (rose-100/600)两个StatCard图标是同一套色号，浅色柱体+深色数字，不是渐变——
// 之前的鲜艳渐变被反馈"太刺眼"，换成跟别的面板统一的柔和pastel配色
const BAR_POSITIVE_FILL = '#d1fae5' // emerald-100
const BAR_POSITIVE_TEXT = '#059669' // emerald-600
const BAR_NEGATIVE_FILL = '#ffe4e6' // rose-100
const BAR_NEGATIVE_TEXT = '#e11d48' // rose-600

// 表格里的同比数字用箭头图标+颜色双重标识方向，不是只靠颜色——色弱用户也能分辨涨跌
function YoyBadge({ value }) {
  if (value == null) return <span className="text-gray-300">—</span>
  const isPositive = value >= 0
  const Icon = isPositive ? ArrowUp : ArrowDown
  return (
    <span
      className="inline-flex items-center gap-0.5 font-medium"
      style={{ color: isPositive ? TABLE_POSITIVE_COLOR : TABLE_NEGATIVE_COLOR }}
    >
      <Icon className="h-3 w-3" />
      {isPositive ? '+' : ''}
      {value}%
    </span>
  )
}

// 横向发散柱状图的数值标签：Bar在layout="vertical"下，正值的柱子从零点往右延伸
// （tip在x+width），负值的柱子从零点往左延伸（tip在x）——不能用LabelList内置的
// position="right"，那对负值柱子会把标签放到零点附近而不是柱子末端，得自己按
// 符号算tip坐标
function DivergingValueLabel({ x, y, width, height, value }) {
  const isPositive = value >= 0
  const tipX = isPositive ? x + width + 4 : x - 4
  return (
    <text
      x={tipX}
      y={y + height / 2}
      dy={3.5}
      textAnchor={isPositive ? 'start' : 'end'}
      fontSize={10}
      fontWeight={600}
      fill={isPositive ? BAR_POSITIVE_TEXT : BAR_NEGATIVE_TEXT}
    >
      {isPositive ? '▲' : '▼'} {isPositive ? '+' : ''}
      {value}%
    </text>
  )
}

function MetricCell({ point }) {
  if (!point) return <span className="text-gray-300">—</span>
  return (
    <>
      {point.val.toLocaleString()}
      <div className="text-xs text-gray-400">
        {point.fy ? `FY${point.fy}` : ''}
        {point.fp ? ` ${point.fp}` : ''}
        {point.fy || point.fp ? ' · ' : ''}
        截至 {point.end}（{point.form}）
      </div>
    </>
  )
}

export default function FinancialsPanel({ ticker }) {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (!ticker) return
    let cancelled = false
    setLoading(true)
    setError(null)
    setData(null)

    fetch(`/api/financials/${ticker}`)
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

  const yoyChartData = data
    ? Object.values(data.metrics)
        .filter((m) => m.latest_annual?.yoy_change_pct != null)
        .map((m) => ({ label: m.label, yoy: m.latest_annual.yoy_change_pct }))
    : []

  return (
    <Panel icon={TrendingUp} iconClassName="bg-emerald-100 text-emerald-600" title="核心财务数据">
      {loading && <p className="text-sm text-gray-400">加载中…</p>}
      {error && <p className="text-sm text-red-600">{error}</p>}

      {data && (
        <>
          <h3 className="text-lg font-medium">
            {data.entity_name}（{data.ticker}，CIK {data.cik}）
          </h3>
          <p className="mt-1 text-xs text-gray-400">
            数据来源：{data.source} · 拉取时间：{data.retrieved_at}
          </p>

          {yoyChartData.length > 0 && (
            <div className="mt-4" style={{ height: Math.max(160, yoyChartData.length * 30) }}>
              <p className="mb-1 text-xs font-medium text-gray-400">年度同比变化（%）</p>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={yoyChartData}
                  layout="vertical"
                  margin={{ left: 4, right: 48, top: 4, bottom: 4 }}
                >
                  {/* 纯浅色底反馈"太单薄"，加一层斜纹理（浅色底+稍深同色系斜线）撑出
                      层次感，不是走回之前被反馈"太刺眼"的高饱和渐变 */}
                  <defs>
                    <pattern
                      id="posStripes"
                      patternUnits="userSpaceOnUse"
                      width="6"
                      height="6"
                      patternTransform="rotate(45)"
                    >
                      <rect width="6" height="6" fill={BAR_POSITIVE_FILL} />
                      <line x1="0" y1="0" x2="0" y2="6" stroke="#6ee7b7" strokeWidth="2" />
                    </pattern>
                    <pattern
                      id="negStripes"
                      patternUnits="userSpaceOnUse"
                      width="6"
                      height="6"
                      patternTransform="rotate(45)"
                    >
                      <rect width="6" height="6" fill={BAR_NEGATIVE_FILL} />
                      <line x1="0" y1="0" x2="0" y2="6" stroke="#fda4af" strokeWidth="2" />
                    </pattern>
                  </defs>
                  <XAxis type="number" hide />
                  <YAxis type="category" dataKey="label" width={92} tick={{ fontSize: 11 }} />
                  <ReferenceLine x={0} stroke="#d1d5db" />
                  <Tooltip formatter={(value) => [`${value > 0 ? '+' : ''}${value}%`, '同比']} />
                  <Bar dataKey="yoy" radius={3}>
                    {yoyChartData.map((entry) => (
                      <Cell
                        key={entry.label}
                        fill={entry.yoy >= 0 ? 'url(#posStripes)' : 'url(#negStripes)'}
                      />
                    ))}
                    <LabelList dataKey="yoy" content={DivergingValueLabel} />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          <div className="mt-4 overflow-x-auto">
            <table className="w-full min-w-[520px] border-collapse text-sm">
              <thead>
                <tr className="border-b border-gray-200 text-left text-gray-500">
                  <th className="py-2">指标</th>
                  <th className="py-2">最新年度</th>
                  <th className="py-2">同比</th>
                  <th className="py-2">最新季度</th>
                  <th className="py-2">同比</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(data.metrics).map(([key, m]) => (
                  <tr key={key} className="border-b border-gray-100">
                    <td className="py-2">{m.label}</td>
                    <td className="py-2">
                      <MetricCell point={m.latest_annual} />
                    </td>
                    <td className="py-2">
                      <YoyBadge value={m.latest_annual?.yoy_change_pct} />
                    </td>
                    <td className="py-2">
                      <MetricCell point={m.latest_quarterly} />
                    </td>
                    <td className="py-2">
                      <YoyBadge value={m.latest_quarterly?.yoy_change_pct} />
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
