import { Building2, DollarSign, Gauge, Layers, Scale, Users } from 'lucide-react'
import { useEffect, useState } from 'react'
import Panel from './Panel'

function formatMarketCap(value) {
  if (value == null) return '—'
  const trillion = 1_000_000_000_000
  const billion = 1_000_000_000
  const million = 1_000_000
  if (value >= trillion) return `$${(value / trillion).toFixed(2)}T`
  if (value >= billion) return `$${(value / billion).toFixed(2)}B`
  if (value >= million) return `$${(value / million).toFixed(2)}M`
  return `$${value.toLocaleString()}`
}

function StatCard({ icon: Icon, iconClassName, label, value, title }) {
  return (
    <div
      title={title}
      className="flex items-center gap-3 rounded-lg border border-gray-100 bg-gray-50 px-3 py-2.5"
    >
      <div
        className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-lg ${iconClassName}`}
      >
        <Icon className="h-4 w-4" />
      </div>
      <div className="min-w-0">
        <div className="text-xs text-gray-400">{label}</div>
        <div className="truncate text-base font-semibold">{value}</div>
      </div>
    </div>
  )
}

export default function CompanyProfilePanel({ ticker }) {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (!ticker) return
    let cancelled = false
    setLoading(true)
    setError(null)
    setData(null)

    fetch(`/api/company-profile/${ticker}`)
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

  return (
    <Panel icon={Building2} iconClassName="bg-indigo-100 text-indigo-600" title="公司概览">
      {loading && <p className="text-sm text-gray-400">加载中…</p>}
      {error && <p className="text-sm text-red-600">{error}</p>}

      {data && !data.has_data && <p className="text-sm text-amber-600">{data.note}</p>}

      {data && data.has_data && (
        <>
          <h3 className="text-lg font-medium">
            {data.name || ticker}
            {data.homepage_url && (
              <a
                href={data.homepage_url}
                target="_blank"
                rel="noreferrer"
                className="ml-2 text-xs font-normal text-indigo-500 hover:underline"
              >
                官网 ↗
              </a>
            )}
          </h3>

          {data.description && <p className="mt-1 text-sm text-gray-600">{data.description}</p>}

          <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-5">
            <StatCard
              icon={DollarSign}
              iconClassName="bg-emerald-100 text-emerald-600"
              label="市值"
              value={formatMarketCap(data.market_cap)}
            />
            <StatCard
              icon={Scale}
              iconClassName="bg-rose-100 text-rose-600"
              label="市盈率(P/E)"
              title="最新收盘价 / 最新一期年度稀释EPS，不是严格意义上过去四个季度滚动的TTM市盈率"
              value={data.pe_ratio != null ? data.pe_ratio.toFixed(1) : '—'}
            />
            <StatCard
              icon={Gauge}
              iconClassName="bg-orange-100 text-orange-600"
              label="20日平均日振幅"
              value={data.adr_20d_pct != null ? `${data.adr_20d_pct}%` : '—'}
            />
            <StatCard
              icon={Layers}
              iconClassName="bg-violet-100 text-violet-600"
              label="细分行业"
              value={data.sic_description || '—'}
            />
            <StatCard
              icon={Users}
              iconClassName="bg-sky-100 text-sky-600"
              label="员工数"
              value={data.total_employees != null ? data.total_employees.toLocaleString() : '—'}
            />
          </div>
        </>
      )}
    </Panel>
  )
}
