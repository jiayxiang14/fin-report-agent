import { ChevronDown, ChevronRight, Radar } from 'lucide-react'
import { useEffect, useState } from 'react'
import { Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, YAxis } from 'recharts'
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

// 10个主题彼此的产业链位置（上游硬件组件→中游基础设施建设运营→下游云端
// 部署），是固定的、通用的产业常识分类，跟具体分析哪家公司无关——不是猜测
// 具体供应商/客户关系。按这个顺序分组展示，比一个平铺的网格更能看出"这是
// 一条产业链"，不只是"这些主题都相关"。
const CHAIN_ORDER = ['上游', '中游', '下游']

const CHAIN_COLORS = {
  上游: 'bg-sky-100 text-sky-700',
  中游: 'bg-amber-100 text-amber-700',
  下游: 'bg-emerald-100 text-emerald-700',
}

function groupByChainPosition(themes) {
  const groups = {}
  for (const theme of themes) {
    const key = theme.chain_position
    if (!groups[key]) groups[key] = []
    groups[key].push(theme)
  }
  return CHAIN_ORDER.filter((position) => groups[position]?.length).map((position) => ({
    position,
    themes: groups[position],
  }))
}

function ThemeCard({ theme, isMatched, isSicMatched }) {
  return (
    <div
      className={`rounded-lg border p-3 ${
        isMatched ? 'border-fuchsia-300 bg-fuchsia-50/60 ring-1 ring-fuchsia-200' : 'border-gray-100 bg-gray-50'
      }`}
    >
      <div className="flex items-center justify-between">
        <span className="text-sm font-semibold text-gray-700">
          {theme.theme_name}
          {isMatched && (
            <span
              title={
                isSicMatched
                  ? '不在预设成分股名单里，是根据公司官方SIC行业分类自动识别出来的'
                  : '公司本身就是这个主题篮子的成分股'
              }
              className="ml-1.5 rounded-full bg-fuchsia-600 px-1.5 py-0.5 text-[10px] font-medium text-white"
            >
              {isSicMatched ? '行业分类识别' : '当前公司'}
            </span>
          )}
        </span>
        <span
          className={`rounded px-1.5 py-0.5 text-xs font-medium ${QUADRANT_COLORS[theme.quadrant] || 'bg-gray-100 text-gray-700'}`}
        >
          {QUADRANT_LABELS[theme.quadrant] || theme.quadrant}
        </span>
      </div>
      <p className="mt-1 text-xs text-gray-400">
        {theme.constituent_tickers.join(' / ')}
      </p>
      <p className="mt-1 text-xs text-gray-400">
        RS-Ratio {theme.rs_ratio} · RS-Momentum {theme.rs_momentum}
      </p>
      {theme.history.length > 0 && (
        <div className="mt-2 h-16">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={theme.history}>
              {/* 之前没声明YAxis时，Recharts数字轴默认从0开始——RS-Ratio/RS-Momentum
                  本身只在100上下小幅波动，跟0比这点波动被压成两条几乎贴底的线，看起来像
                  重合了。domain=['auto','auto']让Y轴紧贴这组数据自己的范围，波动才看得出来 */}
              <YAxis hide domain={['auto', 'auto']} />
              <ReferenceLine y={100} stroke="#cbd5e1" strokeDasharray="3 3" />
              <Tooltip
                labelFormatter={(label) => `日期：${label}`}
                formatter={(value, name) => [value, name === 'rs_ratio' ? 'RS-Ratio' : 'RS-Momentum']}
              />
              <Line type="monotone" dataKey="rs_ratio" stroke="#0ea5e9" dot={false} strokeWidth={2} />
              <Line type="monotone" dataKey="rs_momentum" stroke="#f97316" dot={false} strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  )
}

// 展示的始终是同样10个细分产业主题现在的相对轮动位置，不因为分析哪家公司
// 而改变——但传了ticker就顺带做一次确定性反查（这个ticker本身是不是某个
// 主题篮子的成分股/官方SIC行业分类命中），命中的主题会高亮，不是猜的，是
// 后端代码算出来的。传了ticker但matched_themes是空的（比如分析一家生物科技
// 公司），说明这10个AI算力主题跟这次分析确实无关——默认只显示一行折叠提示，
// 不强行铺满10张卡片制造"看起来都相关"的错觉，用户想看大盘AI板块参考数据
// 时自己点开。
export default function ThematicFlowPanel({ ticker }) {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(true)
  const [expanded, setExpanded] = useState(false)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setExpanded(false)

    const url = ticker ? `/api/thematic-flow?ticker=${encodeURIComponent(ticker)}` : '/api/thematic-flow'

    fetch(url)
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

  const isIrrelevant = Boolean(data?.ticker) && data?.matched_themes?.length === 0
  const showGrid = data && (!isIrrelevant || expanded)

  return (
    <Panel icon={Radar} iconClassName="bg-fuchsia-100 text-fuchsia-600" title="细分产业主题轮动">
      {loading && <p className="text-sm text-gray-400">加载中…</p>}
      {error && <p className="text-sm text-red-600">{error}</p>}

      {isIrrelevant && !expanded && (
        <button
          type="button"
          onClick={() => setExpanded(true)}
          className="flex w-full items-center justify-between rounded-lg border border-gray-100 bg-gray-50 px-3 py-2 text-left text-sm text-gray-500 hover:bg-gray-100"
        >
          <span>当前公司（{data.ticker}）不在预设的AI算力产业链主题库里，跟这10个主题暂无关联</span>
          <ChevronDown className="h-4 w-4 shrink-0 text-gray-400" />
        </button>
      )}

      {showGrid && (
        <>
          {isIrrelevant && (
            <button
              type="button"
              onClick={() => setExpanded(false)}
              className="mb-3 text-xs font-medium text-gray-400 hover:underline"
            >
              收起 · 仅供参考大盘AI板块行情，跟{data.ticker}本身无关
            </button>
          )}
          {/* 所有卡片共用同一个grid，分组标题用col-span-full插入成跨列的分隔行——
              这样任何一张卡片都严格落在同一套列网格里，不会因为某组只有1张卡片
              就导致那一行看起来空荡荡、跟别的组的列对不齐 */}
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {groupByChainPosition(data.themes).flatMap((group) => [
              <div key={`header-${group.position}`} className="col-span-full flex items-center gap-1.5">
                <span
                  className={`rounded-full px-2 py-1 text-xs font-semibold ${CHAIN_COLORS[group.position] || 'bg-gray-100 text-gray-700'}`}
                >
                  {group.position}
                </span>
                {group.position !== CHAIN_ORDER[CHAIN_ORDER.length - 1] && (
                  <ChevronRight className="h-3.5 w-3.5 text-gray-300" />
                )}
              </div>,
              ...group.themes.map((theme) => (
                <ThemeCard
                  key={theme.theme_name}
                  theme={theme}
                  isMatched={data.matched_themes?.includes(theme.theme_name)}
                  isSicMatched={data.sic_matched_themes?.includes(theme.theme_name)}
                />
              )),
            ])}
          </div>
          <p className="mt-3 text-xs text-gray-400">
            相对 {data.benchmark} · 数据截至 {data.as_of} · 上游/中游/下游是通用产业常识分类，不针对具体公司关系 ·{' '}
            {data.note}
          </p>
        </>
      )}
    </Panel>
  )
}
