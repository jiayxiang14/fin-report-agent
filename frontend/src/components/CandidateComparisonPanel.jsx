import { AlertTriangle, Loader2, RefreshCw, Trophy, XCircle } from 'lucide-react'
import { useState } from 'react'
import Panel from './Panel'

const RULE_LABELS = {
  traceability: '数字可追溯性',
  self_verification: '自我核查',
  structure: '三段式结构',
  length: '长度合理性',
}

function statusOf(index, resolvedCount, done) {
  if (index < resolvedCount) return 'resolved' // 打分成功或失败，都算这个候选"跑完了"
  if (index === resolvedCount && !done) return 'running'
  return 'pending'
}

function CandidateCard({ index, candidate, isSelected, status }) {
  const [showReport, setShowReport] = useState(false)

  const failed = Boolean(candidate?.error)

  return (
    <div
      className={`rounded-lg border p-3 ${
        failed
          ? 'border-red-200 bg-red-50/60'
          : isSelected
            ? 'border-teal-400 bg-teal-50/60 ring-1 ring-teal-300'
            : 'border-gray-100 bg-gray-50'
      }`}
    >
      <div className="flex items-center justify-between">
        <span className="text-sm font-semibold text-gray-700">
          候选 {index + 1}
          {candidate && (
            <span className="ml-1.5 font-normal text-gray-400">temperature {candidate.temperature}</span>
          )}
        </span>
        {isSelected && (
          <span className="flex items-center gap-1 rounded-full bg-teal-600 px-2 py-0.5 text-xs font-medium text-white">
            <Trophy className="h-3 w-3" />
            已选中
          </span>
        )}
      </div>

      {status === 'pending' && <p className="mt-2 text-xs text-gray-400">等待中…</p>}
      {status === 'running' && (
        <p className="mt-2 flex items-center gap-1.5 text-xs text-gray-400">
          <Loader2 className="h-3.5 w-3.5 animate-spin" />
          生成中…
        </p>
      )}

      {failed && (
        <p className="mt-2 flex items-start gap-1.5 text-xs text-red-600">
          <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
          <span>这个候选运行失败，未参与打分和选择：{candidate.error}</span>
        </p>
      )}

      {candidate && !failed && (
        <>
          <div className="mt-2 flex items-center gap-2">
            <span className="text-2xl font-bold text-teal-600">{candidate.totalScore.toFixed(1)}</span>
            {candidate.reflexionTriggered && (
              <span className="flex items-center gap-1 rounded-full bg-amber-100 px-2 py-0.5 text-[11px] font-medium text-amber-700">
                <RefreshCw className="h-3 w-3" />
                已整改
              </span>
            )}
          </div>
          <div className="mt-2 grid grid-cols-2 gap-x-3 gap-y-1 text-xs text-gray-500">
            {Object.entries(RULE_LABELS).map(([key, label]) => (
              <div key={key} className="flex justify-between">
                <span>{label}</span>
                <span className="font-medium text-gray-700">{candidate.ruleScore[key].toFixed(1)}</span>
              </div>
            ))}
          </div>
          <div className="mt-2 border-t border-gray-100 pt-2 text-xs text-gray-500">
            结论裁判（评最终简报）：
            {candidate.llmScore != null ? (
              <span className="font-medium text-gray-700"> {candidate.llmScore.toFixed(1)}分</span>
            ) : (
              <span className="text-gray-400"> 未打分</span>
            )}
            {candidate.llmReason && <p className="mt-0.5 text-gray-400">{candidate.llmReason}</p>}
          </div>
          <div className="mt-2 border-t border-gray-100 pt-2 text-xs text-gray-500">
            过程裁判（评决策轨迹）：
            {candidate.trajectoryScore != null ? (
              <span className="font-medium text-gray-700"> {candidate.trajectoryScore.toFixed(1)}分</span>
            ) : (
              <span className="text-gray-400"> 未打分</span>
            )}
            {candidate.trajectoryReason && <p className="mt-0.5 text-gray-400">{candidate.trajectoryReason}</p>}
          </div>

          {candidate.finalReport && (
            <div className="mt-2 border-t border-gray-100 pt-2">
              <button
                type="button"
                onClick={() => setShowReport((v) => !v)}
                className="text-xs font-medium text-teal-600 hover:underline"
              >
                {showReport ? '收起正文' : '查看这份候选的完整简报'}
              </button>
              {showReport && (
                <pre className="mt-2 max-h-64 overflow-y-auto whitespace-pre-wrap break-words rounded-md bg-white p-2 text-xs text-gray-600">
                  {candidate.finalReport}
                </pre>
              )}
            </div>
          )}
        </>
      )}
    </div>
  )
}

// 展示Best-of-N三份候选各自的打分明细，以及最终选中了哪一份——项目书第9节
// 第5阶段验收点"更新前端，展示已生成N份候选、选中评分最高的一份"的落地。
// candidates按打分完成的顺序增长（后端顺序执行），还没轮到的候选显示"等待中"，
// 正在跑的显示"生成中"，这个状态完全从candidates.length和totalCount推算，
// 不需要额外的进度状态。
export default function CandidateComparisonPanel({ candidates, totalCount, result, done, streamError }) {
  const selectedIndex = result ? result.selected_index : null

  return (
    <Panel icon={Trophy} iconClassName="bg-teal-100 text-teal-600" title="候选对比（Best-of-N）">
      <p className="mb-3 text-xs text-gray-400">
        本次生成了{totalCount}份候选简报（不同temperature），用规则打分+结论裁判（评简报本身）+过程裁判（评决策轨迹）选出总分最高的一份展示。
      </p>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        {Array.from({ length: totalCount }, (_, index) => {
          const candidate = candidates.find((c) => c.index === index)
          return (
            <CandidateCard
              key={index}
              index={index}
              candidate={candidate}
              isSelected={selectedIndex === index}
              status={statusOf(index, candidates.length, done)}
            />
          )
        })}
      </div>

      {streamError && (
        <p className="mt-3 flex items-center gap-1.5 text-sm text-red-600">
          <XCircle className="h-4 w-4" />
          深度分析失败：{streamError}
        </p>
      )}
    </Panel>
  )
}
