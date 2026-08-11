export default function Panel({ icon: Icon, iconClassName, title, children }) {
  return (
    <div className="rounded-xl border border-gray-200 bg-white p-4 shadow-sm">
      <div className="flex items-center gap-2">
        <div
          className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-lg ${iconClassName || 'bg-gray-100 text-gray-600'}`}
        >
          <Icon className="h-4 w-4" />
        </div>
        <h2 className="text-sm font-semibold text-gray-700">{title}</h2>
      </div>
      <div className="mt-3">{children}</div>
    </div>
  )
}
