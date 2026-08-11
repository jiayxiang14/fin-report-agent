// 把毫秒格式化成人读的用时文案。60秒以内保留1位小数（这个粒度下的分析大概率
// 在几秒到几十秒之间，1位小数能看出"12.3秒"和"12.8秒"这种细微差别，用户盯着
// 看的时候有"确实在走"的实感）；超过60秒切到"X分Y秒"，不然"185.3秒"这种大数字
// 反而没有"3分5秒"直观。
export function formatElapsed(ms) {
  const totalSeconds = ms / 1000
  if (totalSeconds < 60) {
    return `${totalSeconds.toFixed(1)}秒`
  }
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = Math.round(totalSeconds % 60)
  return `${minutes}分${seconds}秒`
}
