// 把 Agent 生成的 final_report 原始文本，按 generation.md 里要求的三段式
// XML 标签（<conclusion>/<evidence>/<flags>）切分成结构化数据，前端按标签
// 渲染成三个视觉区块（项目书7.1①）。标签之前的前言文字是模型自己按prompt
// 要求写的"数据来源/截止日期"说明（generation.md第19行），原样透传即可，
// 不需要前端自己拼一句静态文案。
//
// sentiment 是第4个标签（generation.md），放在</flags>之后，值只能是
// positive/negative/neutral——ReportPanel用它决定结论区块的颜色（绿/红/中性灰），
// 不是三段式结构本身的一部分，用同一套提取逻辑复用即可，模型没写这个标签时
// sections.sentiment 就是 undefined，前端会退化成中性配色。

const TAG_ORDER = ['conclusion', 'evidence', 'flags', 'sentiment']
const TAG_ALTERNATION = TAG_ORDER.join('|')

export function parseReport(rawText) {
  if (!rawText) {
    return { preamble: '', sections: {}, hasAnyTag: false }
  }

  const sections = {}
  let hasAnyTag = false
  for (const tag of TAG_ORDER) {
    let match = rawText.match(new RegExp(`<${tag}>([\\s\\S]*?)</${tag}>`))
    if (!match) {
      // 兜底：模型偶尔漏写闭合标签（实测约0.1%概率），严格匹配会让整块内容
      // 消失——退化成匹配到下一个已知标签开始或文本结尾为止
      match = rawText.match(new RegExp(`<${tag}>([\\s\\S]*?)(?=<(?:${TAG_ALTERNATION})>|$)`))
    }
    if (match) {
      sections[tag] = match[1].trim()
      hasAnyTag = true
    }
  }

  let preamble = ''
  const firstTagIndex = rawText.indexOf('<conclusion>')
  if (firstTagIndex > 0) {
    preamble = rawText.slice(0, firstTagIndex).trim()
  }

  return { preamble, sections, hasAnyTag }
}
