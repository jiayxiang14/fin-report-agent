// jsdom没有内置EventSource实现。这个mock只做测试需要的最小子集：记录每次
// `new EventSource(url)`创建出来的实例（测试里用来拿到"最新那个连接"去手动
// 触发onmessage/onerror），以及一个可断言的close() spy——真实的网络行为
// （重连、readyState流转）用不上，不需要模拟。
import { vi } from 'vitest'

export class MockEventSource {
  static instances = []

  constructor(url) {
    this.url = url
    this.onmessage = null
    this.onerror = null
    this.close = vi.fn()
    MockEventSource.instances.push(this)
  }

  emit(eventPayload) {
    this.onmessage?.({ data: JSON.stringify(eventPayload) })
  }

  triggerError() {
    this.onerror?.()
  }

  static reset() {
    MockEventSource.instances = []
  }

  static latest() {
    return MockEventSource.instances[MockEventSource.instances.length - 1]
  }
}

export function installMockEventSource() {
  MockEventSource.reset()
  vi.stubGlobal('EventSource', MockEventSource)
}
