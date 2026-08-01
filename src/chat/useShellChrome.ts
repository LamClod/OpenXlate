import { useEffect } from 'react'

/** 可编辑文字区：允许文本拖选 */
const TEXT_SELECTOR =
  'input, textarea, select, option, [contenteditable=""], [contenteditable="true"]'

function closestElement(target: EventTarget | null): Element | null {
  if (target instanceof Element) return target
  if (target instanceof Text) return target.parentElement
  return null
}

function isTextArea(el: Element | null): boolean {
  return !!el?.closest(TEXT_SELECTOR)
}

/**
 * 桌面壳层交互约束：
 * - 禁止右键系统菜单
 * - 禁止拖出图片 / 链接等浏览器默认拖拽
 * 窗体拖动仅保留 data-tauri-drag-region 原生拖区，不做控件长按拖窗。
 */
export function useShellChrome(rootSelector = '.window-shell') {
  useEffect(() => {
    const root = document.querySelector(rootSelector)
    if (!root) return

    const onContextMenu = (event: Event) => {
      event.preventDefault()
    }

    const onDragStart = (event: Event) => {
      const el = closestElement(event.target)
      if (!isTextArea(el)) {
        event.preventDefault()
      }
    }

    root.addEventListener('contextmenu', onContextMenu)
    root.addEventListener('dragstart', onDragStart)

    return () => {
      root.removeEventListener('contextmenu', onContextMenu)
      root.removeEventListener('dragstart', onDragStart)
    }
  }, [rootSelector])
}
