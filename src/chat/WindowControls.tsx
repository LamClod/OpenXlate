import { useCallback } from 'react'
import { Minus, Square, X } from 'lucide-react'
import { api } from '../api/tauri'

/** 自绘三控件（最小化 / 最大化 / 关闭），置于 AI 右栏顶。 */
export function WindowControls() {
  const handleClose = useCallback(() => {
    void api.closeWindow()
  }, [])

  const handleMinimize = useCallback(() => {
    void api.minimizeWindow()
  }, [])

  const handleMaximize = useCallback(() => {
    void api.toggleMaximizeWindow()
  }, [])

  return (
    <div className="chat-win-controls" data-tauri-drag-region="false">
      <button type="button" className="chat-win-btn" onClick={handleMinimize} aria-label="最小化">
        <Minus size={14} strokeWidth={1.9} aria-hidden />
      </button>
      <button type="button" className="chat-win-btn" onClick={handleMaximize} aria-label="最大化">
        <Square size={12} strokeWidth={1.9} aria-hidden />
      </button>
      <button
        type="button"
        className="chat-win-btn chat-win-btn--close"
        onClick={handleClose}
        aria-label="关闭"
      >
        <X size={14} strokeWidth={2.1} aria-hidden />
      </button>
    </div>
  )
}
