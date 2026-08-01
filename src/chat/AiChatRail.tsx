import { useState, type FormEvent, type KeyboardEvent } from 'react'
import { ArrowUp } from 'lucide-react'
import { WindowControls } from './WindowControls'

/**
 * 壳层常驻右栏：不参与业务页切换。
 * 整体收入与业务区同款矩形胶囊，窗控也在胶囊内。
 */
export function AiChatRail() {
  const [draft, setDraft] = useState('')

  const submit = (event?: FormEvent) => {
    event?.preventDefault()
    setDraft('')
  }

  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      if (draft.trim()) submit()
    }
  }

  return (
    <aside className="ai-rail" aria-label="AI Chat" data-tauri-drag-region="false">
      <div className="ai-rail-capsule">
        <header className="ai-rail-header" data-tauri-drag-region>
          <div className="ai-rail-drag" data-tauri-drag-region />
          <WindowControls />
        </header>

        <div className="ai-rail-body" />

        <form className="ai-rail-composer" onSubmit={submit}>
          <textarea
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={onKeyDown}
            rows={2}
            placeholder="消息"
            aria-label="消息输入"
          />
          <button
            type="submit"
            className="ai-rail-send"
            disabled={!draft.trim()}
            aria-label="发送"
          >
            <ArrowUp size={15} strokeWidth={2.2} aria-hidden />
          </button>
        </form>
      </div>
    </aside>
  )
}
