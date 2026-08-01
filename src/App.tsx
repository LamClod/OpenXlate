import { useEffect } from 'react'
import { WindowHost } from './chat/WindowHost'
import { Sidebar } from './chat/Sidebar'
import { AiChatRail } from './chat/AiChatRail'
import { ChatDotGridBackground } from './chat/ChatDotGridBackground'
import { useShellChrome } from './chat/useShellChrome'
import { api, isTauriRuntime } from './api/tauri'
import { ProviderManagement } from './providers/ProviderManagement'
import './index.css'

/** 与 kivio 聊天主窗 CHAT_DEFAULT_SIZE 一致 */
const WINDOW_SIZE = { width: 1280, height: 800 }

export default function App() {
  useShellChrome()

  useEffect(() => {
    if (!isTauriRuntime()) return
    void (async () => {
      await api.resizeWindow(WINDOW_SIZE.width, WINDOW_SIZE.height)
      try {
        const { getCurrentWindow } = await import('@tauri-apps/api/window')
        const { LogicalSize } = await import('@tauri-apps/api/dpi')
        await getCurrentWindow().setMinSize(new LogicalSize(400, 400))
      } catch {
        /* ignore */
      }
    })()
  }, [])

  return (
    <WindowHost>
      <div className="window-shell dark">
        <ChatDotGridBackground />
        {/* 中栏顶部拖拽；左右胶囊栏自带顶栏 drag / no-drag */}
        <div className="window-titlebar-drag" data-tauri-drag-region />
        <div className="app-frame">
          <Sidebar />
          <div className="app-workspace">
            <ProviderManagement />
          </div>
          <AiChatRail />
        </div>
      </div>
    </WindowHost>
  )
}
