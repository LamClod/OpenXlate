import { useEffect, useState, type ReactNode } from 'react'
import { isTauriRuntime } from '../api/tauri'

type WindowHostProps = {
  children: ReactNode
}

/** 窗口外壳：最大化时用 padding 补偿 Windows 无边框溢出 */
export function WindowHost({ children }: WindowHostProps) {
  const [maximized, setMaximized] = useState(false)

  useEffect(() => {
    if (!isTauriRuntime()) return

    let cancelled = false
    let unlistenResized: (() => void) | undefined
    let timer: ReturnType<typeof setTimeout> | undefined

    const syncMaximized = async () => {
      try {
        const { getCurrentWindow } = await import('@tauri-apps/api/window')
        const next = await getCurrentWindow().isMaximized()
        if (!cancelled) setMaximized(next)
      } catch {
        /* ignore */
      }
    }

    const scheduleSync = () => {
      if (timer !== undefined) clearTimeout(timer)
      timer = setTimeout(() => void syncMaximized(), 40)
    }

    const setup = async () => {
      await syncMaximized()
      const { getCurrentWindow } = await import('@tauri-apps/api/window')
      unlistenResized = await getCurrentWindow().onResized(scheduleSync)
    }

    void setup()
    return () => {
      cancelled = true
      if (timer !== undefined) clearTimeout(timer)
      unlistenResized?.()
    }
  }, [])

  const cls = ['window-host', maximized ? 'window-host--maximized' : '']
    .filter(Boolean)
    .join(' ')

  return <div className={cls}>{children}</div>
}
