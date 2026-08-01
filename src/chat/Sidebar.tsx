import { Building2 } from 'lucide-react'

/** 左侧页面导航：与业务区同款圆角矩形栏，不参与右栏 chat。 */
export function Sidebar() {
  return (
    <nav className="sidebar" aria-label="页面导航" data-tauri-drag-region="false">
      <div className="sidebar-capsule">
        <button
          type="button"
          className="sidebar-item sidebar-item--active"
          aria-current="page"
          aria-label="供应商管理"
        >
          <Building2 size={17} strokeWidth={1.9} aria-hidden />
        </button>
      </div>
    </nav>
  )
}
