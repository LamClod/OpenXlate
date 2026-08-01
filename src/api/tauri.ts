import { invoke } from '@tauri-apps/api/core'
import { listen as tauriListen } from '@tauri-apps/api/event'

export type ProviderFormat = 'openai' | 'responses' | 'anthropic' | 'gemini'

export type ProviderConfig = {
  id: string
  name: string
  format: ProviderFormat
  baseUrl: string
  apiKey: string
  enabled: boolean
}

export type ProviderInput = Omit<ProviderConfig, 'id'>

export type GatewayStatus = {
  running: boolean
  port: number
  error: string | null
}

export function isTauriRuntime(): boolean {
  return !!(window as unknown as Record<string, unknown>).__TAURI_INTERNALS__
}

export const api = {
  async translateText(text: string, sourceLang: string, targetLang: string): Promise<string> {
    return invoke<string>('translate_text', { text, sourceLang, targetLang })
  },

  async getSettings(): Promise<Record<string, unknown>> {
    return invoke<Record<string, unknown>>('get_settings')
  },

  async updateSettings(settings: Record<string, unknown>): Promise<void> {
    return invoke<void>('update_settings', { settings })
  },

  async listProviders(): Promise<ProviderConfig[]> {
    return invoke<ProviderConfig[]>('list_providers')
  },

  async createProvider(input: ProviderInput): Promise<ProviderConfig> {
    return invoke<ProviderConfig>('create_provider', { input })
  },

  async updateProvider(provider: ProviderConfig): Promise<ProviderConfig> {
    return invoke<ProviderConfig>('update_provider', { provider })
  },

  async deleteProvider(id: string): Promise<void> {
    return invoke<void>('delete_provider', { id })
  },

  async getGatewayStatus(): Promise<GatewayStatus> {
    return invoke<GatewayStatus>('get_gateway_status')
  },

  async detectLanguage(text: string): Promise<string> {
    return invoke<string>('detect_language', { text })
  },

  async getSupportedLanguages(): Promise<Array<{ code: string; name: string }>> {
    return invoke<Array<{ code: string; name: string }>>('get_supported_languages')
  },

  async resizeWindow(width: number, height: number): Promise<void> {
    if (!isTauriRuntime()) return
    const { getCurrentWindow } = await import('@tauri-apps/api/window')
    const { LogicalSize } = await import('@tauri-apps/api/dpi')
    await getCurrentWindow().setSize(new LogicalSize(width, height))
  },

  async closeWindow(): Promise<void> {
    if (!isTauriRuntime()) return
    const { getCurrentWindow } = await import('@tauri-apps/api/window')
    await getCurrentWindow().hide()
  },

  async minimizeWindow(): Promise<void> {
    if (!isTauriRuntime()) return
    const { getCurrentWindow } = await import('@tauri-apps/api/window')
    await getCurrentWindow().minimize()
  },

  async toggleMaximizeWindow(): Promise<void> {
    if (!isTauriRuntime()) return
    const { getCurrentWindow } = await import('@tauri-apps/api/window')
    await getCurrentWindow().toggleMaximize()
  },

  async showWindow(): Promise<void> {
    if (!isTauriRuntime()) return
    const { getCurrentWindow } = await import('@tauri-apps/api/window')
    await getCurrentWindow().show()
  },

  async focusWindow(): Promise<void> {
    if (!isTauriRuntime()) return
    const { getCurrentWindow } = await import('@tauri-apps/api/window')
    await getCurrentWindow().setFocus()
  },
}

export { tauriListen as listen }
