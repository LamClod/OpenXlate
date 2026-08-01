import { useCallback, useEffect, useState } from 'react'
import {
  Check,
  Clipboard,
  Eye,
  EyeOff,
  LoaderCircle,
  Plus,
  Save,
  Trash2,
} from 'lucide-react'
import {
  api,
  isTauriRuntime,
  type GatewayStatus,
  type ProviderConfig,
  type ProviderFormat,
  type ProviderInput,
} from '../api/tauri'

type FormatDefinition = {
  id: ProviderFormat
  label: string
  localPath: string
  defaultBaseUrl: string
}

const FORMATS: FormatDefinition[] = [
  {
    id: 'openai',
    label: 'OpenAI Chat Completions',
    localPath: '/v1/chat/completions',
    defaultBaseUrl: 'https://api.openxlate.com',
  },
  {
    id: 'responses',
    label: 'OpenAI Responses',
    localPath: '/v1/responses',
    defaultBaseUrl: 'https://api.openxlate.com',
  },
  {
    id: 'anthropic',
    label: 'Anthropic Messages',
    localPath: '/v1/messages',
    defaultBaseUrl: 'https://api.openxlate.com',
  },
  {
    id: 'gemini',
    label: 'Gemini generateContent',
    localPath: '/v1beta/models/{model}:generateContent',
    defaultBaseUrl: 'https://api.openxlate.com',
  },
]

const LOCAL_PORT = 5150

function definitionFor(format: ProviderFormat) {
  return FORMATS.find((definition) => definition.id === format) ?? FORMATS[0]
}

function newProvider(format: ProviderFormat = 'openai'): ProviderInput {
  const definition = definitionFor(format)
  return {
    name: '',
    format,
    baseUrl: definition.defaultBaseUrl,
    apiKey: '',
    enabled: true,
  }
}

function toMessage(error: unknown) {
  return typeof error === 'string' ? error : error instanceof Error ? error.message : '操作失败，请重试。'
}

export function ProviderManagement() {
  const [providers, setProviders] = useState<ProviderConfig[]>([])
  const [draft, setDraft] = useState<ProviderConfig | ProviderInput>(newProvider)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [gateway, setGateway] = useState<GatewayStatus>({ running: false, port: LOCAL_PORT, error: null })
  const [loading, setLoading] = useState(isTauriRuntime())
  const [saving, setSaving] = useState(false)
  const [showKey, setShowKey] = useState(false)
  const [notice, setNotice] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const isNew = selectedId === null

  const refreshGateway = useCallback(async () => {
    if (!isTauriRuntime()) return
    try {
      setGateway(await api.getGatewayStatus())
    } catch (nextError) {
      setGateway({ running: false, port: LOCAL_PORT, error: toMessage(nextError) })
    }
  }, [])

  useEffect(() => {
    if (!isTauriRuntime()) return
    const load = async () => {
      try {
        const [nextProviders] = await Promise.all([api.listProviders(), refreshGateway()])
        setProviders(nextProviders)
        if (nextProviders[0]) {
          setSelectedId(nextProviders[0].id)
          setDraft(nextProviders[0])
        }
      } catch (nextError) {
        setError(toMessage(nextError))
      } finally {
        setLoading(false)
      }
    }
    void load()
    const timer = window.setInterval(() => void refreshGateway(), 5000)
    return () => window.clearInterval(timer)
  }, [refreshGateway])

  const updateDraft = <Key extends keyof ProviderInput>(key: Key, value: ProviderInput[Key]) => {
    setDraft((current) => ({ ...current, [key]: value }))
    setError(null)
    setNotice(null)
  }

  const selectProvider = (provider: ProviderConfig) => {
    setSelectedId(provider.id)
    setDraft(provider)
    setShowKey(false)
    setError(null)
    setNotice(null)
  }

  const addProvider = () => {
    setSelectedId(null)
    setDraft(newProvider())
    setShowKey(false)
    setError(null)
    setNotice(null)
  }

  const changeFormat = (format: ProviderFormat) => {
    const definition = definitionFor(format)
    setDraft((current) => ({
      ...current,
      format,
      baseUrl: definition.defaultBaseUrl,
    }))
    setError(null)
    setNotice(null)
  }

  const saveProvider = async () => {
    if (!draft.name.trim() || !draft.baseUrl.trim()) {
      setError('请填写供应商名称和上游地址。')
      return
    }
    if (draft.name.trim().includes('-')) {
      setError('供应商名称不能包含连字符 `-`，它用于拼接本地模型名。')
      return
    }
    setSaving(true)
    setError(null)
    try {
      const saved = isNew
        ? await api.createProvider(draft as ProviderInput)
        : await api.updateProvider(draft as ProviderConfig)
      setProviders((current) => {
        const index = current.findIndex((provider) => provider.id === saved.id)
        if (index === -1) return [...current, saved]
        return current.map((provider) => (provider.id === saved.id ? saved : provider))
      })
      setDraft(saved)
      setSelectedId(saved.id)
      setNotice('供应商已保存。')
    } catch (nextError) {
      setError(toMessage(nextError))
    } finally {
      setSaving(false)
    }
  }

  const toggleProviderEnabled = async (provider: ProviderConfig, enabled: boolean) => {
    if (!isTauriRuntime()) {
      setProviders((current) =>
        current.map((item) => (item.id === provider.id ? { ...item, enabled } : item)),
      )
      if (selectedId === provider.id) {
        setDraft((current) => ({ ...current, enabled }))
      }
      return
    }

    setError(null)
    try {
      const saved = await api.updateProvider({ ...provider, enabled })
      setProviders((current) =>
        current.map((item) => (item.id === saved.id ? saved : item)),
      )
      if (selectedId === saved.id) {
        setDraft(saved)
      }
      setNotice(enabled ? `已启用 ${saved.name}` : `已停用 ${saved.name}`)
    } catch (nextError) {
      setError(toMessage(nextError))
    }
  }

  const removeProvider = async () => {
    if (isNew || !('id' in draft)) return
    if (!window.confirm(`删除供应商“${draft.name}”？`)) return
    setSaving(true)
    setError(null)
    try {
      await api.deleteProvider(draft.id)
      const remaining = providers.filter((provider) => provider.id !== draft.id)
      setProviders(remaining)
      if (remaining[0]) {
        selectProvider(remaining[0])
      } else {
        addProvider()
      }
      setNotice('供应商已删除。')
    } catch (nextError) {
      setError(toMessage(nextError))
    } finally {
      setSaving(false)
    }
  }

  const copy = async (value: string, label: string) => {
    try {
      await navigator.clipboard.writeText(value)
      setNotice(`${label}已复制。`)
    } catch {
      setError('无法访问剪贴板。')
    }
  }

  const gatewayPort = gateway.port || LOCAL_PORT

  return (
    <main className="provider-management">
      <div className="provider-workspace" data-tauri-drag-region="false">
        <aside className="provider-list" aria-label="已配置供应商">
          <div className="provider-list-header" data-tauri-drag-region="false">
            <span>供应商</span>
            <button
              type="button"
              className="icon-button"
              onClick={addProvider}
              aria-label="新增供应商"
              data-tauri-drag-region="false"
            >
              <Plus size={17} aria-hidden />
            </button>
          </div>
          <div className="provider-list-items" data-tauri-drag-region="false">
            {loading ? (
              <div className="provider-list-loading"><LoaderCircle size={18} className="spin" />正在读取配置</div>
            ) : providers.length === 0 ? (
              <div className="provider-list-empty">还没有供应商配置</div>
            ) : (
              providers.map((provider) => (
                <div
                  key={provider.id}
                  role="button"
                  tabIndex={0}
                  className={`provider-list-item${provider.id === selectedId ? ' provider-list-item--active' : ''}${provider.enabled ? '' : ' provider-list-item--off'}`}
                  onClick={() => selectProvider(provider)}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter' || event.key === ' ') {
                      event.preventDefault()
                      selectProvider(provider)
                    }
                  }}
                >
                  <span className="provider-list-item-name">{provider.name || '未命名'}</span>
                  <label
                    className="enabled-control provider-list-switch"
                    onClick={(event) => event.stopPropagation()}
                    onKeyDown={(event) => event.stopPropagation()}
                  >
                    <input
                      type="checkbox"
                      checked={provider.enabled}
                      onChange={(event) => void toggleProviderEnabled(provider, event.target.checked)}
                      aria-label={`${provider.enabled ? '停用' : '启用'} ${provider.name || '供应商'}`}
                    />
                    <span className="switch" aria-hidden />
                  </label>
                </div>
              ))
            )}
          </div>
        </aside>

        <section className="provider-editor" aria-label={isNew ? '编辑新供应商' : '编辑供应商'}>
          <div className="provider-form-grid">
            <label className="field">
              <span>供应商名称</span>
              <input
                value={draft.name}
                onChange={(event) => updateDraft('name', event.target.value)}
                placeholder="例如 OpenAI（不可含 -）"
                autoComplete="off"
              />
            </label>
            <label className="field">
              <span>上游协议</span>
              <select value={draft.format} onChange={(event) => changeFormat(event.target.value as ProviderFormat)}>
                {FORMATS.map((format) => <option key={format.id} value={format.id}>{format.label}</option>)}
              </select>
            </label>
            <label className="field">
              <span>上游地址</span>
              <input value={draft.baseUrl} onChange={(event) => updateDraft('baseUrl', event.target.value)} placeholder="https://api.openxlate.com" spellCheck={false} autoComplete="url" />
            </label>
            <label className="field">
              <span>API Key</span>
              <span className="secret-field">
                <input type={showKey ? 'text' : 'password'} value={draft.apiKey} onChange={(event) => updateDraft('apiKey', event.target.value)} placeholder="可留空" autoComplete="off" />
                <button type="button" className="secret-toggle" onClick={() => setShowKey((visible) => !visible)} aria-label={showKey ? '隐藏 API Key' : '显示 API Key'}>
                  {showKey ? <EyeOff size={16} aria-hidden /> : <Eye size={16} aria-hidden />}
                </button>
              </span>
            </label>
          </div>

          <section className="gateway-endpoints" aria-labelledby="gateway-endpoints-title">
            <div className="gateway-endpoints-heading">
              <div>
                <p className="eyebrow">LOOPBACK ENDPOINTS</p>
                <h3 id="gateway-endpoints-title">本地接口</h3>
              </div>
              <button type="button" className="copy-address-button" onClick={() => void copy(`http://127.0.0.1:${gatewayPort}`, '本地地址')}>
                <Clipboard size={15} aria-hidden />
                复制地址
              </button>
            </div>
            <div className="endpoint-table">
              <div className="endpoint-row">
                <span>Models</span>
                <code>GET http://127.0.0.1:{gatewayPort}/v1/models</code>
              </div>
              {FORMATS.map((format) => (
                <div className="endpoint-row" key={format.id}>
                  <span>{format.label}</span>
                  <code>POST http://127.0.0.1:{gatewayPort}{format.localPath}</code>
                </div>
              ))}
            </div>
          </section>

          {gateway.error && <p className="feedback feedback--error">{gateway.error}</p>}
          {error && <p className="feedback feedback--error">{error}</p>}
          {notice && <p className="feedback feedback--success"><Check size={15} aria-hidden />{notice}</p>}

          <footer className="provider-editor-actions">
            {!isNew && (
              <button type="button" className="delete-button" onClick={() => void removeProvider()} disabled={saving}>
                <Trash2 size={16} aria-hidden />
                删除
              </button>
            )}
            <button type="button" className="save-button" onClick={() => void saveProvider()} disabled={saving}>
              {saving ? <LoaderCircle size={16} className="spin" aria-hidden /> : <Save size={16} aria-hidden />}
              保存供应商
            </button>
          </footer>
        </section>
      </div>
    </main>
  )
}


