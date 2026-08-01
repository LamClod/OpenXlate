use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use rusqlite::{params, Connection};

use crate::codec::CodecFormat;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    pub theme: String,
    pub language: String,
    pub source_lang: String,
    pub target_lang: String,
    pub provider: String,
    pub api_key: String,
    pub api_base_url: String,
    pub model: String,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            theme: "system".to_string(),
            language: "zh".to_string(),
            source_lang: "auto".to_string(),
            target_lang: "en".to_string(),
            provider: "openai".to_string(),
            api_key: String::new(),
            api_base_url: String::new(),
            model: "gpt-4o".to_string(),
        }
    }
}

/// Provider identity only. Upstream models are discovered at runtime via
/// the provider's model list API and routed by `{name}-{upstream_model}`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderConfig {
    pub id: String,
    pub name: String,
    pub format: CodecFormat,
    pub base_url: String,
    pub api_key: String,
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderInput {
    pub name: String,
    pub format: CodecFormat,
    pub base_url: String,
    pub api_key: String,
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GatewayStatus {
    pub running: bool,
    pub port: u16,
    pub error: Option<String>,
}

impl GatewayStatus {
    pub const fn starting() -> Self {
        Self {
            running: false,
            port: 5150,
            error: None,
        }
    }
}

#[derive(Clone)]
pub struct GatewayState {
    pub providers: Arc<RwLock<Vec<ProviderConfig>>>,
    pub http_client: reqwest::Client,
    pub status: Arc<RwLock<GatewayStatus>>,
}

pub struct AppState {
    pub settings: RwLock<Settings>,
    providers: Arc<RwLock<Vec<ProviderConfig>>>,
    http_client: reqwest::Client,
    gateway_status: Arc<RwLock<GatewayStatus>>,
    provider_database_path: PathBuf,
}

impl AppState {
    pub fn new(provider_database_path: PathBuf) -> Result<Self, String> {
        initialize_database(&provider_database_path)?;
        let http_client = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(15))
            .timeout(std::time::Duration::from_secs(10 * 60))
            .build()
            .expect("failed to build HTTP client");

        Ok(Self {
            settings: RwLock::new(Settings::default()),
            providers: Arc::new(RwLock::new(load_providers(&provider_database_path)?)),
            http_client,
            gateway_status: Arc::new(RwLock::new(GatewayStatus::starting())),
            provider_database_path,
        })
    }

    pub fn http_client(&self) -> &reqwest::Client {
        &self.http_client
    }

    pub fn providers(&self) -> Result<Vec<ProviderConfig>, String> {
        self.providers
            .read()
            .map(|providers| providers.clone())
            .map_err(|error| error.to_string())
    }

    pub fn create_provider(&self, input: ProviderInput) -> Result<ProviderConfig, String> {
        validate_provider_input(&input)?;
        let provider = ProviderConfig {
            id: uuid::Uuid::new_v4().to_string(),
            name: input.name.trim().to_string(),
            format: input.format,
            base_url: input.base_url.trim_end_matches('/').to_string(),
            api_key: input.api_key.trim().to_string(),
            enabled: input.enabled,
        };

        insert_provider(&self.provider_database_path, &provider)?;
        self.providers
            .write()
            .map_err(|error| error.to_string())?
            .push(provider.clone());
        Ok(provider)
    }

    pub fn update_provider(&self, provider: ProviderConfig) -> Result<ProviderConfig, String> {
        validate_provider_input(&ProviderInput {
            name: provider.name.clone(),
            format: provider.format,
            base_url: provider.base_url.clone(),
            api_key: provider.api_key.clone(),
            enabled: provider.enabled,
        })?;
        if provider.id.trim().is_empty() {
            return Err("供应商 ID 不能为空".to_string());
        }

        let normalized = ProviderConfig {
            id: provider.id,
            name: provider.name.trim().to_string(),
            format: provider.format,
            base_url: provider.base_url.trim_end_matches('/').to_string(),
            api_key: provider.api_key.trim().to_string(),
            enabled: provider.enabled,
        };
        update_provider_record(&self.provider_database_path, &normalized)?;
        let mut providers = self.providers.write().map_err(|error| error.to_string())?;
        let Some(existing) = providers.iter_mut().find(|item| item.id == normalized.id) else {
            return Err("供应商缓存已过期；请重启应用后重试".to_string());
        };
        *existing = normalized.clone();
        Ok(normalized)
    }

    pub fn delete_provider(&self, id: &str) -> Result<(), String> {
        delete_provider_record(&self.provider_database_path, id)?;
        self.providers
            .write()
            .map_err(|error| error.to_string())?
            .retain(|provider| provider.id != id);
        Ok(())
    }

    pub fn gateway_state(&self) -> GatewayState {
        GatewayState {
            providers: Arc::clone(&self.providers),
            http_client: self.http_client.clone(),
            status: Arc::clone(&self.gateway_status),
        }
    }

    pub fn gateway_status(&self) -> Result<GatewayStatus, String> {
        self.gateway_status
            .read()
            .map(|status| status.clone())
            .map_err(|error| error.to_string())
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new(std::env::temp_dir().join("openxlate.db"))
            .expect("failed to initialize default OpenXlate database")
    }
}

fn initialize_database(path: &PathBuf) -> Result<(), String> {
    let mut connection = open_database(path)?;
    connection
        .execute_batch(
            "
            PRAGMA foreign_keys = ON;
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            ",
        )
        .map_err(database_error)?;

    let current_version: i64 = connection
        .query_row(
            "SELECT COALESCE(MAX(version), 0) FROM schema_migrations",
            [],
            |row| row.get(0),
        )
        .map_err(database_error)?;

    if current_version < 1 {
        let transaction = connection.transaction().map_err(database_error)?;
        transaction
            .execute_batch(
                "
                CREATE TABLE providers (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL CHECK (length(trim(name)) > 0),
                    local_model TEXT NOT NULL UNIQUE,
                    protocol TEXT NOT NULL CHECK (protocol IN ('openai', 'responses', 'anthropic', 'gemini')),
                    base_url TEXT NOT NULL,
                    api_key TEXT NOT NULL,
                    upstream_model TEXT NOT NULL CHECK (length(trim(upstream_model)) > 0),
                    enabled INTEGER NOT NULL DEFAULT 1 CHECK (enabled IN (0, 1)),
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX idx_providers_enabled_local_model ON providers (enabled, local_model);
                INSERT INTO schema_migrations (version) VALUES (1);
                ",
            )
            .map_err(database_error)?;
        transaction.commit().map_err(database_error)?;
    }

    let current_version: i64 = connection
        .query_row(
            "SELECT COALESCE(MAX(version), 0) FROM schema_migrations",
            [],
            |row| row.get(0),
        )
        .map_err(database_error)?;

    if current_version < 2 {
        let transaction = connection.transaction().map_err(database_error)?;
        // Drop fixed upstream_model / local_model routing. Provider name is the
        // unique routing prefix; models are discovered from the upstream list.
        transaction
            .execute_batch(
                "
                CREATE TABLE providers_v2 (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE CHECK (length(trim(name)) > 0),
                    protocol TEXT NOT NULL CHECK (protocol IN ('openai', 'responses', 'anthropic', 'gemini')),
                    base_url TEXT NOT NULL,
                    api_key TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1 CHECK (enabled IN (0, 1)),
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                INSERT INTO providers_v2 (id, name, protocol, base_url, api_key, enabled, created_at, updated_at)
                SELECT id, name, protocol, base_url, api_key, enabled, created_at, updated_at
                FROM providers;
                DROP TABLE providers;
                ALTER TABLE providers_v2 RENAME TO providers;
                CREATE INDEX idx_providers_enabled_name ON providers (enabled, name);
                INSERT INTO schema_migrations (version) VALUES (2);
                ",
            )
            .map_err(database_error)?;
        transaction.commit().map_err(database_error)?;
    }

    Ok(())
}

fn open_database(path: &PathBuf) -> Result<Connection, String> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("无法创建应用数据目录: {error}"))?;
    }
    let connection = Connection::open(path).map_err(database_error)?;
    connection
        .busy_timeout(std::time::Duration::from_secs(3))
        .map_err(database_error)?;
    Ok(connection)
}

fn load_providers(path: &PathBuf) -> Result<Vec<ProviderConfig>, String> {
    let connection = open_database(path)?;
    let mut statement = connection
        .prepare(
            "SELECT id, name, protocol, base_url, api_key, enabled
             FROM providers
             ORDER BY updated_at DESC, name COLLATE NOCASE",
        )
        .map_err(database_error)?;
    let rows = statement
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, i64>(5)?,
            ))
        })
        .map_err(database_error)?;

    let mut providers = Vec::new();
    for row in rows {
        let (id, name, format, base_url, api_key, enabled) = row.map_err(database_error)?;
        providers.push(ProviderConfig {
            id,
            name,
            format: parse_format(&format)?,
            base_url,
            api_key,
            enabled: enabled != 0,
        });
    }
    Ok(providers)
}

fn insert_provider(path: &PathBuf, provider: &ProviderConfig) -> Result<(), String> {
    let connection = open_database(path)?;
    connection
        .execute(
            "INSERT INTO providers (id, name, protocol, base_url, api_key, enabled)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                provider.id,
                provider.name,
                provider.format.as_str(),
                provider.base_url,
                provider.api_key,
                i64::from(provider.enabled),
            ],
        )
        .map_err(provider_database_error)?;
    Ok(())
}

fn update_provider_record(path: &PathBuf, provider: &ProviderConfig) -> Result<(), String> {
    let connection = open_database(path)?;
    let affected = connection
        .execute(
            "UPDATE providers
             SET name = ?1, protocol = ?2, base_url = ?3, api_key = ?4,
                 enabled = ?5, updated_at = CURRENT_TIMESTAMP
             WHERE id = ?6",
            params![
                provider.name,
                provider.format.as_str(),
                provider.base_url,
                provider.api_key,
                i64::from(provider.enabled),
                provider.id,
            ],
        )
        .map_err(provider_database_error)?;
    if affected == 0 {
        return Err("找不到要更新的供应商".to_string());
    }
    Ok(())
}

fn delete_provider_record(path: &PathBuf, id: &str) -> Result<(), String> {
    let connection = open_database(path)?;
    let affected = connection
        .execute("DELETE FROM providers WHERE id = ?1", params![id])
        .map_err(database_error)?;
    if affected == 0 {
        return Err("找不到要删除的供应商".to_string());
    }
    Ok(())
}

fn parse_format(value: &str) -> Result<CodecFormat, String> {
    value
        .parse()
        .map_err(|error| format!("SQLite 中存在未知的供应商协议 {value}: {error}"))
}

fn database_error(error: rusqlite::Error) -> String {
    format!("无法访问供应商数据库: {error}")
}

fn provider_database_error(error: rusqlite::Error) -> String {
    let message = error.to_string();
    if message.contains("providers.name") || message.contains("UNIQUE") {
        "供应商名称已存在；名称是本地路由前缀，必须唯一".to_string()
    } else {
        format!("无法保存供应商配置: {message}")
    }
}

fn validate_provider_input(input: &ProviderInput) -> Result<(), String> {
    if input.name.trim().is_empty() {
        return Err("供应商名称不能为空".to_string());
    }
    if input.name.trim().contains('-') {
        // Hyphen is the local routing separator: `{name}-{upstream_model}`.
        return Err("供应商名称不能包含连字符 `-`，它用于拼接本地模型名".to_string());
    }
    let base_url = input.base_url.trim();
    if !(base_url.starts_with("https://") || base_url.starts_with("http://")) {
        return Err("上游地址必须以 http:// 或 https:// 开头".to_string());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sqlite_persists_providers_and_rejects_duplicate_names() {
        let path =
            std::env::temp_dir().join(format!("openxlate-state-test-{}.db", uuid::Uuid::new_v4()));
        let input = ProviderInput {
            name: "OpenAI".into(),
            format: CodecFormat::OpenAi,
            base_url: "https://api.openai.com".into(),
            api_key: "test-key".into(),
            enabled: true,
        };

        let state = AppState::new(path.clone()).expect("database should initialize");
        state
            .create_provider(input.clone())
            .expect("provider should be inserted");
        drop(state);

        let reopened = AppState::new(path.clone()).expect("database should reopen");
        let providers = reopened.providers().expect("providers should load");
        assert_eq!(providers.len(), 1);
        assert_eq!(providers[0].name, "OpenAI");
        assert!(reopened
            .create_provider(input)
            .expect_err("duplicate name must fail")
            .contains("唯一"));
        drop(reopened);

        std::fs::remove_file(path).expect("temporary database should be removable");
    }
}
