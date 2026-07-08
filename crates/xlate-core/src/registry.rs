use dashmap::DashMap;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ModelType {
    TextGeneration,
    DeepThinking,
    VisionUnderstanding,
    ImageGeneration,
    TextEmbedding,
    SpeechSynthesis,
    Omni,
}

impl Default for ModelType {
    fn default() -> Self {
        Self::TextGeneration
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelMeta {
    pub id: String,
    #[serde(default)]
    pub display_name: String,
    #[serde(default)]
    pub vendor: String,
    #[serde(default)]
    pub model_type: ModelType,
    #[serde(default)]
    pub family: Option<String>,
    #[serde(default)]
    pub capabilities: ModelCapabilities,
    #[serde(default)]
    pub limits: ModelLimits,
    #[serde(default)]
    pub pricing_slug: Option<String>,
    #[serde(default)]
    pub deprecated: bool,
    #[serde(default)]
    pub thinking: ThinkingSupport,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelCapabilities {
    #[serde(default)]
    pub vision: bool,
    #[serde(default)]
    pub tool_use: bool,
    #[serde(default)]
    pub streaming: bool,
    #[serde(default)]
    pub image_generation: bool,
    #[serde(default)]
    pub pdf: bool,
    #[serde(default)]
    pub audio_input: bool,
    #[serde(default)]
    pub audio_output: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelLimits {
    #[serde(default)]
    pub context_window: Option<u32>,
    #[serde(default)]
    pub max_output_tokens: Option<u32>,
    #[serde(default)]
    pub max_images: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThinkingSupport {
    None,
    Optional,
    Mandatory { budget_min: u32 },
}

impl Default for ThinkingSupport {
    fn default() -> Self {
        Self::None
    }
}

pub struct ModelRegistry {
    models: DashMap<String, ModelMeta>,
    aliases: DashMap<String, String>,
}

impl ModelRegistry {
    pub fn new() -> Self {
        let reg = Self {
            models: DashMap::new(),
            aliases: DashMap::new(),
        };
        reg.load_builtin();
        reg
    }

    fn load_builtin(&self) {
        let vision_models = [
            ("gpt-4o", "openai", true),
            ("gpt-4o-*", "openai", true),
            ("gpt-4-turbo*", "openai", true),
            ("gpt-4.1*", "openai", true),
            ("gpt-5*", "openai", true),
            ("o1*", "openai", true),
            ("o3*", "openai", true),
            ("o4*", "openai", true),
            ("claude-*", "anthropic", true),
            ("gemini-*", "google", true),
        ];

        for (id, vendor, vision) in vision_models {
            self.models.insert(
                id.to_string(),
                ModelMeta {
                    id: id.to_string(),
                    display_name: id.to_string(),
                    vendor: vendor.to_string(),
                    model_type: ModelType::TextGeneration,
                    family: None,
                    capabilities: ModelCapabilities {
                        vision,
                        tool_use: true,
                        streaming: true,
                        ..Default::default()
                    },
                    limits: ModelLimits::default(),
                    pricing_slug: None,
                    deprecated: false,
                    thinking: ThinkingSupport::Optional,
                },
            );
        }
    }

    pub fn register(&self, meta: ModelMeta) {
        self.models.insert(meta.id.clone(), meta);
    }

    pub fn register_if_absent(&self, meta: ModelMeta) {
        self.models.entry(meta.id.clone()).or_insert(meta);
    }

    pub fn add_alias(&self, alias: String, target: String) {
        self.aliases.insert(alias, target);
    }

    pub fn resolve_alias(&self, model: &str) -> Option<String> {
        self.aliases.get(model).map(|v| v.clone())
    }

    pub fn get(&self, model: &str) -> Option<ModelMeta> {
        if let Some(meta) = self.models.get(model) {
            return Some(meta.clone());
        }
        for entry in self.models.iter() {
            if glob_match::glob_match(entry.key(), model) {
                return Some(entry.value().clone());
            }
        }
        None
    }

    pub fn supports_vision(&self, model: &str) -> bool {
        self.get(model)
            .map(|m| m.capabilities.vision)
            .unwrap_or(false)
    }

    pub fn supports_tools(&self, model: &str) -> bool {
        self.get(model)
            .map(|m| m.capabilities.tool_use)
            .unwrap_or(true)
    }

    pub fn context_window(&self, model: &str) -> Option<u32> {
        self.get(model).and_then(|m| m.limits.context_window)
    }

    pub fn list_models(&self) -> Vec<ModelMeta> {
        self.models.iter().map(|e| e.value().clone()).collect()
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}
