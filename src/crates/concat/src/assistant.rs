//! Blocking clients for the in-app editing assistant.

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::path::Path;
use std::time::Duration;

/// Selects the assistant provider.
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Provider {
    /// Try the local model first and use Claude after a local failure when configured.
    #[default]
    Auto,
    /// Use the local Ollama service.
    Local,
    /// Use Anthropic's Claude API.
    Claude,
}

/// Provider settings for the assistant.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct Config {
    /// Provider selection.
    pub provider: Provider,
    /// Base URL of the Ollama service.
    pub ollama_url: String,
    /// Ollama model name.
    pub ollama_model: String,
    /// Claude model name.
    pub claude_model: String,
    /// Claude API key, when explicitly saved.
    pub claude_key: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            provider: Provider::Auto,
            ollama_url: "http://localhost:11434".into(),
            ollama_model: "qwen2.5:7b".into(),
            claude_model: "claude-sonnet-5-5".into(),
            claude_key: None,
        }
    }
}

impl Config {
    /// Loads `assistant.json`, falling back to defaults and the environment key.
    pub fn load(dir: &Path) -> Config {
        let mut config: Config = std::fs::read(dir.join("assistant.json"))
            .ok()
            .and_then(|bytes| serde_json::from_slice(&bytes).ok())
            .unwrap_or_default();
        if config.claude_key.as_deref().is_none_or(str::is_empty) {
            config.claude_key = std::env::var("ANTHROPIC_API_KEY")
                .ok()
                .filter(|key| !key.is_empty());
        }
        config
    }

    /// Saves settings without persisting an API key sourced from the environment.
    pub fn save(&self, dir: &Path) -> Result<(), String> {
        std::fs::create_dir_all(dir)
            .map_err(|error| format!("No se pudo crear la carpeta de configuración: {error}"))?;
        let mut saved = self.clone();
        if std::env::var("ANTHROPIC_API_KEY").ok().as_deref() == saved.claude_key.as_deref() {
            saved.claude_key = None;
        }
        let bytes = serde_json::to_vec_pretty(&saved)
            .map_err(|error| format!("No se pudo guardar la configuración: {error}"))?;
        std::fs::write(dir.join("assistant.json"), bytes)
            .map_err(|error| format!("No se pudo guardar la configuración: {error}"))
    }

    /// Returns whether a nonempty Claude key is available.
    pub fn has_claude(&self) -> bool {
        self.claude_key
            .as_deref()
            .is_some_and(|key| !key.is_empty())
    }
}

/// A tool offered to the model with its JSON Schema input definition.
#[derive(Clone, Debug)]
pub struct Tool {
    /// Tool identifier.
    pub name: String,
    /// Human-readable tool description.
    pub description: String,
    /// JSON Schema object describing tool input.
    pub schema: Value,
}

/// One text, tool invocation, or tool result in a conversation turn.
#[derive(Clone, Debug, PartialEq)]
pub enum Part {
    /// Plain text content.
    Text(String),
    /// A model request to execute a named tool.
    Call {
        id: String,
        name: String,
        input: Value,
    },
    /// Output returned by a tool execution.
    Result {
        id: String,
        name: String,
        content: String,
    },
}

/// The speaker of a conversation turn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    /// The user.
    User,
    /// The assistant model.
    Assistant,
}

/// A message turn in the conversation history.
#[derive(Clone, Debug, PartialEq)]
pub struct Turn {
    /// Speaker role.
    pub role: Role,
    /// Text and tool content in order.
    pub parts: Vec<Part>,
}

/// Final response and provider metadata.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Reply {
    /// Final assistant text.
    pub text: String,
    /// Provider that completed the run (`local` or `claude`).
    pub provider: &'static str,
    /// Number of provider requests made.
    pub steps: usize,
}

/// Builds a Claude Messages API request body.
pub fn claude_body(config: &Config, system: &str, history: &[Turn], tools: &[Tool]) -> Value {
    let messages: Vec<Value> = history.iter().map(|turn| json!({"role": match turn.role { Role::User => "user", Role::Assistant => "assistant" }, "content": turn.parts.iter().map(|part| match part {
        Part::Text(text) => json!({"type":"text","text":text}),
        Part::Call { id, name, input } => json!({"type":"tool_use","id":id,"name":name,"input":input}),
        Part::Result { id, content, .. } => json!({"type":"tool_result","tool_use_id":id,"content":content}),
    }).collect::<Vec<_>>() })).collect();
    json!({"model":config.claude_model,"max_tokens":1024,"system":system,"tools":tools.iter().map(|tool| json!({"name":tool.name,"description":tool.description,"input_schema":tool.schema})).collect::<Vec<_>>(),"messages":messages})
}

/// Parses a Claude Messages API response into an assistant turn.
pub fn parse_claude(value: &Value) -> Result<Turn, String> {
    if let Some(error) = value.get("error") {
        return Err(error
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or("Error desconocido de Claude")
            .to_owned());
    }
    let blocks = value
        .get("content")
        .and_then(Value::as_array)
        .ok_or_else(|| "Claude devolvió una respuesta sin contenido".to_owned())?;
    let mut parts = Vec::new();
    for block in blocks {
        match block.get("type").and_then(Value::as_str) {
            Some("text") => {
                if let Some(text) = block.get("text").and_then(Value::as_str) {
                    parts.push(Part::Text(text.to_owned()));
                }
            }
            Some("tool_use") => parts.push(Part::Call {
                id: block
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or("call_0")
                    .to_owned(),
                name: block
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_owned(),
                input: block.get("input").cloned().unwrap_or_else(|| json!({})),
            }),
            _ => {}
        }
    }
    Ok(Turn {
        role: Role::Assistant,
        parts,
    })
}

/// Builds an Ollama chat request body.
pub fn ollama_body(config: &Config, system: &str, history: &[Turn], tools: &[Tool]) -> Value {
    let mut messages = vec![json!({"role":"system","content":system})];
    for turn in history {
        for part in &turn.parts {
            match part {
                Part::Text(content) => messages.push(json!({"role":if turn.role == Role::User {"user"} else {"assistant"},"content":content})),
                Part::Call { name, input, .. } => messages.push(json!({"role":"assistant","content":"","tool_calls":[{"function":{"name":name,"arguments":input}}]})),
                Part::Result { name, content, .. } => messages.push(json!({"role":"tool","content":content,"tool_name":name})),
            }
        }
    }
    json!({"model":config.ollama_model,"stream":false,"options":{"temperature":0.2},"tools":tools.iter().map(|tool| json!({"type":"function","function":{"name":tool.name,"description":tool.description,"parameters":tool.schema}})).collect::<Vec<_>>(),"messages":messages})
}

/// Parses an Ollama chat response into an assistant turn.
pub fn parse_ollama(value: &Value) -> Result<Turn, String> {
    let message = value
        .get("message")
        .ok_or_else(|| "Ollama devolvió una respuesta sin mensaje".to_owned())?;
    let mut parts = Vec::new();
    if let Some(text) = message
        .get("content")
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty())
    {
        parts.push(Part::Text(text.to_owned()));
    }
    if let Some(calls) = message.get("tool_calls").and_then(Value::as_array) {
        for (index, call) in calls.iter().enumerate() {
            let function = call.get("function").unwrap_or(call);
            let input_value = function
                .get("arguments")
                .cloned()
                .unwrap_or_else(|| json!({}));
            let input = if let Some(raw) = input_value.as_str() {
                serde_json::from_str(raw)
                    .map_err(|error| format!("Argumentos JSON no válidos de Ollama: {error}"))?
            } else {
                input_value
            };
            parts.push(Part::Call {
                id: call
                    .get("id")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
                    .unwrap_or_else(|| format!("call_{index}")),
                name: function
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_owned(),
                input,
            });
        }
    }
    Ok(Turn {
        role: Role::Assistant,
        parts,
    })
}

/// Runs a blocking assistant conversation, executing requested tools as needed.
pub fn run(
    config: &Config,
    system: &str,
    history: &mut Vec<Turn>,
    tools: &[Tool],
    exec: &mut dyn FnMut(&str, &Value) -> String,
    max_steps: usize,
) -> Result<Reply, String> {
    let mut provider = match config.provider {
        Provider::Claude => "claude",
        Provider::Auto | Provider::Local => "local",
    };
    let mut steps = 0;
    for _ in 0..max_steps {
        steps += 1;
        let turn = step_with(
            &mut provider,
            config.provider == Provider::Auto && config.has_claude(),
            |selected| step(config, selected, system, history, tools),
        )?;
        let calls: Vec<(String, String, Value)> = turn
            .parts
            .iter()
            .filter_map(|part| {
                if let Part::Call { id, name, input } = part {
                    Some((id.clone(), name.clone(), input.clone()))
                } else {
                    None
                }
            })
            .collect();
        let final_text = turn
            .parts
            .iter()
            .filter_map(|part| {
                if let Part::Text(text) = part {
                    Some(text.as_str())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("\n");
        // An empty answer is not kept: Claude refuses a turn with no content.
        if turn.parts.is_empty() {
            return Ok(Reply {
                text: final_text,
                provider,
                steps,
            });
        }
        history.push(turn);
        if calls.is_empty() {
            return Ok(Reply {
                text: final_text,
                provider,
                steps,
            });
        }
        history.push(Turn {
            role: Role::User,
            parts: calls
                .iter()
                .map(|(id, name, input)| Part::Result {
                    id: id.clone(),
                    name: name.clone(),
                    content: exec(name, input),
                })
                .collect(),
        });
    }
    Err("La petición necesitó demasiados pasos: pídela por partes, una cosa cada vez".to_owned())
}

fn step_with<T>(
    provider: &mut &'static str,
    can_fallback: bool,
    mut call: impl FnMut(&str) -> Result<T, String>,
) -> Result<T, String> {
    match call(provider) {
        Ok(value) => Ok(value),
        Err(_) if *provider == "local" && can_fallback => {
            *provider = "claude";
            call(provider)
        }
        Err(error) => Err(error),
    }
}

fn step(
    config: &Config,
    provider: &str,
    system: &str,
    history: &[Turn],
    tools: &[Tool],
) -> Result<Turn, String> {
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(5))
        .timeout_read(Duration::from_secs(180))
        .build();
    let (url, body) = if provider == "claude" {
        (
            "https://api.anthropic.com/v1/messages".to_owned(),
            claude_body(config, system, history, tools),
        )
    } else {
        (
            format!("{}/api/chat", config.ollama_url.trim_end_matches('/')),
            ollama_body(config, system, history, tools),
        )
    };
    let mut request = agent.post(&url).set("content-type", "application/json");
    if provider == "claude" {
        request = request
            .set("x-api-key", config.claude_key.as_deref().unwrap_or(""))
            .set("anthropic-version", "2023-06-01");
    }
    let response = request
        .send_string(&body.to_string())
        .map_err(|error| provider_error(provider, &error.to_string()))?;
    let raw = response
        .into_string()
        .map_err(|error| format!("No se pudo leer la respuesta del proveedor: {error}"))?;
    let value: Value = serde_json::from_str(&raw)
        .map_err(|error| format!("No se pudo interpretar la respuesta del proveedor: {error}"))?;
    if provider == "claude" {
        parse_claude(&value)
    } else {
        parse_ollama(&value)
    }
}

fn provider_error(provider: &str, error: &str) -> String {
    if provider == "claude" && (error.contains("401") || error.contains("Unauthorized")) {
        "La clave de Claude no es válida".to_owned()
    } else if provider == "local" {
        "No encuentro el modelo local: instala Ollama (ollama.com) y ejecuta `ollama pull qwen2.5:7b`, o añade una clave de Claude.".to_owned()
    } else {
        format!("No se pudo conectar con Claude: {error}. Revisa la conexión y la clave de API.")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn config() -> Config {
        Config::default()
    }
    fn tool() -> Tool {
        Tool {
            name: "seek".into(),
            description: "Seek the playhead".into(),
            schema: json!({"type":"object"}),
        }
    }
    fn history() -> Vec<Turn> {
        vec![Turn {
            role: Role::User,
            parts: vec![Part::Text("Find the goal".into())],
        }]
    }

    #[test]
    fn bodies_represent_tool_round_trip() {
        let mut turns = history();
        turns.push(Turn {
            role: Role::Assistant,
            parts: vec![
                Part::Text("Looking".into()),
                Part::Call {
                    id: "c1".into(),
                    name: "seek".into(),
                    input: json!({"seconds":4}),
                },
            ],
        });
        turns.push(Turn {
            role: Role::User,
            parts: vec![Part::Result {
                id: "c1".into(),
                name: "seek".into(),
                content: "done".into(),
            }],
        });
        let claude = claude_body(&config(), "system", &turns, &[tool()]);
        assert_eq!(claude["messages"][1]["content"][1]["type"], "tool_use");
        assert_eq!(claude["messages"][2]["content"][0]["tool_use_id"], "c1");
        let ollama = ollama_body(&config(), "system", &turns, &[tool()]);
        assert_eq!(
            ollama["messages"][2]["tool_calls"][0]["function"]["name"],
            "seek"
        );
        assert_eq!(ollama["messages"][3]["tool_name"], "seek");
    }

    #[test]
    fn parses_provider_text_and_calls() {
        assert_eq!(
            parse_claude(&json!({"content":[{"type":"text","text":"ok"}]}))
                .unwrap()
                .parts,
            vec![Part::Text("ok".into())]
        );
        assert_eq!(
            parse_claude(
                &json!({"content":[{"type":"tool_use","id":"id","name":"seek","input":{"x":1}}]})
            )
            .unwrap()
            .parts[0],
            Part::Call {
                id: "id".into(),
                name: "seek".into(),
                input: json!({"x":1})
            }
        );
        assert_eq!(
            parse_ollama(&json!({"message":{"content":"ok"}}))
                .unwrap()
                .parts,
            vec![Part::Text("ok".into())]
        );
        assert_eq!(parse_ollama(&json!({"message":{"tool_calls":[{"function":{"name":"seek","arguments":"{\"x\":1}"}}]}})).unwrap().parts[0], Part::Call { id:"call_0".into(), name:"seek".into(), input:json!({"x":1}) });
    }

    #[test]
    fn config_load_defaults_and_round_trip() {
        let dir = std::env::temp_dir().join(format!("concat-assistant-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(Config::load(&dir).ollama_model, "qwen2.5:7b");
        let config = Config {
            provider: Provider::Claude,
            claude_key: Some("test-key".into()),
            ..Config::default()
        };
        config.save(&dir).unwrap();
        let loaded = Config::load(&dir);
        assert_eq!(loaded.provider, Provider::Claude);
        assert_eq!(loaded.claude_key.as_deref(), Some("test-key"));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn auto_fallback_seam_reuses_step_call() {
        let calls = AtomicUsize::new(0);
        let mut provider = "local";
        let reply = step_with(&mut provider, true, |name| {
            calls.fetch_add(1, Ordering::Relaxed);
            if name == "local" {
                Err("offline".to_owned())
            } else {
                Ok(7)
            }
        });
        assert_eq!(reply, Ok(7));
        assert_eq!(provider, "claude");
        assert_eq!(calls.load(Ordering::Relaxed), 2);
    }
}
