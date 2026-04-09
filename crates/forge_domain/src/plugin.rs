//! Plugin manifest and runtime types.
//!
//! Forge plugins are directories that bundle skills, commands, agents, hooks
//! and MCP servers behind a single `plugin.json` manifest. The on-disk format
//! is intentionally compatible with Claude Code plugins so a directory copied
//! from `~/.claude/plugins/` into `~/forge/plugins/` will load without
//! modification.
//!
//! This module defines only the **data shapes** — the parsing, discovery and
//! enable/disable logic lives in `forge_repo::ForgePluginRepository` and
//! `forge_app::plugin_loader`.
//!
//! References:
//! - Claude Code manifest schema: `claude-code/src/utils/plugins/schemas.ts`
//! - Claude Code `LoadedPlugin` type: `claude-code/src/types/plugin.ts`

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::McpServerConfig;

/// Top-level plugin manifest, parsed from `plugin.json`.
///
/// All component fields are optional — a plugin may declare any subset of
/// `skills`, `commands`, `agents`, `hooks` and `mcpServers`. Unknown fields
/// are silently dropped (matching Claude Code's permissive parser) so future
/// schema additions don't break older Forge versions.
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct PluginManifest {
    /// Unique plugin name. Required by validation, but kept as `Option` here
    /// so deserialization of malformed manifests can produce a structured
    /// error message instead of a serde panic.
    #[serde(default)]
    pub name: Option<String>,

    /// Semver-style version string (e.g. `"1.2.3"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,

    /// Free-form short description shown in `:plugin list`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// Author information; accepts either a bare string or a structured
    /// object for compatibility with both `npm`-style and Claude Code
    /// manifests.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub author: Option<PluginAuthor>,

    /// Optional homepage URL.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub homepage: Option<String>,

    /// Optional repository URL.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repository: Option<String>,

    /// Optional SPDX license identifier.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,

    /// Free-form tags for plugin marketplaces.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub keywords: Vec<String>,

    /// Names of other plugins this plugin depends on. Phase 1 records the
    /// list but does not enforce ordering.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub dependencies: Vec<String>,

    /// Hook configuration: either a path to `hooks.json`, an inline object,
    /// or an array mixing both. Phase 3 will replace the placeholder body
    /// with typed hook definitions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hooks: Option<PluginHooksManifestField>,

    /// Path(s) to commands directory. When omitted the loader auto-detects
    /// `commands/` at the plugin root.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub commands: Option<PluginComponentPath>,

    /// Path(s) to agents directory. When omitted the loader auto-detects
    /// `agents/` at the plugin root.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agents: Option<PluginComponentPath>,

    /// Path(s) to skills directory. When omitted the loader auto-detects
    /// `skills/` at the plugin root.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub skills: Option<PluginComponentPath>,

    /// Inline MCP server definitions, keyed by server name. Merged into the
    /// global MCP manager during Phase 4 implementation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mcp_servers: Option<BTreeMap<String, McpServerConfig>>,
}

/// Author of a plugin.
///
/// Accepts both shorthand (`"Jane Doe"`) and verbose
/// (`{"name": "Jane Doe", "email": "..."}`) forms during deserialization.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PluginAuthor {
    /// Shorthand: just the author's display name.
    Name(String),
    /// Detailed form with optional email and homepage.
    Detailed {
        name: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        email: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        url: Option<String>,
    },
}

/// Component directory specification: either a single relative path or a
/// list of paths. The loader resolves each path relative to the plugin
/// root.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PluginComponentPath {
    /// Single relative path, e.g. `"./commands"`.
    Single(String),
    /// Multiple relative paths, e.g. `["./commands", "./extra-commands"]`.
    Multiple(Vec<String>),
}

impl PluginComponentPath {
    /// Returns the configured paths as a `Vec<&str>` for uniform iteration.
    pub fn as_paths(&self) -> Vec<&str> {
        match self {
            Self::Single(p) => vec![p.as_str()],
            Self::Multiple(ps) => ps.iter().map(String::as_str).collect(),
        }
    }
}

/// Hook configuration field on a plugin manifest.
///
/// Phase 1 only records the shape — actual hook execution lives in Phase 3.
/// The variants mirror Claude Code's `HooksField` schema:
///
/// - `Path`: relative path to a `hooks.json` file
/// - `Inline`: a hooks object directly inside the manifest
/// - `Array`: list mixing paths and inline objects (Claude Code uses this for
///   multi-file hook setups)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PluginHooksManifestField {
    /// Relative path to a `hooks.json` file.
    Path(String),
    /// Inline hooks configuration.
    Inline(PluginHooksConfig),
    /// Array of mixed paths and inline configs.
    Array(Vec<PluginHooksManifestField>),
}

/// Placeholder for parsed hook configuration.
///
/// Phase 1 only stores the raw JSON value so the manifest round-trips
/// without losing data. Phase 3 will replace `raw` with typed
/// `LifecycleHook` definitions.
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
pub struct PluginHooksConfig {
    /// Raw JSON value preserved verbatim until Phase 3 introduces the
    /// typed schema.
    #[serde(flatten)]
    pub raw: serde_json::Value,
}

/// Where a plugin was discovered. Used by the loader for precedence rules
/// (Project > Global > Builtin) and shown to the user in `:plugin list`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PluginSource {
    /// Discovered in `~/forge/plugins/`.
    Global,
    /// Discovered in `./.forge/plugins/` for the current workspace.
    Project,
    /// Loaded from a path supplied via `--plugin-dir` CLI flag.
    CliFlag,
    /// Compiled into the Forge binary.
    Builtin,
}

/// Runtime representation of a discovered plugin.
///
/// Built by `ForgePluginRepository::load_plugins()` and consumed by all
/// downstream subsystems (skills loader, hook chain, MCP manager, plugin CLI).
#[derive(Debug, Clone, PartialEq)]
pub struct LoadedPlugin {
    /// Effective plugin name. Falls back to the directory name if the
    /// manifest does not declare one.
    pub name: String,

    /// Parsed manifest. Always present even when the on-disk file was
    /// missing required fields — in that case the loader records the
    /// validation error and still returns a `LoadedPlugin` with sensible
    /// defaults so listing commands can show the broken plugin.
    pub manifest: PluginManifest,

    /// Absolute path to the plugin root directory.
    pub path: PathBuf,

    /// Where this plugin was discovered.
    pub source: PluginSource,

    /// Effective enabled state, after consulting `ForgeConfig.plugins`.
    pub enabled: bool,

    /// `true` for plugins compiled into the binary.
    pub is_builtin: bool,

    /// Resolved absolute paths to all commands directories. Either
    /// auto-detected as `<plugin>/commands/` or specified by
    /// `manifest.commands`.
    pub commands_paths: Vec<PathBuf>,

    /// Resolved absolute paths to all agents directories.
    pub agents_paths: Vec<PathBuf>,

    /// Resolved absolute paths to all skills directories.
    pub skills_paths: Vec<PathBuf>,

    /// Parsed hooks configuration, if the manifest referenced one and the
    /// file was readable. Phase 1 keeps this opaque; Phase 3 introduces a
    /// typed schema.
    pub hooks_config: Option<PluginHooksConfig>,

    /// MCP servers contributed by this plugin. Sourced from either
    /// `manifest.mcp_servers` or a sibling `.mcp.json` file.
    pub mcp_servers: Option<BTreeMap<String, McpServerConfig>>,
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn test_parse_minimal_manifest() {
        let json = r#"{ "name": "demo" }"#;
        let actual: PluginManifest = serde_json::from_str(json).unwrap();
        let expected = PluginManifest {
            name: Some("demo".to_string()),
            ..Default::default()
        };
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_parse_full_manifest() {
        let json = r#"{
            "name": "deploy-tools",
            "version": "0.1.0",
            "description": "Deployment helpers",
            "author": { "name": "Jane Doe", "email": "jane@example.com" },
            "homepage": "https://example.com",
            "repository": "https://github.com/example/deploy-tools",
            "license": "MIT",
            "keywords": ["deploy", "ops"],
            "dependencies": ["base-tools"],
            "commands": "./commands",
            "skills": ["./skills", "./extra-skills"]
        }"#;

        let actual: PluginManifest = serde_json::from_str(json).unwrap();

        assert_eq!(actual.name.as_deref(), Some("deploy-tools"));
        assert_eq!(actual.version.as_deref(), Some("0.1.0"));
        assert_eq!(actual.description.as_deref(), Some("Deployment helpers"));
        assert_eq!(actual.homepage.as_deref(), Some("https://example.com"));
        assert_eq!(actual.license.as_deref(), Some("MIT"));
        assert_eq!(actual.keywords, vec!["deploy", "ops"]);
        assert_eq!(actual.dependencies, vec!["base-tools"]);

        match actual.author {
            Some(PluginAuthor::Detailed { name, email, url }) => {
                assert_eq!(name, "Jane Doe");
                assert_eq!(email.as_deref(), Some("jane@example.com"));
                assert_eq!(url, None);
            }
            other => panic!("expected detailed author, got {other:?}"),
        }

        match actual.commands {
            Some(PluginComponentPath::Single(p)) => assert_eq!(p, "./commands"),
            other => panic!("expected single commands path, got {other:?}"),
        }

        match actual.skills {
            Some(PluginComponentPath::Multiple(ps)) => {
                assert_eq!(ps, vec!["./skills", "./extra-skills"]);
            }
            other => panic!("expected multiple skills paths, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_author_as_string() {
        let json = r#"{ "name": "demo", "author": "Jane Doe" }"#;
        let actual: PluginManifest = serde_json::from_str(json).unwrap();
        assert_eq!(actual.author, Some(PluginAuthor::Name("Jane Doe".into())));
    }

    #[test]
    fn test_parse_author_as_object() {
        let json = r#"{
            "name": "demo",
            "author": { "name": "Jane", "email": "j@x.com", "url": "https://x.com" }
        }"#;
        let actual: PluginManifest = serde_json::from_str(json).unwrap();
        match actual.author {
            Some(PluginAuthor::Detailed { name, email, url }) => {
                assert_eq!(name, "Jane");
                assert_eq!(email.as_deref(), Some("j@x.com"));
                assert_eq!(url.as_deref(), Some("https://x.com"));
            }
            other => panic!("expected detailed author, got {other:?}"),
        }
    }

    #[test]
    fn test_component_path_single() {
        let json = r#""./foo""#;
        let actual: PluginComponentPath = serde_json::from_str(json).unwrap();
        assert_eq!(actual, PluginComponentPath::Single("./foo".into()));
        assert_eq!(actual.as_paths(), vec!["./foo"]);
    }

    #[test]
    fn test_component_path_multiple() {
        let json = r#"["./a", "./b"]"#;
        let actual: PluginComponentPath = serde_json::from_str(json).unwrap();
        assert_eq!(
            actual,
            PluginComponentPath::Multiple(vec!["./a".into(), "./b".into()])
        );
        assert_eq!(actual.as_paths(), vec!["./a", "./b"]);
    }

    #[test]
    fn test_parse_hooks_field_path() {
        let json = r#"{ "name": "demo", "hooks": "hooks/hooks.json" }"#;
        let actual: PluginManifest = serde_json::from_str(json).unwrap();
        assert!(matches!(
            actual.hooks,
            Some(PluginHooksManifestField::Path(ref p)) if p == "hooks/hooks.json"
        ));
    }

    #[test]
    fn test_parse_hooks_field_inline() {
        let json = r#"{
            "name": "demo",
            "hooks": { "PreToolUse": [] }
        }"#;
        let actual: PluginManifest = serde_json::from_str(json).unwrap();
        assert!(matches!(
            actual.hooks,
            Some(PluginHooksManifestField::Inline(_))
        ));
    }

    #[test]
    fn test_parse_unknown_fields_are_ignored() {
        // Forward-compat: unknown manifest fields must not cause errors.
        let json = r#"{
            "name": "demo",
            "futureFeature": { "anything": [1, 2, 3] }
        }"#;
        let actual: PluginManifest = serde_json::from_str(json).unwrap();
        assert_eq!(actual.name.as_deref(), Some("demo"));
    }

    #[test]
    fn test_parse_malformed_json_returns_error() {
        let json = r#"{ "name": "demo", "#; // truncated
        let result: Result<PluginManifest, _> = serde_json::from_str(json);
        assert!(result.is_err());
    }
}
