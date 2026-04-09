use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Context as _;
use forge_app::domain::{
    LoadedPlugin, McpServerConfig, PluginComponentPath, PluginHooksConfig, PluginHooksManifestField,
    PluginLoadError, PluginLoadResult, PluginManifest, PluginRepository, PluginSource,
};
use forge_app::{DirectoryReaderInfra, EnvironmentInfra, FileInfoInfra, FileReaderInfra};
use forge_config::PluginSetting;
use futures::future::join_all;

/// Forge implementation of [`PluginRepository`].
///
/// Discovers plugins by scanning two directories:
///
/// 1. **Global**: `~/forge/plugins/<plugin>/` (from `Environment::plugin_path`)
/// 2. **Project-local**: `./.forge/plugins/<plugin>/` (from
///    `Environment::plugin_cwd_path`)
///
/// For each subdirectory the loader probes for a manifest file in priority
/// order:
///
/// 1. `<plugin>/.forge-plugin/plugin.json` (Forge-native marker)
/// 2. `<plugin>/.claude-plugin/plugin.json` (Claude Code 1:1 compatibility)
/// 3. `<plugin>/plugin.json` (legacy/bare)
///
/// When more than one marker is present the loader prefers the Forge-native
/// one and emits a `tracing::warn` to flag the ambiguity.
///
/// ## Precedence
///
/// When the same plugin name appears in both directories, the project-local
/// copy wins. This mirrors `claude-code/src/utils/plugins/pluginLoader.ts`
/// which gives workspace-scoped plugins precedence over global ones.
///
/// ## Component path resolution
///
/// Manifest fields `commands`, `agents` and `skills` are optional. If a
/// manifest omits them, the loader auto-detects sibling directories named
/// `commands/`, `agents/` and `skills/` at the plugin root. Manifest values
/// always take precedence over auto-detection — even when they point to a
/// non-existent path (so the user notices the typo).
///
/// ## MCP servers
///
/// MCP server definitions can come from either `manifest.mcp_servers`
/// (inline) or a sibling `.mcp.json` file at the plugin root. When both
/// are present they are merged with the inline manifest entries winning.
///
/// ## Error handling
///
/// Per-plugin failures (malformed JSON, missing required fields, unreadable
/// `hooks.json`) are logged via `tracing::warn` and the plugin is skipped.
/// Top-level filesystem errors (e.g. permission denied on the parent
/// directory) bubble up. Discovery never fails the whole CLI startup just
/// because one plugin is broken.
pub struct ForgePluginRepository<I> {
    infra: Arc<I>,
}

impl<I> ForgePluginRepository<I> {
    pub fn new(infra: Arc<I>) -> Self {
        Self { infra }
    }
}

#[async_trait::async_trait]
impl<I> PluginRepository for ForgePluginRepository<I>
where
    I: EnvironmentInfra<Config = forge_config::ForgeConfig>
        + FileReaderInfra
        + FileInfoInfra
        + DirectoryReaderInfra,
{
    async fn load_plugins(&self) -> anyhow::Result<Vec<LoadedPlugin>> {
        // Delegate to the error-surfacing variant and discard the diagnostic
        // tail so existing call sites keep their old signature.
        self.load_plugins_with_errors().await.map(|r| r.plugins)
    }

    async fn load_plugins_with_errors(&self) -> anyhow::Result<PluginLoadResult> {
        let env = self.infra.get_environment();
        let config = self.infra.get_config().ok();
        let plugin_settings: BTreeMap<String, PluginSetting> =
            config.and_then(|cfg| cfg.plugins).unwrap_or_default();

        // Scan global and project-local plugin roots in parallel.
        let global_root = env.plugin_path();
        let project_root = env.plugin_cwd_path();

        let (global, project) = futures::future::join(
            self.scan_root(&global_root, PluginSource::Global),
            self.scan_root(&project_root, PluginSource::Project),
        )
        .await;

        let (mut plugins, mut errors): (Vec<LoadedPlugin>, Vec<PluginLoadError>) =
            (Vec::new(), Vec::new());

        let (global_plugins, global_errors) = global?;
        plugins.extend(global_plugins);
        errors.extend(global_errors);

        let (project_plugins, project_errors) = project?;
        plugins.extend(project_plugins);
        errors.extend(project_errors);

        // Apply Project > Global precedence: a later (project) entry with the
        // same name replaces the earlier (global) one.
        let plugins = resolve_plugin_conflicts(plugins);

        // Apply enabled overrides from .forge.toml.
        let plugins = plugins
            .into_iter()
            .map(|mut plugin| {
                if let Some(setting) = plugin_settings.get(&plugin.name) {
                    plugin.enabled = setting.enabled;
                }
                plugin
            })
            .collect();

        Ok(PluginLoadResult { plugins, errors })
    }
}

impl<I> ForgePluginRepository<I>
where
    I: FileReaderInfra + FileInfoInfra + DirectoryReaderInfra,
{
    /// Scans a single root directory and returns all plugins discovered
    /// underneath along with any per-plugin load errors.
    ///
    /// Subdirectories without a recognised manifest file are silently
    /// skipped. Malformed manifests (unreadable, bad JSON, missing fields)
    /// are logged via `tracing::warn` for immediate operator visibility
    /// and also accumulated into the returned error vector so the Phase 9
    /// `:plugin list` command can surface them to the user.
    async fn scan_root(
        &self,
        root: &Path,
        source: PluginSource,
    ) -> anyhow::Result<(Vec<LoadedPlugin>, Vec<PluginLoadError>)> {
        if !self.infra.exists(root).await? {
            return Ok((Vec::new(), Vec::new()));
        }

        let entries = self
            .infra
            .list_directory_entries(root)
            .await
            .with_context(|| format!("Failed to list plugin root: {}", root.display()))?;

        let load_futs = entries
            .into_iter()
            .filter(|(_, is_dir)| *is_dir)
            .map(|(path, _)| {
                let infra = Arc::clone(&self.infra);
                let source_copy = source;
                async move {
                    let result = load_one_plugin(infra, path.clone(), source_copy).await;
                    (path, result)
                }
            });

        let results = join_all(load_futs).await;
        let mut plugins = Vec::new();
        let mut errors = Vec::new();
        for (path, res) in results {
            match res {
                Ok(Some(plugin)) => plugins.push(plugin),
                Ok(None) => {}
                Err(e) => {
                    tracing::warn!("Failed to load plugin: {e:#}");
                    // Capture the directory name (if any) as a best-effort
                    // plugin identifier; callers render this alongside the
                    // error message in `:plugin list`.
                    let plugin_name = path
                        .file_name()
                        .and_then(|s| s.to_str())
                        .map(String::from);
                    errors.push(PluginLoadError {
                        plugin_name,
                        path,
                        error: format!("{e:#}"),
                    });
                }
            }
        }

        Ok((plugins, errors))
    }
}

/// Loads a single plugin directory.
///
/// Returns:
/// - `Ok(Some(plugin))` when a manifest was found and parsed successfully
/// - `Ok(None)` when no manifest is present (the directory is not a plugin)
/// - `Err(_)` when a manifest was found but could not be parsed
async fn load_one_plugin<I>(
    infra: Arc<I>,
    plugin_dir: PathBuf,
    source: PluginSource,
) -> anyhow::Result<Option<LoadedPlugin>>
where
    I: FileReaderInfra + FileInfoInfra + DirectoryReaderInfra,
{
    let manifest_path = match find_manifest(&infra, &plugin_dir).await? {
        Some(path) => path,
        None => return Ok(None),
    };

    let raw = infra
        .read_utf8(&manifest_path)
        .await
        .with_context(|| format!("Failed to read manifest: {}", manifest_path.display()))?;

    let manifest: PluginManifest = serde_json::from_str(&raw)
        .with_context(|| format!("Failed to parse manifest: {}", manifest_path.display()))?;

    let dir_name = plugin_dir
        .file_name()
        .and_then(|s| s.to_str())
        .map(String::from)
        .unwrap_or_else(|| "<unknown>".to_string());

    let name = manifest.name.clone().unwrap_or_else(|| dir_name.clone());

    // Resolve component paths.
    let commands_paths =
        resolve_component_dirs(&infra, &plugin_dir, manifest.commands.as_ref(), "commands").await;
    let agents_paths =
        resolve_component_dirs(&infra, &plugin_dir, manifest.agents.as_ref(), "agents").await;
    let skills_paths =
        resolve_component_dirs(&infra, &plugin_dir, manifest.skills.as_ref(), "skills").await;

    // Resolve hooks config: either inline (Phase 3 will deserialize the body)
    // or from a file path. Phase 1 only stores the raw shape.
    let hooks_config = resolve_hooks_config(&infra, &plugin_dir, manifest.hooks.as_ref()).await;

    // Resolve MCP servers: merge inline manifest entries with sibling .mcp.json
    // when present.
    let mcp_servers = resolve_mcp_servers(&infra, &plugin_dir, &manifest).await;

    Ok(Some(LoadedPlugin {
        name,
        manifest,
        path: plugin_dir,
        source,
        // Plugins are enabled by default; the caller will apply ForgeConfig
        // overrides afterwards.
        enabled: true,
        is_builtin: false,
        commands_paths,
        agents_paths,
        skills_paths,
        hooks_config,
        mcp_servers,
    }))
}

/// Locates the manifest file inside a plugin directory.
///
/// Probes in priority order:
/// 1. `.forge-plugin/plugin.json`
/// 2. `.claude-plugin/plugin.json`
/// 3. `plugin.json`
///
/// When more than one marker is present, the function returns the
/// highest-priority match and logs a warning so the user is aware of the
/// ambiguity.
async fn find_manifest<I>(infra: &Arc<I>, plugin_dir: &Path) -> anyhow::Result<Option<PathBuf>>
where
    I: FileInfoInfra,
{
    let candidates = [
        plugin_dir.join(".forge-plugin").join("plugin.json"),
        plugin_dir.join(".claude-plugin").join("plugin.json"),
        plugin_dir.join("plugin.json"),
    ];

    let mut found = Vec::new();
    for path in &candidates {
        if infra.exists(path).await? {
            found.push(path.clone());
        }
    }

    if found.len() > 1 {
        tracing::warn!(
            "Plugin {} has multiple manifest files; using {} (other candidates: {:?})",
            plugin_dir.display(),
            found[0].display(),
            &found[1..]
        );
    }

    Ok(found.into_iter().next())
}

/// Resolves a component directory list (`commands`, `agents`, `skills`).
///
/// When the manifest declared explicit paths, those win even if they point
/// to non-existent directories — the user gets a chance to see the typo via
/// follow-up validation. When the manifest is silent, the auto-detected
/// `<plugin>/<default_name>/` is returned only if it exists on disk.
async fn resolve_component_dirs<I>(
    infra: &Arc<I>,
    plugin_dir: &Path,
    declared: Option<&PluginComponentPath>,
    default_name: &str,
) -> Vec<PathBuf>
where
    I: FileInfoInfra,
{
    if let Some(spec) = declared {
        return spec
            .as_paths()
            .into_iter()
            .map(|p| plugin_dir.join(p))
            .collect();
    }

    let auto = plugin_dir.join(default_name);
    match infra.exists(&auto).await {
        Ok(true) => vec![auto],
        _ => Vec::new(),
    }
}

/// Resolves a hooks manifest field into a [`PluginHooksConfig`].
///
/// Phase 1 only loads the raw JSON value so the manifest round-trips. Phase 3
/// will replace the body with typed hook definitions and stricter validation.
async fn resolve_hooks_config<I>(
    infra: &Arc<I>,
    plugin_dir: &Path,
    declared: Option<&PluginHooksManifestField>,
) -> Option<PluginHooksConfig>
where
    I: FileReaderInfra + FileInfoInfra,
{
    let field = declared?;

    match field {
        PluginHooksManifestField::Inline(cfg) => Some(cfg.clone()),
        PluginHooksManifestField::Path(rel) => {
            let abs = plugin_dir.join(rel);
            load_hooks_file(infra, &abs).await
        }
        PluginHooksManifestField::Array(items) => {
            // Merge all referenced configs by concatenating their raw values
            // into a JSON array. Phase 3 will replace this with proper
            // structural merging.
            let mut merged: Vec<serde_json::Value> = Vec::new();
            for item in items {
                let nested = Box::pin(resolve_hooks_config(infra, plugin_dir, Some(item))).await;
                if let Some(cfg) = nested {
                    merged.push(cfg.raw);
                }
            }
            Some(PluginHooksConfig { raw: serde_json::Value::Array(merged) })
        }
    }
}

async fn load_hooks_file<I>(infra: &Arc<I>, path: &Path) -> Option<PluginHooksConfig>
where
    I: FileReaderInfra + FileInfoInfra,
{
    match infra.exists(path).await {
        Ok(true) => {}
        Ok(false) => {
            tracing::warn!("Plugin hooks file not found: {}", path.display());
            return None;
        }
        Err(e) => {
            tracing::warn!("Failed to stat plugin hooks file {}: {e:#}", path.display());
            return None;
        }
    }

    match infra.read_utf8(path).await {
        Ok(raw) => match serde_json::from_str::<serde_json::Value>(&raw) {
            Ok(value) => Some(PluginHooksConfig { raw: value }),
            Err(e) => {
                tracing::warn!(
                    "Plugin hooks file {} is not valid JSON: {e:#}",
                    path.display()
                );
                None
            }
        },
        Err(e) => {
            tracing::warn!("Failed to read plugin hooks file {}: {e:#}", path.display());
            None
        }
    }
}

/// Resolves MCP server definitions for a plugin.
///
/// Inline manifest entries always win over `.mcp.json`. The merge is shallow:
/// for each server name only one definition is kept.
async fn resolve_mcp_servers<I>(
    infra: &Arc<I>,
    plugin_dir: &Path,
    manifest: &PluginManifest,
) -> Option<BTreeMap<String, McpServerConfig>>
where
    I: FileReaderInfra + FileInfoInfra,
{
    let mut merged: BTreeMap<String, McpServerConfig> = BTreeMap::new();

    // Sibling .mcp.json contributes first.
    let sidecar = plugin_dir.join(".mcp.json");
    if matches!(infra.exists(&sidecar).await, Ok(true))
        && let Ok(raw) = infra.read_utf8(&sidecar).await
    {
        // .mcp.json typically wraps servers under "mcpServers". Try that
        // shape first; fall back to a bare map for compat with simpler
        // hand-written files.
        #[derive(serde::Deserialize)]
        struct McpJsonFile {
            #[serde(default, alias = "mcp_servers")]
            mcp_servers: BTreeMap<String, McpServerConfig>,
        }

        if let Ok(parsed) = serde_json::from_str::<McpJsonFile>(&raw) {
            merged.extend(parsed.mcp_servers);
        } else if let Ok(bare) = serde_json::from_str::<BTreeMap<String, McpServerConfig>>(&raw) {
            merged.extend(bare);
        } else {
            tracing::warn!(
                "Plugin .mcp.json {} is not valid: ignored",
                sidecar.display()
            );
        }
    }

    // Inline manifest entries override sidecar entries with the same key.
    if let Some(inline) = &manifest.mcp_servers {
        for (name, cfg) in inline {
            merged.insert(name.clone(), cfg.clone());
        }
    }

    if merged.is_empty() {
        None
    } else {
        Some(merged)
    }
}

/// Resolves duplicate plugin names by keeping the *last* occurrence.
///
/// Because [`ForgePluginRepository::load_plugins`] pushes global plugins
/// before project-local ones, "last wins" implements the documented
/// `Project > Global` precedence.
fn resolve_plugin_conflicts(plugins: Vec<LoadedPlugin>) -> Vec<LoadedPlugin> {
    let mut seen: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut result: Vec<LoadedPlugin> = Vec::new();

    for plugin in plugins {
        if let Some(idx) = seen.get(&plugin.name) {
            result[*idx] = plugin;
        } else {
            seen.insert(plugin.name.clone(), result.len());
            result.push(plugin);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use forge_app::domain::PluginSource;
    use forge_config::ForgeConfig;
    use forge_infra::ForgeInfra;
    use pretty_assertions::assert_eq;

    use super::*;

    fn fixture_plugin(name: &str, source: PluginSource) -> LoadedPlugin {
        LoadedPlugin {
            name: name.to_string(),
            manifest: PluginManifest { name: Some(name.to_string()), ..Default::default() },
            path: PathBuf::from("/fake").join(name),
            source,
            enabled: true,
            is_builtin: false,
            commands_paths: Vec::new(),
            agents_paths: Vec::new(),
            skills_paths: Vec::new(),
            hooks_config: None,
            mcp_servers: None,
        }
    }

    /// Builds a real [`ForgePluginRepository`] backed by [`ForgeInfra`].
    ///
    /// Mirrors the `fixture_skill_repo` helper in `src/skill.rs`: we use the
    /// production infra (real filesystem I/O) because `scan_root` probes
    /// nested directories, checks manifest markers and reads JSON, and
    /// replicating that semantics in a fake is tedious and error-prone.
    fn fixture_plugin_repo() -> ForgePluginRepository<ForgeInfra> {
        let config = ForgeConfig::read().unwrap_or_default();
        let services_url = config.services_url.parse().unwrap();
        let infra = Arc::new(ForgeInfra::new(
            std::env::current_dir().unwrap(),
            config,
            services_url,
        ));
        ForgePluginRepository::new(infra)
    }

    #[test]
    fn test_resolve_plugin_conflicts_keeps_last() {
        let plugins = vec![
            fixture_plugin("alpha", PluginSource::Global),
            fixture_plugin("beta", PluginSource::Global),
            fixture_plugin("alpha", PluginSource::Project),
        ];

        let actual = resolve_plugin_conflicts(plugins);

        assert_eq!(actual.len(), 2);
        let alpha = actual.iter().find(|p| p.name == "alpha").unwrap();
        assert_eq!(alpha.source, PluginSource::Project);
        let beta = actual.iter().find(|p| p.name == "beta").unwrap();
        assert_eq!(beta.source, PluginSource::Global);
    }

    #[test]
    fn test_resolve_plugin_conflicts_no_duplicates() {
        let plugins = vec![
            fixture_plugin("alpha", PluginSource::Global),
            fixture_plugin("beta", PluginSource::Project),
        ];

        let actual = resolve_plugin_conflicts(plugins);

        assert_eq!(actual.len(), 2);
    }

    /// Integration-style test exercising the full Claude Code
    /// (`.claude-plugin/plugin.json`) discovery path against a fixture
    /// directory checked in under `src/fixtures/plugins/`.
    ///
    /// Verifies that:
    /// - the `.claude-plugin/plugin.json` marker (not the Forge-native
    ///   `.forge-plugin/plugin.json`) is detected,
    /// - `manifest` fields (name, version, description, author, hooks) are
    ///   parsed correctly,
    /// - declared component paths (commands, skills, agents) resolve to
    ///   absolute paths rooted at the plugin directory, and
    /// - `PluginSource` reflects the value supplied by the caller.
    #[tokio::test]
    async fn test_scan_root_loads_claude_code_plugin_fixture() {
        // Fixture: a real on-disk Claude Code-style plugin layout.
        let fixture_root =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("src/fixtures/plugins");

        let repo = fixture_plugin_repo();
        let (plugins, errors) = repo
            .scan_root(&fixture_root, PluginSource::Project)
            .await
            .expect("scan_root should succeed for a healthy fixture");

        assert!(
            errors.is_empty(),
            "Claude Code fixture must load cleanly, but got errors: {errors:?}"
        );
        assert_eq!(
            plugins.len(),
            1,
            "Expected exactly one plugin under the fixture root"
        );

        let plugin = &plugins[0];
        assert_eq!(plugin.name, "claude-code-demo");
        assert_eq!(plugin.manifest.version.as_deref(), Some("0.1.0"));
        assert_eq!(
            plugin.manifest.description.as_deref(),
            Some(
                "Claude Code 1:1 compatibility fixture used to verify ForgePluginRepository discovery."
            )
        );
        assert_eq!(plugin.source, PluginSource::Project);
        assert!(plugin.enabled, "plugins default to enabled before config overrides");
        assert!(!plugin.is_builtin);

        // Author should come through the detailed form.
        match &plugin.manifest.author {
            Some(forge_domain::PluginAuthor::Detailed { name, email, url }) => {
                assert_eq!(name, "Forge Test Harness");
                assert_eq!(email.as_deref(), Some("test@forgecode.dev"));
                assert!(url.is_none());
            }
            other => panic!("expected detailed author, got {other:?}"),
        }

        // Component paths must be resolved relative to the plugin root.
        let expected_root = fixture_root.join("claude_code_plugin");
        assert_eq!(plugin.path, expected_root);

        assert_eq!(plugin.commands_paths.len(), 1);
        assert!(
            plugin.commands_paths[0].ends_with("claude_code_plugin/commands"),
            "commands path should resolve to <plugin>/commands, got {:?}",
            plugin.commands_paths[0]
        );

        assert_eq!(plugin.skills_paths.len(), 1);
        assert!(
            plugin.skills_paths[0].ends_with("claude_code_plugin/skills"),
            "skills path should resolve to <plugin>/skills, got {:?}",
            plugin.skills_paths[0]
        );

        assert_eq!(plugin.agents_paths.len(), 1);
        assert!(
            plugin.agents_paths[0].ends_with("claude_code_plugin/agents"),
            "agents path should resolve to <plugin>/agents, got {:?}",
            plugin.agents_paths[0]
        );

        // Hooks must be parsed from the referenced JSON file even though
        // Phase 1 keeps them opaque.
        assert!(
            plugin.hooks_config.is_some(),
            "hooks_config should be populated from hooks/hooks.json"
        );

        // No MCP servers were declared.
        assert!(plugin.mcp_servers.is_none());
    }
}
