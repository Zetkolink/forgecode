//! Hook configuration loader — merges `hooks.json` from every source
//! (user-global, project, and enabled plugins) into a single
//! [`MergedHooksConfig`] consumed by the dispatcher.
//!
//! Precedence rules:
//!
//! 1. **User global** — `~/forge/hooks.json` (via `Environment::base_path`).
//! 2. **Project** — `./.forge/hooks.json` (via `Environment::cwd`).
//! 3. **Plugin** — every enabled plugin's `manifest.hooks` field, which may be
//!    an inline object, a relative path to a JSON file, or a mixed array of
//!    both (see [`forge_domain::PluginHooksManifestField`]).
//!
//! All three sources are **additive** — matchers from all three live in
//! the same per-event list. The dispatcher walks the combined list in
//! order, so the effective execution order is user → project → plugin
//! (roughly alphabetical within each group).
//!
//! Each entry carries a [`HookConfigSource`] plus an optional plugin
//! name/root so the shell executor can inject `FORGE_PLUGIN_ROOT` and
//! related environment variables correctly.
//!
//! The loader caches the merged result in an `RwLock<Option<Arc<_>>>`
//! using the same double-checked-locking pattern as
//! [`crate::tool_services::ForgePluginLoader`]. Call
//! [`HookConfigLoader::invalidate`] to force a re-scan after a plugin
//! enable/disable.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use forge_app::hook_runtime::{
    HookConfigLoaderService, HookConfigSource, HookMatcherWithSource, MergedHooksConfig,
};
use forge_app::{EnvironmentInfra, FileInfoInfra, FileReaderInfra};
use forge_domain::{HooksConfig, LoadedPlugin, PluginHooksManifestField, PluginRepository};
use tokio::sync::RwLock;

/// Extension helper for [`MergedHooksConfig`] that owns the merge logic.
/// Kept as a free function instead of an inherent method so the data type
/// stays in `forge_app` with zero dependencies on `forge_domain`'s heavier
/// types.
fn extend_from(
    merged: &mut MergedHooksConfig,
    config: HooksConfig,
    source: HookConfigSource,
    plugin_root: Option<PathBuf>,
    plugin_name: Option<String>,
) {
    for (event, matchers) in config.0 {
        let entry = merged.entries.entry(event).or_default();
        for matcher in matchers {
            entry.push(HookMatcherWithSource {
                matcher,
                source: source.clone(),
                plugin_root: plugin_root.clone(),
                plugin_name: plugin_name.clone(),
            });
        }
    }
}

/// Loads and caches the [`MergedHooksConfig`].
///
/// Generic over `F`, which must provide environment + file access. The
/// plugin repository is passed as `Arc<dyn PluginRepository>` so the
/// loader doesn't need to know about the concrete `ForgePluginLoader`
/// type (which would create a circular service dependency).
pub struct ForgeHookConfigLoader<F> {
    infra: Arc<F>,
    plugin_repository: Arc<dyn PluginRepository>,
    cache: RwLock<Option<Arc<MergedHooksConfig>>>,
}

impl<F> ForgeHookConfigLoader<F>
where
    F: EnvironmentInfra + FileReaderInfra + FileInfoInfra + Send + Sync,
{
    /// Creates a new loader. The cache is empty until
    /// [`load`](HookConfigLoaderService::load) is called for the first
    /// time.
    pub fn new(infra: Arc<F>, plugin_repository: Arc<dyn PluginRepository>) -> Self {
        Self { infra, plugin_repository, cache: RwLock::new(None) }
    }

    /// Internal helper: do the actual merge without touching the cache.
    async fn load_uncached(&self) -> anyhow::Result<MergedHooksConfig> {
        let mut merged = MergedHooksConfig::default();

        let env = self.infra.get_environment();

        // 1. User-global: ~/forge/hooks.json
        let user_path = env.base_path.join("hooks.json");
        if let Some(config) = self.read_hooks_json(&user_path).await? {
            extend_from(
                &mut merged,
                config,
                HookConfigSource::UserGlobal,
                None,
                None,
            );
        }

        // 2. Project: ./.forge/hooks.json
        let project_path = env.cwd.join(".forge/hooks.json");
        if let Some(config) = self.read_hooks_json(&project_path).await? {
            extend_from(&mut merged, config, HookConfigSource::Project, None, None);
        }

        // 3. Plugin hooks
        let plugin_result = self.plugin_repository.load_plugins_with_errors().await?;
        for plugin in plugin_result.plugins.iter().filter(|p| p.enabled) {
            if let Err(e) = self.merge_plugin(plugin, &mut merged).await {
                tracing::warn!(
                    plugin = plugin.name.as_str(),
                    error = %e,
                    "failed to load plugin hooks.json; skipping this plugin"
                );
            }
        }

        Ok(merged)
    }

    /// Merge hooks contributed by a single plugin into `merged`.
    ///
    /// Handles all three variants of [`PluginHooksManifestField`]:
    ///
    /// - `Path("hooks/hooks.json")` — resolve relative to plugin root and read
    ///   the file.
    /// - `Inline(...)` — re-serialise and re-parse the `serde_json::Value`
    ///   placeholder into a proper [`HooksConfig`].
    /// - `Array([...])` — recursively merge each element.
    async fn merge_plugin(
        &self,
        plugin: &LoadedPlugin,
        merged: &mut MergedHooksConfig,
    ) -> anyhow::Result<()> {
        let Some(hooks_field) = plugin.manifest.hooks.as_ref() else {
            return Ok(());
        };
        self.merge_hooks_field(plugin, hooks_field, merged).await
    }

    /// Recursively merges a [`PluginHooksManifestField`] into `merged`.
    ///
    /// Uses `Box<dyn Future>` so the recursive call compiles under
    /// `async fn` (Rust doesn't allow direct recursion in `async fn`
    /// without boxing).
    fn merge_hooks_field<'a>(
        &'a self,
        plugin: &'a LoadedPlugin,
        field: &'a PluginHooksManifestField,
        merged: &'a mut MergedHooksConfig,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = anyhow::Result<()>> + Send + 'a>> {
        Box::pin(async move {
            match field {
                PluginHooksManifestField::Path(rel) => {
                    let abs = plugin.path.join(rel);
                    if let Some(config) = self.read_hooks_json(&abs).await? {
                        extend_from(
                            merged,
                            config,
                            HookConfigSource::Plugin,
                            Some(plugin.path.clone()),
                            Some(plugin.name.clone()),
                        );
                    }
                }
                PluginHooksManifestField::Inline(inline) => {
                    // The placeholder `PluginHooksConfig.raw` is a flattened
                    // `serde_json::Value`. Re-serialise then re-parse into
                    // `HooksConfig` — cheap and keeps parsing centralised.
                    let value = serde_json::to_value(&inline.raw)?;
                    let config: HooksConfig = serde_json::from_value(value)?;
                    extend_from(
                        merged,
                        config,
                        HookConfigSource::Plugin,
                        Some(plugin.path.clone()),
                        Some(plugin.name.clone()),
                    );
                }
                PluginHooksManifestField::Array(items) => {
                    for item in items {
                        self.merge_hooks_field(plugin, item, merged).await?;
                    }
                }
            }
            Ok(())
        })
    }

    /// Read a `hooks.json` file at `path` and parse it into a
    /// [`HooksConfig`]. Returns `Ok(None)` when the file is missing (the
    /// common case — most projects don't have a `hooks.json`).
    async fn read_hooks_json(&self, path: &Path) -> anyhow::Result<Option<HooksConfig>> {
        if !self.infra.exists(path).await? {
            return Ok(None);
        }
        let raw = self.infra.read_utf8(path).await?;
        let parsed: HooksConfig = serde_json::from_str(&raw).map_err(|e| {
            anyhow::anyhow!("failed to parse hooks.json at {}: {}", path.display(), e)
        })?;
        Ok(Some(parsed))
    }
}

#[async_trait::async_trait]
impl<F> HookConfigLoaderService for ForgeHookConfigLoader<F>
where
    F: EnvironmentInfra + FileReaderInfra + FileInfoInfra + Send + Sync + 'static,
{
    /// Returns the merged hook config, loading it from disk on first
    /// call (or after [`invalidate`](Self::invalidate)).
    async fn load(&self) -> anyhow::Result<Arc<MergedHooksConfig>> {
        // Fast path: read lock, clone Arc if populated.
        {
            let guard = self.cache.read().await;
            if let Some(config) = guard.as_ref() {
                return Ok(Arc::clone(config));
            }
        }

        // Slow path: write lock, double-check, then load.
        let mut guard = self.cache.write().await;
        if let Some(config) = guard.as_ref() {
            return Ok(Arc::clone(config));
        }

        let merged = self.load_uncached().await?;
        let arc = Arc::new(merged);
        *guard = Some(Arc::clone(&arc));
        Ok(arc)
    }

    async fn invalidate(&self) -> anyhow::Result<()> {
        let mut guard = self.cache.write().await;
        *guard = None;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Mutex;

    use async_trait::async_trait;
    use forge_app::{DirectoryReaderInfra, EnvironmentInfra, FileInfoInfra, FileReaderInfra};
    use forge_domain::{
        ConfigOperation, Environment, FileInfo, HookCommand, HookEventName, LoadedPlugin,
        PluginHooksConfig, PluginHooksManifestField, PluginLoadResult, PluginManifest,
        PluginRepository, PluginSource,
    };
    use futures::{Stream, stream};
    use pretty_assertions::assert_eq;
    use tempfile::TempDir;

    use super::*;

    /// Minimal test infrastructure that satisfies the trait bounds of
    /// [`ForgeHookConfigLoader`] by delegating to a real temporary directory.
    #[derive(Clone)]
    struct TestInfra {
        env: Environment,
    }

    impl TestInfra {
        fn new(base: PathBuf, cwd: PathBuf) -> Self {
            let env = Environment {
                os: "linux".to_string(),
                cwd,
                home: None,
                shell: "/bin/bash".to_string(),
                base_path: base,
            };
            Self { env }
        }
    }

    impl EnvironmentInfra for TestInfra {
        type Config = forge_config::ForgeConfig;

        fn get_environment(&self) -> Environment {
            self.env.clone()
        }

        fn get_config(&self) -> anyhow::Result<forge_config::ForgeConfig> {
            Ok(forge_config::ForgeConfig::default())
        }

        async fn update_environment(&self, _ops: Vec<ConfigOperation>) -> anyhow::Result<()> {
            Ok(())
        }

        fn get_env_var(&self, _key: &str) -> Option<String> {
            None
        }

        fn get_env_vars(&self) -> BTreeMap<String, String> {
            BTreeMap::new()
        }
    }

    #[async_trait]
    impl FileReaderInfra for TestInfra {
        async fn read_utf8(&self, path: &Path) -> anyhow::Result<String> {
            tokio::fs::read_to_string(path)
                .await
                .map_err(anyhow::Error::from)
        }

        async fn read(&self, path: &Path) -> anyhow::Result<Vec<u8>> {
            tokio::fs::read(path).await.map_err(anyhow::Error::from)
        }

        async fn range_read_utf8(
            &self,
            path: &Path,
            _start_line: u64,
            _end_line: u64,
        ) -> anyhow::Result<(String, FileInfo)> {
            let text = self.read_utf8(path).await?;
            let total_lines = text.lines().count() as u64;
            Ok((
                text,
                FileInfo {
                    start_line: 1,
                    end_line: total_lines,
                    total_lines,
                    content_hash: String::new(),
                },
            ))
        }

        fn read_batch_utf8(
            &self,
            _batch_size: usize,
            _paths: Vec<PathBuf>,
        ) -> impl Stream<Item = (PathBuf, anyhow::Result<String>)> + Send {
            stream::empty()
        }
    }

    #[async_trait]
    impl FileInfoInfra for TestInfra {
        async fn is_binary(&self, _path: &Path) -> anyhow::Result<bool> {
            Ok(false)
        }

        async fn is_file(&self, path: &Path) -> anyhow::Result<bool> {
            Ok(path.is_file())
        }

        async fn exists(&self, path: &Path) -> anyhow::Result<bool> {
            Ok(path.exists())
        }

        async fn file_size(&self, path: &Path) -> anyhow::Result<u64> {
            let meta = tokio::fs::metadata(path).await?;
            Ok(meta.len())
        }
    }

    #[async_trait]
    impl DirectoryReaderInfra for TestInfra {
        async fn list_directory_entries(
            &self,
            _directory: &Path,
        ) -> anyhow::Result<Vec<(PathBuf, bool)>> {
            Ok(Vec::new())
        }

        async fn read_directory_files(
            &self,
            _directory: &Path,
            _pattern: Option<&str>,
        ) -> anyhow::Result<Vec<(PathBuf, String)>> {
            Ok(Vec::new())
        }
    }

    /// Controllable plugin repository backed by a `Mutex<Vec<LoadedPlugin>>`.
    #[derive(Default)]
    struct TestPluginRepository {
        plugins: Mutex<Vec<LoadedPlugin>>,
    }

    impl TestPluginRepository {
        fn with(plugins: Vec<LoadedPlugin>) -> Self {
            Self { plugins: Mutex::new(plugins) }
        }
    }

    #[async_trait]
    impl PluginRepository for TestPluginRepository {
        async fn load_plugins(&self) -> anyhow::Result<Vec<LoadedPlugin>> {
            Ok(self.plugins.lock().unwrap().clone())
        }

        async fn load_plugins_with_errors(&self) -> anyhow::Result<PluginLoadResult> {
            Ok(PluginLoadResult::new(
                self.plugins.lock().unwrap().clone(),
                Vec::new(),
            ))
        }
    }

    fn sample_hooks_json() -> &'static str {
        r#"{
            "PreToolUse": [
                {
                    "matcher": "Bash",
                    "hooks": [{"type": "command", "command": "echo hi"}]
                }
            ]
        }"#
    }

    #[tokio::test]
    async fn test_loader_with_no_hook_files_returns_empty_config() {
        let temp = TempDir::new().unwrap();
        let base = temp.path().join("base");
        let cwd = temp.path().join("cwd");
        std::fs::create_dir_all(&base).unwrap();
        std::fs::create_dir_all(&cwd).unwrap();

        let infra = Arc::new(TestInfra::new(base, cwd));
        let repo: Arc<dyn PluginRepository> = Arc::new(TestPluginRepository::default());
        let loader = ForgeHookConfigLoader::new(infra, repo);

        let merged = loader.load().await.unwrap();
        assert!(merged.is_empty());
        assert_eq!(merged.total_matchers(), 0);
    }

    #[tokio::test]
    async fn test_loader_reads_user_global_hooks_json() {
        let temp = TempDir::new().unwrap();
        let base = temp.path().join("base");
        let cwd = temp.path().join("cwd");
        std::fs::create_dir_all(&base).unwrap();
        std::fs::create_dir_all(&cwd).unwrap();
        std::fs::write(base.join("hooks.json"), sample_hooks_json()).unwrap();

        let infra = Arc::new(TestInfra::new(base, cwd));
        let repo: Arc<dyn PluginRepository> = Arc::new(TestPluginRepository::default());
        let loader = ForgeHookConfigLoader::new(infra, repo);

        let merged = loader.load().await.unwrap();
        assert_eq!(merged.total_matchers(), 1);
        let pre = merged.entries.get(&HookEventName::PreToolUse).unwrap();
        assert_eq!(pre.len(), 1);
        assert_eq!(pre[0].source, HookConfigSource::UserGlobal);
        assert!(pre[0].plugin_name.is_none());
        assert!(pre[0].plugin_root.is_none());
        assert_eq!(pre[0].matcher.hooks.len(), 1);
        match &pre[0].matcher.hooks[0] {
            HookCommand::Command(c) => assert_eq!(c.command, "echo hi"),
            other => panic!("expected Command, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_loader_reads_plugin_hooks_from_path_variant() {
        let temp = TempDir::new().unwrap();
        let base = temp.path().join("base");
        let cwd = temp.path().join("cwd");
        let plugin_root = temp.path().join("plugins/demo");
        std::fs::create_dir_all(&base).unwrap();
        std::fs::create_dir_all(&cwd).unwrap();
        std::fs::create_dir_all(plugin_root.join("hooks")).unwrap();
        std::fs::write(plugin_root.join("hooks/hooks.json"), sample_hooks_json()).unwrap();

        let plugin = LoadedPlugin {
            name: "demo".to_string(),
            manifest: PluginManifest {
                name: Some("demo".to_string()),
                hooks: Some(PluginHooksManifestField::Path(
                    "hooks/hooks.json".to_string(),
                )),
                ..Default::default()
            },
            path: plugin_root.clone(),
            source: PluginSource::Global,
            enabled: true,
            is_builtin: false,
            commands_paths: Vec::new(),
            agents_paths: Vec::new(),
            skills_paths: Vec::new(),
            hooks_config: None,
            mcp_servers: None,
        };

        let infra = Arc::new(TestInfra::new(base, cwd));
        let repo: Arc<dyn PluginRepository> = Arc::new(TestPluginRepository::with(vec![plugin]));
        let loader = ForgeHookConfigLoader::new(infra, repo);

        let merged = loader.load().await.unwrap();
        assert_eq!(merged.total_matchers(), 1);
        let pre = merged.entries.get(&HookEventName::PreToolUse).unwrap();
        assert_eq!(pre[0].source, HookConfigSource::Plugin);
        assert_eq!(pre[0].plugin_name.as_deref(), Some("demo"));
        assert_eq!(pre[0].plugin_root.as_deref(), Some(plugin_root.as_path()));
    }

    #[tokio::test]
    async fn test_loader_reads_plugin_hooks_from_inline_variant() {
        let temp = TempDir::new().unwrap();
        let base = temp.path().join("base");
        let cwd = temp.path().join("cwd");
        std::fs::create_dir_all(&base).unwrap();
        std::fs::create_dir_all(&cwd).unwrap();

        // Inline hooks object as a raw JSON value.
        let raw: serde_json::Value = serde_json::from_str(sample_hooks_json()).unwrap();

        let plugin = LoadedPlugin {
            name: "inline-demo".to_string(),
            manifest: PluginManifest {
                name: Some("inline-demo".to_string()),
                hooks: Some(PluginHooksManifestField::Inline(PluginHooksConfig { raw })),
                ..Default::default()
            },
            path: temp.path().join("plugins/inline-demo"),
            source: PluginSource::Global,
            enabled: true,
            is_builtin: false,
            commands_paths: Vec::new(),
            agents_paths: Vec::new(),
            skills_paths: Vec::new(),
            hooks_config: None,
            mcp_servers: None,
        };

        let infra = Arc::new(TestInfra::new(base, cwd));
        let repo: Arc<dyn PluginRepository> = Arc::new(TestPluginRepository::with(vec![plugin]));
        let loader = ForgeHookConfigLoader::new(infra, repo);

        let merged = loader.load().await.unwrap();
        assert_eq!(merged.total_matchers(), 1);
        let pre = merged.entries.get(&HookEventName::PreToolUse).unwrap();
        assert_eq!(pre[0].source, HookConfigSource::Plugin);
        assert_eq!(pre[0].plugin_name.as_deref(), Some("inline-demo"));
    }

    #[tokio::test]
    async fn test_loader_merges_all_three_sources_additively() {
        let temp = TempDir::new().unwrap();
        let base = temp.path().join("base");
        let cwd = temp.path().join("cwd");
        let plugin_root = temp.path().join("plugins/demo");
        std::fs::create_dir_all(&base).unwrap();
        std::fs::create_dir_all(cwd.join(".forge")).unwrap();
        std::fs::create_dir_all(&plugin_root).unwrap();

        // User global with PreToolUse matcher.
        std::fs::write(base.join("hooks.json"), sample_hooks_json()).unwrap();
        // Project with PostToolUse matcher.
        std::fs::write(
            cwd.join(".forge/hooks.json"),
            r#"{"PostToolUse":[{"matcher":"*","hooks":[{"type":"command","command":"post"}]}]}"#,
        )
        .unwrap();

        // Plugin inline with SessionStart matcher.
        let inline_raw: serde_json::Value = serde_json::from_str(
            r#"{"SessionStart":[{"hooks":[{"type":"command","command":"start"}]}]}"#,
        )
        .unwrap();

        let plugin = LoadedPlugin {
            name: "demo".to_string(),
            manifest: PluginManifest {
                name: Some("demo".to_string()),
                hooks: Some(PluginHooksManifestField::Inline(PluginHooksConfig {
                    raw: inline_raw,
                })),
                ..Default::default()
            },
            path: plugin_root,
            source: PluginSource::Global,
            enabled: true,
            is_builtin: false,
            commands_paths: Vec::new(),
            agents_paths: Vec::new(),
            skills_paths: Vec::new(),
            hooks_config: None,
            mcp_servers: None,
        };

        let infra = Arc::new(TestInfra::new(base, cwd));
        let repo: Arc<dyn PluginRepository> = Arc::new(TestPluginRepository::with(vec![plugin]));
        let loader = ForgeHookConfigLoader::new(infra, repo);

        let merged = loader.load().await.unwrap();
        assert_eq!(merged.total_matchers(), 3);
        assert_eq!(
            merged
                .entries
                .get(&HookEventName::PreToolUse)
                .map(Vec::len)
                .unwrap_or(0),
            1
        );
        assert_eq!(
            merged
                .entries
                .get(&HookEventName::PostToolUse)
                .map(Vec::len)
                .unwrap_or(0),
            1
        );
        assert_eq!(
            merged
                .entries
                .get(&HookEventName::SessionStart)
                .map(Vec::len)
                .unwrap_or(0),
            1
        );

        let pre = &merged.entries[&HookEventName::PreToolUse][0];
        let post = &merged.entries[&HookEventName::PostToolUse][0];
        let start = &merged.entries[&HookEventName::SessionStart][0];
        assert_eq!(pre.source, HookConfigSource::UserGlobal);
        assert_eq!(post.source, HookConfigSource::Project);
        assert_eq!(start.source, HookConfigSource::Plugin);
    }

    #[tokio::test]
    async fn test_loader_invalidate_forces_rescan() {
        let temp = TempDir::new().unwrap();
        let base = temp.path().join("base");
        let cwd = temp.path().join("cwd");
        std::fs::create_dir_all(&base).unwrap();
        std::fs::create_dir_all(&cwd).unwrap();

        let infra = Arc::new(TestInfra::new(base.clone(), cwd));
        let repo: Arc<dyn PluginRepository> = Arc::new(TestPluginRepository::default());
        let loader = ForgeHookConfigLoader::new(infra, repo);

        // First load: empty.
        let first = loader.load().await.unwrap();
        assert_eq!(first.total_matchers(), 0);

        // Write hooks.json, then load again — the cache should still
        // return the empty result.
        std::fs::write(base.join("hooks.json"), sample_hooks_json()).unwrap();
        let cached = loader.load().await.unwrap();
        assert_eq!(cached.total_matchers(), 0);

        // Invalidate, then reload — now we pick up the new file.
        loader.invalidate().await.unwrap();
        let fresh = loader.load().await.unwrap();
        assert_eq!(fresh.total_matchers(), 1);
    }
}
