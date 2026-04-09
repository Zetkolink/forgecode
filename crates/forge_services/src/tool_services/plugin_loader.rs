use std::sync::Arc;

use forge_app::PluginLoader;
use forge_domain::{LoadedPlugin, PluginRepository};
use tokio::sync::RwLock;

/// In-process plugin loader that caches discovery results.
///
/// Wraps a [`PluginRepository`] (typically `ForgePluginRepository`) and
/// memoises its output in an `RwLock<Option<Arc<Vec<LoadedPlugin>>>>`.
///
/// Mirrors `ForgeSkillFetch` — the first call scans the filesystem, later
/// calls return a cheap `Arc::clone` of the cached vector. Callers can
/// drop the cache via [`invalidate_cache`](PluginLoader::invalidate_cache)
/// (invoked by `:plugin reload` / `:plugin enable` / `:plugin disable`
/// once Phase 9 lands).
///
/// ## Why not memoise inside `ForgePluginRepository`?
///
/// Keeping the repository stateless makes it trivially testable with
/// temporary directories and keeps the I/O layer honest (every call hits
/// disk). The service layer is the correct place to trade freshness for
/// speed.
pub struct ForgePluginLoader<R> {
    repository: Arc<R>,
    /// In-memory cache of plugins loaded from the repository.
    ///
    /// We store `Arc<Vec<_>>` (rather than `Vec<_>`) so that
    /// [`list_plugins`](PluginLoader::list_plugins) can return a cheap
    /// clone without holding the read lock for the duration of the
    /// caller's work. Callers that mutate the returned vector only touch
    /// their own clone.
    cache: RwLock<Option<Arc<Vec<LoadedPlugin>>>>,
}

impl<R> ForgePluginLoader<R> {
    /// Creates a new plugin loader wrapping `repository`.
    pub fn new(repository: Arc<R>) -> Self {
        Self { repository, cache: RwLock::new(None) }
    }

    /// Returns a cached `Arc<Vec<LoadedPlugin>>` or loads it from the
    /// repository on first call.
    ///
    /// Uses double-checked locking: a cheap read-lock fast path, falling
    /// back to an expensive write-lock slow path when the cache is empty.
    async fn get_or_load(&self) -> anyhow::Result<Arc<Vec<LoadedPlugin>>>
    where
        R: PluginRepository,
    {
        // Fast path: read lock, clone Arc if populated.
        {
            let guard = self.cache.read().await;
            if let Some(plugins) = guard.as_ref() {
                return Ok(Arc::clone(plugins));
            }
        }

        // Slow path: write lock, re-check, load.
        let mut guard = self.cache.write().await;
        if let Some(plugins) = guard.as_ref() {
            return Ok(Arc::clone(plugins));
        }

        let plugins = Arc::new(self.repository.load_plugins().await?);
        *guard = Some(Arc::clone(&plugins));
        Ok(plugins)
    }
}

#[async_trait::async_trait]
impl<R: PluginRepository + Send + Sync + 'static> PluginLoader for ForgePluginLoader<R> {
    async fn list_plugins(&self) -> anyhow::Result<Vec<LoadedPlugin>> {
        let plugins = self.get_or_load().await?;
        Ok((*plugins).clone())
    }

    async fn invalidate_cache(&self) {
        let mut guard = self.cache.write().await;
        *guard = None;
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use forge_domain::{LoadedPlugin, PluginRepository};
    use pretty_assertions::assert_eq;

    use super::*;

    /// Test repository that counts how many times `load_plugins` was
    /// invoked and returns a mutable list.
    struct MockPluginRepository {
        plugins: Mutex<Vec<LoadedPlugin>>,
        load_calls: Mutex<u32>,
    }

    impl MockPluginRepository {
        fn new(plugins: Vec<LoadedPlugin>) -> Self {
            Self { plugins: Mutex::new(plugins), load_calls: Mutex::new(0) }
        }

        fn load_call_count(&self) -> u32 {
            *self.load_calls.lock().unwrap()
        }

        fn set_plugins(&self, plugins: Vec<LoadedPlugin>) {
            *self.plugins.lock().unwrap() = plugins;
        }
    }

    #[async_trait::async_trait]
    impl PluginRepository for MockPluginRepository {
        async fn load_plugins(&self) -> anyhow::Result<Vec<LoadedPlugin>> {
            *self.load_calls.lock().unwrap() += 1;
            Ok(self.plugins.lock().unwrap().clone())
        }
    }

    fn fixture_plugin(name: &str) -> LoadedPlugin {
        use std::path::PathBuf;

        use forge_domain::{PluginManifest, PluginSource};

        LoadedPlugin {
            name: name.to_string(),
            path: PathBuf::from(format!("/fake/{name}")),
            manifest: PluginManifest { name: Some(name.to_string()), ..Default::default() },
            source: PluginSource::Global,
            enabled: true,
            is_builtin: false,
            commands_paths: Vec::new(),
            agents_paths: Vec::new(),
            skills_paths: Vec::new(),
            hooks_config: None,
            mcp_servers: None,
        }
    }

    #[tokio::test]
    async fn test_list_plugins_first_call_reads_repository() {
        let repo = Arc::new(MockPluginRepository::new(vec![fixture_plugin("alpha")]));
        let loader = ForgePluginLoader::new(repo.clone());

        let actual = loader.list_plugins().await.unwrap();

        assert_eq!(actual.len(), 1);
        assert_eq!(actual[0].name, "alpha");
        assert_eq!(repo.load_call_count(), 1);
    }

    #[tokio::test]
    async fn test_list_plugins_second_call_returns_cached() {
        let repo = Arc::new(MockPluginRepository::new(vec![
            fixture_plugin("alpha"),
            fixture_plugin("beta"),
        ]));
        let loader = ForgePluginLoader::new(repo.clone());

        let first = loader.list_plugins().await.unwrap();
        let second = loader.list_plugins().await.unwrap();

        assert_eq!(first.len(), 2);
        assert_eq!(second.len(), 2);
        // Repository was only hit once despite two calls.
        assert_eq!(repo.load_call_count(), 1);
    }

    #[tokio::test]
    async fn test_invalidate_cache_forces_reload() {
        let repo = Arc::new(MockPluginRepository::new(vec![fixture_plugin("alpha")]));
        let loader = ForgePluginLoader::new(repo.clone());

        // First call populates cache.
        let _ = loader.list_plugins().await.unwrap();
        assert_eq!(repo.load_call_count(), 1);

        // Invalidate and verify the next call re-reads.
        loader.invalidate_cache().await;
        let _ = loader.list_plugins().await.unwrap();
        assert_eq!(repo.load_call_count(), 2);
    }

    #[tokio::test]
    async fn test_invalidate_cache_surfaces_new_plugins() {
        let repo = Arc::new(MockPluginRepository::new(vec![fixture_plugin("alpha")]));
        let loader = ForgePluginLoader::new(repo.clone());

        // Cache the initial state.
        let before = loader.list_plugins().await.unwrap();
        assert_eq!(before.len(), 1);

        // Simulate a new plugin landing on disk mid-session.
        repo.set_plugins(vec![fixture_plugin("alpha"), fixture_plugin("beta")]);

        // Without invalidation, we still see the cached snapshot.
        let stale = loader.list_plugins().await.unwrap();
        assert_eq!(stale.len(), 1);

        // After invalidation, the new plugin surfaces.
        loader.invalidate_cache().await;
        let fresh = loader.list_plugins().await.unwrap();
        assert_eq!(fresh.len(), 2);
        assert_eq!(fresh[1].name, "beta");
    }
}
