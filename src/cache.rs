use lru::LruCache;
use sha2::{Digest, Sha256};
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

use opencl3::program::Program;

// ── In-memory LRU program cache (per device) ────────────────────────────────

const PROGRAM_CACHE_CAPACITY: usize = 128;

/// LRU cache of compiled OpenCL programs, keyed by SHA-256 of the full program source.
pub struct ProgramCache {
    inner: LruCache<String, Arc<Program>>,
}

impl ProgramCache {
    pub fn new() -> Self {
        Self {
            inner: LruCache::new(NonZeroUsize::new(PROGRAM_CACHE_CAPACITY).unwrap()),
        }
    }

    pub fn get(&mut self, key: &str) -> Option<Arc<Program>> {
        self.inner.get(key).cloned()
    }

    pub fn put(&mut self, key: String, program: Arc<Program>) {
        self.inner.put(key, program);
    }

    pub fn contains(&self, key: &str) -> bool {
        self.inner.contains(key)
    }

    /// Get the number of cached programs.
    pub fn size(&self) -> usize {
        self.inner.len()
    }

    /// Clear all cached programs.
    pub fn clear(&mut self) {
        self.inner.clear();
    }
}

impl Default for ProgramCache {
    fn default() -> Self {
        Self::new()
    }
}

// ── Disk cache (singleton) ───────────────────────────────────────────────────

/// Persistent on-disk cache of compiled OpenCL program binaries.
/// Stored at `~/.cache/clesperanto/<device_hash>/<source_hash>.bin`.
/// Disabled when `CLESPERANTO_NO_CACHE` environment variable is set.
pub struct DiskCache {
    root: Option<PathBuf>,
}

static DISK_CACHE: OnceLock<DiskCache> = OnceLock::new();

impl DiskCache {
    /// Access the global DiskCache singleton.
    pub fn instance() -> &'static DiskCache {
        DISK_CACHE.get_or_init(|| {
            if std::env::var("CLESPERANTO_NO_CACHE").is_ok() {
                return DiskCache { root: None };
            }
            let root = dirs::cache_dir()
                .map(|d| d.join("clesperanto"))
                .or_else(|| Some(PathBuf::from(".cache/clesperanto")));
            DiskCache { root }
        })
    }

    /// SHA-256 hex digest of the given string (used for cache keys).
    pub fn hash(input: &str) -> String {
        hex::encode(Sha256::digest(input.as_bytes()))
    }

    pub fn get_file_path(
        &self,
        device_hash: &str,
        source_hash: &str,
        ext: &str,
    ) -> Option<PathBuf> {
        self.root
            .as_ref()
            .map(|r| r.join(device_hash).join(format!("{}.{}", source_hash, ext)))
    }

    pub fn is_enabled(&self) -> bool {
        self.root.is_some() && std::env::var("CLESPERANTO_NO_CACHE").is_err()
    }

    pub fn set_enabled(&self, flag: bool) {
        if flag {
            std::env::remove_var("CLESPERANTO_NO_CACHE");
        } else {
            std::env::set_var("CLESPERANTO_NO_CACHE", "1");
        }
    }

    pub fn get_cache_directory(&self) -> Option<&std::path::Path> {
        self.root.as_deref()
    }

    pub fn exists(&self, device_hash: &str, source_hash: &str, ext: &str) -> bool {
        self.get_file_path(device_hash, source_hash, ext)
            .is_some_and(|path| path.exists())
    }

    /// Load a cached binary. Returns `None` if not found or cache is disabled.
    pub fn load_binary(&self, device_hash: &str, source_hash: &str, ext: &str) -> Option<Vec<u8>> {
        if !self.is_enabled() {
            return None;
        }
        let path = self.get_file_path(device_hash, source_hash, ext)?;
        std::fs::read(&path).ok()
    }

    /// Save a compiled binary to the disk cache.
    pub fn save_binary(&self, device_hash: &str, source_hash: &str, ext: &str, data: &[u8]) {
        if !self.is_enabled() {
            return;
        }
        let Some(path) = self.get_file_path(device_hash, source_hash, ext) else {
            return;
        };
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = std::fs::write(&path, data);
    }
}

// ── Shared cache mutex wrapper used inside OpenCLDevice ─────────────────────

pub type SharedProgramCache = Arc<Mutex<ProgramCache>>;

pub fn new_shared_program_cache() -> SharedProgramCache {
    Arc::new(Mutex::new(ProgramCache::new()))
}

pub fn is_cache_enabled() -> bool {
    DiskCache::instance().is_enabled()
}

pub fn use_cache(flag: bool) {
    DiskCache::instance().set_enabled(flag);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn program_cache_lru() {
        // Just test that the cache compiles and basic operations work without a GPU
        let mut cache = ProgramCache::new();
        assert!(!cache.contains("nonexistent"));
        assert!(cache.get("nonexistent").is_none());
        assert_eq!(cache.size(), 0);
        cache.clear();
        assert_eq!(cache.size(), 0);
    }

    #[test]
    fn disk_cache_hash_deterministic() {
        let h1 = DiskCache::hash("hello world");
        let h2 = DiskCache::hash("hello world");
        assert_eq!(h1, h2);
        assert_ne!(h1, DiskCache::hash("different"));
    }

    #[test]
    fn disk_cache_save_load_roundtrip() {
        let cache = DiskCache::instance();
        if cache.root.is_none() {
            return; // disk cache disabled
        }
        let device_hash = "test_device_abc";
        let source_hash = "test_source_xyz";
        let data = b"binary_data_1234";
        cache.save_binary(device_hash, source_hash, "bin", data);
        let loaded = cache.load_binary(device_hash, source_hash, "bin");
        assert_eq!(loaded.as_deref(), Some(data.as_slice()));
    }
}
