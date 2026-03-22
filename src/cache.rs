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

    fn path(&self, device_hash: &str, source_hash: &str, ext: &str) -> Option<PathBuf> {
        self.root.as_ref().map(|r| {
            r.join(device_hash).join(format!("{}.{}", source_hash, ext))
        })
    }

    /// Load a cached binary. Returns `None` if not found or cache is disabled.
    pub fn load(&self, device_hash: &str, source_hash: &str, ext: &str) -> Option<Vec<u8>> {
        let path = self.path(device_hash, source_hash, ext)?;
        std::fs::read(&path).ok()
    }

    /// Save a compiled binary to the disk cache.
    pub fn save(&self, device_hash: &str, source_hash: &str, ext: &str, data: &[u8]) {
        let Some(path) = self.path(device_hash, source_hash, ext) else { return };
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn program_cache_lru() {
        // Just test that the cache compiles and basic operations work without a GPU
        let mut cache = ProgramCache::new();
        assert!(!cache.contains("nonexistent"));
        assert!(cache.get("nonexistent").is_none());
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
        cache.save(device_hash, source_hash, "bin", data);
        let loaded = cache.load(device_hash, source_hash, "bin");
        assert_eq!(loaded.as_deref(), Some(data.as_slice()));
    }
}
