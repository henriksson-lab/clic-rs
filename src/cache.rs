use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

use opencl3::program::Program;

// ── In-memory LRU program cache (per device) ────────────────────────────────

const PROGRAM_CACHE_CAPACITY: usize = 128;

/// LRU cache of compiled OpenCL programs, keyed by a hash of the full program source.
pub struct ProgramCache {
    cache: HashMap<String, Arc<Program>>,
    lru: VecDeque<String>,
}

impl ProgramCache {
    pub fn new() -> Self {
        Self {
            cache: HashMap::with_capacity(PROGRAM_CACHE_CAPACITY),
            lru: VecDeque::with_capacity(PROGRAM_CACHE_CAPACITY),
        }
    }

    pub fn put(&mut self, key: String, program: Arc<Program>) {
        if let Some(entry) = self.cache.get_mut(&key) {
            self.lru.retain(|item| item != &key);
            self.lru.push_back(key);
            *entry = program;
            return;
        }

        if self.cache.len() >= PROGRAM_CACHE_CAPACITY {
            if let Some(oldest) = self.lru.pop_front() {
                self.cache.remove(&oldest);
            }
        }

        self.lru.push_back(key.clone());
        self.cache.insert(key, program);
    }

    pub fn get(&mut self, key: &str) -> Option<Arc<Program>> {
        let program = self.cache.get(key).cloned()?;
        self.lru.retain(|item| item != key);
        self.lru.push_back(key.to_string());
        Some(program)
    }

    pub fn contains(&self, key: &str) -> bool {
        self.cache.contains_key(key)
    }

    /// Get the number of cached programs.
    pub fn size(&self) -> usize {
        self.cache.len()
    }

    /// Clear all cached programs.
    pub fn clear(&mut self) {
        self.cache.clear();
        self.lru.clear();
    }
}

impl Drop for ProgramCache {
    fn drop(&mut self) {}
}

// ── Disk cache (singleton) ───────────────────────────────────────────────────

/// Persistent on-disk cache of compiled OpenCL program binaries.
/// Stored at `~/.cache/clesperanto/<device_hash>/<source_hash>.bin`.
/// Disabled when `CLESPERANTO_NO_CACHE` environment variable is set.
pub struct DiskCache {
    root: PathBuf,
}

static DISK_CACHE: OnceLock<DiskCache> = OnceLock::new();

impl DiskCache {
    pub fn resolve_cache_directory() -> PathBuf {
        #[cfg(windows)]
        {
            if let Some(path) = std::env::var_os("LOCALAPPDATA") {
                return PathBuf::from(path).join("clesperanto");
            }
            eprintln!("Failed to get AppData\\Local directory");
            return std::env::current_dir()
                .unwrap_or_else(|_| PathBuf::from("."))
                .join("clesperanto");
        }

        #[cfg(not(windows))]
        {
            if let Some(home_dir) = std::env::var_os("HOME") {
                return PathBuf::from(home_dir).join(".cache").join("clesperanto");
            }
            eprintln!("Failed to get user home directory");
            std::env::current_dir()
                .unwrap_or_else(|_| PathBuf::from("."))
                .join(".cache")
                .join("clesperanto")
        }
    }

    pub fn new() -> Self {
        Self {
            root: Self::resolve_cache_directory(),
        }
    }

    /// Access the global DiskCache singleton.
    pub fn instance() -> &'static DiskCache {
        DISK_CACHE.get_or_init(Self::new)
    }

    pub fn is_enabled(&self) -> bool {
        std::env::var("CLESPERANTO_NO_CACHE").is_err()
    }

    pub fn set_enabled(&self, flag: bool) {
        if flag {
            std::env::remove_var("CLESPERANTO_NO_CACHE");
        } else {
            std::env::set_var("CLESPERANTO_NO_CACHE", "1");
        }
    }

    pub fn get_cache_directory(&self) -> &std::path::Path {
        self.root.as_path()
    }

    pub fn get_file_path(&self, device_hash: &str, source_hash: &str, ext: &str) -> PathBuf {
        self.root
            .join(device_hash)
            .join(format!("{}.{}", source_hash, ext))
    }

    /// Save a compiled binary to the disk cache.
    pub fn save_binary(&self, device_hash: &str, source_hash: &str, ext: &str, data: &[u8]) {
        let binary_path = self.get_file_path(device_hash, source_hash, ext);
        if let Some(parent) = binary_path.parent() {
            std::fs::create_dir_all(parent).unwrap_or_else(|_| {
                panic!(
                    "Error: Failed to open cache file for writing: {}",
                    binary_path.display()
                )
            });
        }
        let mut outfile = std::fs::File::create(&binary_path).unwrap_or_else(|_| {
            panic!(
                "Error: Failed to open cache file for writing: {}",
                binary_path.display()
            )
        });
        outfile.write_all(data).unwrap_or_else(|_| {
            panic!(
                "Error: Failed to write cache file: {}",
                binary_path.display()
            )
        });
    }

    /// Load a cached binary. Returns `None` if not found or unreadable.
    pub fn load_binary(&self, device_hash: &str, source_hash: &str, ext: &str) -> Option<Vec<u8>> {
        let binary_path = self.get_file_path(device_hash, source_hash, ext);
        if !binary_path.exists() {
            return None;
        }

        let mut infile = match std::fs::File::open(&binary_path) {
            Ok(file) => file,
            Err(_) => {
                eprintln!(
                    "Error: Failed to open cache file: {}",
                    binary_path.display()
                );
                return None;
            }
        };

        let file_size = match infile.seek(SeekFrom::End(0)) {
            Ok(size) => size as usize,
            Err(_) => {
                eprintln!(
                    "Error: Failed to read cache file: {}",
                    binary_path.display()
                );
                return None;
            }
        };
        if file_size == 0 {
            eprintln!("Error: Cache file is empty: {}", binary_path.display());
            return None;
        }

        if infile.seek(SeekFrom::Start(0)).is_err() {
            eprintln!(
                "Error: Failed to read cache file: {}",
                binary_path.display()
            );
            return None;
        }

        let mut data = vec![0; file_size];
        if infile.read_exact(&mut data).is_err() {
            eprintln!(
                "Error: Failed to read cache file: {}",
                binary_path.display()
            );
            return None;
        }
        Some(data)
    }

    pub fn exists(&self, device_hash: &str, source_hash: &str, ext: &str) -> bool {
        self.get_file_path(device_hash, source_hash, ext).exists()
    }

    /// Hash string of the given input (used for cache keys).
    pub fn hash(input: &str) -> String {
        let mut hasher = DefaultHasher::new();
        input.hash(&mut hasher);
        hasher.finish().to_string()
    }
}

impl Drop for DiskCache {
    fn drop(&mut self) {}
}

// ── Shared cache mutex wrapper used inside OpenCLDevice ─────────────────────

pub type SharedProgramCache = Arc<Mutex<ProgramCache>>;

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
    fn disk_cache_resolve_directory_uses_clesperanto_leaf() {
        let path = DiskCache::resolve_cache_directory();
        assert_eq!(
            path.file_name().and_then(|name| name.to_str()),
            Some("clesperanto")
        );
    }

    #[test]
    fn disk_cache_save_load_roundtrip() {
        let cache = DiskCache::instance();
        if !cache.is_enabled() {
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
