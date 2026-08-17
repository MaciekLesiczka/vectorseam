//! Shared recommendation value bounds.

/// Smallest valid `hnsw.ef_search` recommendation.
pub const MIN_EF_SEARCH: i32 = 1;

/// Largest valid `hnsw.ef_search` recommendation.
pub const MAX_EF_SEARCH: i32 = 1_000;
