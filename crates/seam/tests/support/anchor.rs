use std::path::PathBuf;

use anyhow::{Context, Result, ensure};
use serde::de::DeserializeOwned;
use serde_json::Value;

pub(crate) fn fixture_root() -> PathBuf {
    std::env::var_os("SEAM_F_PG_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/seam-fixtures/f-pg")
        })
}

pub(crate) fn read_anchor_comparison<T>() -> Result<T>
where
    T: DeserializeOwned,
{
    let fixture_root = fixture_root();
    let path = fixture_root.join("anchor/comparison.json");
    let bytes = std::fs::read(&path).with_context(|| format!("read {}", path.display()))?;
    let comparison: Value = serde_json::from_slice(&bytes)?;
    ensure!(comparison["format_version"] == 1);
    ensure!(
        comparison["query_order"]
            .as_array()
            .is_some_and(|rows| rows.len() == 500)
    );
    ensure!(
        comparison["recall_rows"]
            .as_array()
            .is_some_and(|rows| rows.len() == 2_500)
    );
    Ok(serde_json::from_value(comparison)?)
}
