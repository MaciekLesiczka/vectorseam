use std::path::PathBuf;

use anyhow::{Context, Result, ensure};
use serde_json::Value;

pub fn read_anchor_comparison() -> Result<Value> {
    let fixture_root = std::env::var_os("SEAM_F_PG_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/seam-fixtures/f-pg")
        });
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
    Ok(comparison)
}
