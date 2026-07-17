use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use tracing_subscriber::EnvFilter;

use seam::config::Config;

#[derive(Debug, Parser)]
#[command(about = "Continuously calibrate pgvector HNSW ef_search")]
struct Cli {
    /// Tuner YAML configuration file.
    #[arg(long, env = "SEAM_CONFIG", value_name = "PATH")]
    config: PathBuf,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();
    let yaml = std::fs::read_to_string(&cli.config)
        .with_context(|| format!("reading tuner config {}", cli.config.display()))?;
    let config = Config::from_yaml_str(&yaml)
        .with_context(|| format!("loading tuner config {}", cli.config.display()))?;
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("building Tokio runtime")?;
    Ok(runtime.block_on(seam::daemon::run(config))?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stage4_cli_accepts_explicit_config_path() {
        let cli = Cli::try_parse_from(["seam", "--config", "/tmp/seam.yaml"]).unwrap();
        assert_eq!(cli.config, PathBuf::from("/tmp/seam.yaml"));
    }
}
