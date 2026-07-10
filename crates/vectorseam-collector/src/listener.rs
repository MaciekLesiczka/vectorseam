use std::io;
use std::path::{Path as FsPath, PathBuf};

use anyhow::{Context, Result, anyhow};
use tokio::net::{TcpListener, TcpStream, UnixListener, UnixStream};
use tracing::warn;

use crate::config::Config;

pub(crate) enum BoundListener {
    Tcp(TcpListener),
    Unix {
        listener: UnixListener,
        path: PathBuf,
    },
}

pub(crate) enum AcceptedConnection {
    Tcp(TcpStream),
    Unix(UnixStream),
}

impl BoundListener {
    pub(crate) async fn bind(config: &Config) -> Result<Self> {
        if let Some(path) = &config.unix_socket {
            prepare_socket(path)?;
            let listener = UnixListener::bind(path)
                .with_context(|| format!("binding unix socket {}", path.display()))?;
            return Ok(Self::Unix {
                listener,
                path: path.clone(),
            });
        }

        let listener = TcpListener::bind(config.listen)
            .await
            .with_context(|| format!("binding TCP listener {}", config.listen))?;
        Ok(Self::Tcp(listener))
    }

    pub(crate) async fn accept(&self) -> Result<AcceptedConnection> {
        match self {
            Self::Tcp(listener) => {
                let (stream, _addr) = listener
                    .accept()
                    .await
                    .context("accepting TCP connection")?;
                Ok(AcceptedConnection::Tcp(stream))
            }
            Self::Unix { listener, .. } => {
                let (stream, _addr) = listener
                    .accept()
                    .await
                    .context("accepting unix socket connection")?;
                Ok(AcceptedConnection::Unix(stream))
            }
        }
    }

    pub(crate) fn cleanup(&self) {
        if let Self::Unix { path, .. } = self {
            cleanup_socket(path);
        }
    }
}

fn prepare_socket(path: &FsPath) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating socket directory {}", parent.display()))?;
        }
    }

    match std::fs::symlink_metadata(path) {
        Ok(metadata) => {
            if is_unix_socket(&metadata) {
                std::fs::remove_file(path)
                    .with_context(|| format!("removing stale socket {}", path.display()))?;
            } else {
                return Err(anyhow!(
                    "socket path {} already exists and is not a socket",
                    path.display()
                ));
            }
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(error).with_context(|| format!("checking socket path {}", path.display()));
        }
    }
    Ok(())
}

#[cfg(unix)]
fn is_unix_socket(metadata: &std::fs::Metadata) -> bool {
    use std::os::unix::fs::FileTypeExt;

    metadata.file_type().is_socket()
}

#[cfg(not(unix))]
fn is_unix_socket(_metadata: &std::fs::Metadata) -> bool {
    false
}

fn cleanup_socket(path: &FsPath) {
    if let Err(error) = std::fs::remove_file(path) {
        if error.kind() != io::ErrorKind::NotFound {
            warn!(path = %path.display(), %error, "failed to remove socket path");
        }
    }
}
