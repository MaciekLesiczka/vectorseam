//! Core protocol, cohort, segment, and window helpers for VectorSeam.
//!
//! This crate is intentionally free of async runtimes and IO. It operates on
//! byte buffers, timestamps, and object key strings so all components
//! share the same format logic.

#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]
#![forbid(unsafe_code)]

mod binary;

pub mod cohort;
pub mod frame;
//pub mod segment;
//pub mod window;
