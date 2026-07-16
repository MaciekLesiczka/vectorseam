//! VectorSeam tuner.
//!
//! Phase B is intentionally split between synchronous intermediate-reading
//! glue and a pure, deterministic estimator.

#![forbid(unsafe_code)]

pub mod config;
