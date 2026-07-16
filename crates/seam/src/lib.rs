//! VectorSeam tuner.
//!
//! Phase B is intentionally split between synchronous intermediate-reading
//! glue and a pure, deterministic estimator.

#![forbid(unsafe_code)]

mod accounting;
pub mod aggregate;
pub mod config;
pub mod intermediate;
pub mod math;
pub mod model;
mod population;
