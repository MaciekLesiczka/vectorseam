//! VectorSeam tuner.
//!
//! Phase B is intentionally split between synchronous intermediate-reading
//! glue and a pure, deterministic estimator.

#![forbid(unsafe_code)]

mod accounting;
pub mod aggregate;
pub mod config;
mod database;
pub mod intermediate;
pub mod math;
mod measure;
pub mod model;
mod pacer;
mod pipeline;
mod population;
pub mod tuner;
