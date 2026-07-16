#![allow(dead_code)]

pub mod anchor;
pub mod f_agg;

pub fn pending<T>(criterion: &str) -> T {
    panic!("{criterion} is intentionally pending tuner implementation")
}
