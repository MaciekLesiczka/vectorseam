pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::version;

    #[test]
    fn reports_crate_version() {
        assert_eq!(version(), "0.1.0");
    }
}
