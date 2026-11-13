//! Planted defects for the Rust quality gate.
pub mod unformatted;

/// Planted clippy::needless_return in library code.
pub fn double(x: i32) -> i32 {
    return x * 2;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn doubles() {
        // Planted clippy::bool_assert_comparison in TEST code: only
        // `--all-targets` lints this.
        assert_eq!(double(2) == 4, true);
    }
}

pub mod macro_calls;
pub mod spread;
pub mod two_lints;
