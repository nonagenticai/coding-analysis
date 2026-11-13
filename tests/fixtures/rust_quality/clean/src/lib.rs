//! A clean crate: the negative control for the Rust quality gate.

/// Doubles a number.
pub fn double(x: i32) -> i32 {
    x * 2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn doubles() {
        assert_eq!(double(2), 4);
    }
}
