#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("duplicated")]
    Duplicated,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_debug() {
        let e = Error::Duplicated;
        assert_eq!(format!("{e:?}"), "Duplicated");
    }

    #[test]
    fn error_display() {
        let e = Error::Duplicated;
        assert_eq!(format!("{e}"), "duplicated");
    }
}
