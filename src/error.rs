#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("duplicated")]
    Duplicated,
}
