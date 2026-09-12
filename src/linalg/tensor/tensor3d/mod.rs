use crate::linalg::storage::{StackStorage, Storage};
use crate::scalar::Scalar;

mod access;
mod construction;
mod stats;

/// `CHANNELS`/`ROWS`/`COLS` fix the shape for good: there is no runtime
/// shape/stride state, no in-place transpose, nothing to keep in sync with
/// the type. Strides are always `ROWS * COLS`/`COLS`/`1` and folded into
/// every flat index at the call site instead.
pub struct Tensor3D<
    const CHANNELS: usize,
    const ROWS: usize,
    const COLS: usize,
    S: Storage<[[[Scalar; COLS]; ROWS]; CHANNELS]> = StackStorage<[[[Scalar; COLS]; ROWS]; CHANNELS]>,
> {
    data: S,
}
