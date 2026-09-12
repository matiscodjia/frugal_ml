use crate::linalg::storage::{StackStorage, Storage};
use crate::scalar::Scalar;

mod access;
mod construction;
mod shape;
mod stats;

/// The nested-array buffer shape backing a `Tensor4D<BATCHES, CHANNELS, ROWS,
/// COLS, ..>`: named so cross-tensor generic bounds (`tensordot_3`,
/// `cross_correlate2d`, [`Tensor4DBoxed`]) don't have to spell the nesting
/// out at every call site.
pub type Tensor4DBuffer<
    const BATCHES: usize,
    const CHANNELS: usize,
    const ROWS: usize,
    const COLS: usize,
> = [[[[Scalar; COLS]; ROWS]; CHANNELS]; BATCHES];

/// `S` determines where the buffer lives, the stack by default, so
/// existing instantiations (`Tensor4D::<1, 3, 32, 32>`) are
/// unchanged. See [`Tensor4DBoxed`] for the heap variant.
///
/// `BATCHES`/`CHANNELS`/`ROWS`/`COLS` fix the shape for good: there is no
/// runtime shape/stride state. Strides are always `CHANNELS * ROWS *
/// COLS`/`ROWS * COLS`/`COLS`/`1` and folded into every flat index at the
/// call site instead.
pub struct Tensor4D<
    const BATCHES: usize,
    const CHANNELS: usize,
    const ROWS: usize,
    const COLS: usize,
    S: Storage<Tensor4DBuffer<BATCHES, CHANNELS, ROWS, COLS>> = StackStorage<
        Tensor4DBuffer<BATCHES, CHANNELS, ROWS, COLS>,
    >,
> {
    data: S,
}

/// `Tensor4D` whose buffer lives on the heap, for shapes the stack
/// cannot carry (scaling-up benchmarks).
#[cfg(feature = "alloc")]
pub type Tensor4DBoxed<
    const BATCHES: usize,
    const CHANNELS: usize,
    const ROWS: usize,
    const COLS: usize,
> = Tensor4D<
    BATCHES,
    CHANNELS,
    ROWS,
    COLS,
    crate::linalg::storage::HeapStorage<Tensor4DBuffer<BATCHES, CHANNELS, ROWS, COLS>>,
>;
