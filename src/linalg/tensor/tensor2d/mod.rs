use crate::linalg::storage::{StackStorage, Storage};
use crate::scalar::Scalar;

mod access;
mod algebra;
mod construction;
mod indexing;
mod ops;
mod shape;
mod stats;

/// `ROWS`/`COLS` fix the shape for good: unlike [`TensorView`], nothing here
/// mutates it, so there is no `shape`/stride state to carry at runtime.
/// `row_stride`/`col_stride` are always `COLS`/`1` and folded into every flat
/// index at the call site instead.
#[derive(Clone, Copy, Debug)]
#[allow(unused_variables)]
pub struct Tensor<
    const ROWS: usize,
    const COLS: usize,
    S: Storage<[[Scalar; COLS]; ROWS]> = StackStorage<[[Scalar; COLS]; ROWS]>,
> {
    data: S,
}

/// Goes through the raw buffer (`get_raw_buffer`) rather than a `#[derive]`:
/// `S` (`StackStorage`/`HeapStorage`) has no reason to implement
/// `defmt::Format` itself, only the logged content matters for the log.
#[cfg(feature = "defmt")]
impl<const ROWS: usize, const COLS: usize, S: Storage<[[Scalar; COLS]; ROWS]>> defmt::Format
    for Tensor<ROWS, COLS, S>
{
    fn format(&self, fmt: defmt::Formatter) {
        // Generic `{}` rather than `{=[f32]}`: `Scalar` is `f32` or `f64`
        // depending on the `f64` feature, and both implement `defmt::Format`.
        defmt::write!(
            fmt,
            "Tensor<{=usize}, {=usize}> {}",
            ROWS,
            COLS,
            self.get_raw_buffer()
        );
    }
}

pub struct TensorView<'a> {
    data: &'a [Scalar],
    reference_index: usize,
    row_stride: usize,
    col_stride: usize,
    shape: (usize, usize),
}
impl<'a> TensorView<'a> {
    pub fn get(self: &Self, i: usize, j: usize) -> Scalar {
        debug_assert!(i < self.shape.0 && j < self.shape.1);
        let flat_index: usize = i * self.row_stride + j * self.col_stride;
        let index: usize = flat_index + self.reference_index;
        self.data[index]
    }
}

/// A vector is just a `Tensor` with one column: this alias is the only
/// thing that distinguishes it, `Tensor` is the crate's one elementary
/// structure.
pub type Vector<const N: usize> = Tensor<N, 1>;
