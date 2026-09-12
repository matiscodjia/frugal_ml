use super::{Tensor6D, Tensor6DBuffer, TensorView6D};
use crate::linalg::storage::{Buffer, Storage};
use crate::scalar::Scalar;

/// Read-only access to a rank-6 tensor, whichever way it holds its elements:
/// `Tensor6D` owns them, `TensorView6D` only aliases someone else's buffer.
///
/// The six dimensions are const parameters rather than runtime values, so a
/// contraction over an implementor keeps checking its shared axes at compile
/// time: the trait erases ownership, not shape.
pub trait Rank6<
    const D0: usize,
    const D1: usize,
    const D2: usize,
    const D3: usize,
    const D4: usize,
    const D5: usize,
>
{
    fn get(self: &Self, i0: usize, i1: usize, i2: usize, i3: usize, i4: usize, i5: usize)
        -> Scalar;
    /// # Safety
    /// The caller guarantees i0 < D0, i1 < D1, i2 < D2, i3 < D3, i4 < D4, i5 < D5.
    unsafe fn get_unchecked(
        self: &Self,
        i0: usize,
        i1: usize,
        i2: usize,
        i3: usize,
        i4: usize,
        i5: usize,
    ) -> Scalar;
    /// The flat, untransformed backing buffer. Every `Rank6` implementor stores its
    /// last axis (D5) with stride 1, so a caller that wants to walk that axis can
    /// index this buffer directly with `row_offset(..) + i5` instead of paying for
    /// a full `get_unchecked` (which recomputes every stride term) on each step.
    fn get_raw_buffer(self: &Self) -> &[Scalar];
    /// Flat offset of (i0, i1, i2, i3, i4, 0) into `get_raw_buffer()`: the start of
    /// the contiguous row along the last axis.
    /// # Safety
    /// The caller guarantees i0 < D0, i1 < D1, i2 < D2, i3 < D3, i4 < D4.
    unsafe fn row_offset(
        self: &Self,
        i0: usize,
        i1: usize,
        i2: usize,
        i3: usize,
        i4: usize,
    ) -> usize;
    fn shape(self: &Self) -> [usize; 6];
}

impl<
        const BATCHES: usize,
        const GROUPS: usize,
        const CHANNELS: usize,
        const DEPTH: usize,
        const ROWS: usize,
        const COLS: usize,
        S: Storage<Tensor6DBuffer<BATCHES, GROUPS, CHANNELS, DEPTH, ROWS, COLS>>,
    > Rank6<BATCHES, GROUPS, CHANNELS, DEPTH, ROWS, COLS>
    for Tensor6D<BATCHES, GROUPS, CHANNELS, DEPTH, ROWS, COLS, S>
{
    fn get(self: &Self, b: usize, g: usize, c: usize, d: usize, i: usize, j: usize) -> Scalar {
        Tensor6D::get(self, b, g, c, d, i, j)
    }
    unsafe fn get_unchecked(
        self: &Self,
        b: usize,
        g: usize,
        c: usize,
        d: usize,
        i: usize,
        j: usize,
    ) -> Scalar {
        Tensor6D::get_unchecked(self, b, g, c, d, i, j)
    }
    fn get_raw_buffer(self: &Self) -> &[Scalar] {
        self.data.as_flat()
    }
    unsafe fn row_offset(self: &Self, b: usize, g: usize, c: usize, d: usize, i: usize) -> usize {
        b * (GROUPS * CHANNELS * DEPTH * ROWS * COLS)
            + g * (CHANNELS * DEPTH * ROWS * COLS)
            + c * (DEPTH * ROWS * COLS)
            + d * (ROWS * COLS)
            + i * COLS
    }
    fn shape(self: &Self) -> [usize; 6] {
        [BATCHES, GROUPS, CHANNELS, DEPTH, ROWS, COLS]
    }
}

impl<
        'a,
        const N: usize,
        const C: usize,
        const H: usize,
        const W: usize,
        const H_OUT: usize,
        const W_OUT: usize,
        const KH: usize,
        const KW: usize,
    > Rank6<N, H_OUT, W_OUT, C, KH, KW> for TensorView6D<'a, N, C, H, W, H_OUT, W_OUT, KH, KW>
{
    fn get(self: &Self, n: usize, i: usize, j: usize, c: usize, p: usize, q: usize) -> Scalar {
        TensorView6D::get(self, n, i, j, c, p, q)
    }
    unsafe fn get_unchecked(
        self: &Self,
        n: usize,
        i: usize,
        j: usize,
        c: usize,
        p: usize,
        q: usize,
    ) -> Scalar {
        TensorView6D::get_unchecked(self, n, i, j, c, p, q)
    }
    fn get_raw_buffer(self: &Self) -> &[Scalar] {
        self.data
    }
    unsafe fn row_offset(self: &Self, n: usize, i: usize, j: usize, c: usize, p: usize) -> usize {
        n * self.n_stride
            + i * self.h_out_stride
            + j * self.w_out_stride
            + c * self.channel_stride
            + p * self.kh_stride
            + self.reference_index
    }
    fn shape(self: &Self) -> [usize; 6] {
        self.shape
    }
}
