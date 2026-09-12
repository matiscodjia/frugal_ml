use super::{Tensor4D, Tensor4DBuffer};
use crate::linalg::storage::{Buffer, Storage};
use crate::linalg::tensor::tensor6d::TensorView6D;

impl<
        const BATCHES: usize,
        const CHANNELS: usize,
        const ROWS: usize,
        const COLS: usize,
        S: Storage<Tensor4DBuffer<BATCHES, CHANNELS, ROWS, COLS>>,
    > Tensor4D<BATCHES, CHANNELS, ROWS, COLS, S>
{
    /// Builds the im2col view of this (N x C x H x W) tensor for a KH x KW
    /// window sliding by `stride`, without copying any data.
    ///
    /// `H_OUT` and `W_OUT` must be passed explicitly (deriving them from
    /// `stride` would need `generic_const_exprs`) and are checked here against
    /// the usual convolution output size, `(H - KH) / stride + 1`.
    pub fn im2col_view<
        'a,
        const H_OUT: usize,
        const W_OUT: usize,
        const KH: usize,
        const KW: usize,
    >(
        self: &'a Self,
        stride: usize,
    ) -> TensorView6D<'a, BATCHES, CHANNELS, ROWS, COLS, H_OUT, W_OUT, KH, KW> {
        assert!(stride >= 1 && KH >= 1 && KW >= 1 && KH <= ROWS && KW <= COLS);
        assert!(H_OUT == (ROWS - KH) / stride + 1 && W_OUT == (COLS - KW) / stride + 1);
        TensorView6D {
            data: self.data.as_flat(),
            reference_index: 0,
            n_stride: CHANNELS * ROWS * COLS,
            // moving one output pixel slides the window by `stride` input pixels
            h_out_stride: stride * COLS,
            w_out_stride: stride,
            channel_stride: ROWS * COLS,
            // inside a window we walk the input row by row, element by element
            kh_stride: COLS,
            kw_stride: 1,
            shape: [BATCHES, H_OUT, W_OUT, CHANNELS, KH, KW],
        }
    }
}
