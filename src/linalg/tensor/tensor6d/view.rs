use super::TensorView6D;
use crate::scalar::Scalar;

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
    > TensorView6D<'a, N, C, H, W, H_OUT, W_OUT, KH, KW>
{
    /// Logical axes: (N x H_OUT x W_OUT x C x KH x KW), the operand order
    /// expected by `tensordot_3`. The underlying buffer stays the (N x C x H x W)
    /// input tensor: only the strides move.
    pub fn get(self: &Self, n: usize, i: usize, j: usize, c: usize, p: usize, q: usize) -> Scalar {
        debug_assert!(
            n < self.shape[0]
                && i < self.shape[1]
                && j < self.shape[2]
                && c < self.shape[3]
                && p < self.shape[4]
                && q < self.shape[5]
        );
        let flat_index: usize = n * self.n_stride
            + i * self.h_out_stride
            + j * self.w_out_stride
            + c * self.channel_stride
            + p * self.kh_stride
            + q * self.kw_stride;
        let index: usize = flat_index + self.reference_index;
        self.data[index]
    }
    /// # Safety
    /// The caller guarantees n < N, i < H_OUT, j < W_OUT, c < C, p < KH, q < KW.
    pub unsafe fn get_unchecked(
        &self,
        n: usize,
        i: usize,
        j: usize,
        c: usize,
        p: usize,
        q: usize,
    ) -> Scalar {
        let flat_index: usize = n * self.n_stride
            + i * self.h_out_stride
            + j * self.w_out_stride
            + c * self.channel_stride
            + p * self.kh_stride
            + q * self.kw_stride;
        let index: usize = flat_index + self.reference_index;

        *self.data.get_unchecked(index)
    }
}
