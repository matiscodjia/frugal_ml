use super::{Tensor4D, Tensor4DBuffer};
use crate::linalg::storage::{Buffer, Storage, StorageMut};
use crate::scalar::Scalar;

impl<
        const BATCHES: usize,
        const CHANNELS: usize,
        const ROWS: usize,
        const COLS: usize,
        S: Storage<Tensor4DBuffer<BATCHES, CHANNELS, ROWS, COLS>>,
    > Tensor4D<BATCHES, CHANNELS, ROWS, COLS, S>
{
    pub fn get_data(&self) -> &[Scalar] {
        self.data.as_flat()
    }

    pub fn get_shape(&self) -> [usize; 4] {
        [BATCHES, CHANNELS, ROWS, COLS]
    }

    pub fn get(self: &Self, b: usize, c: usize, i: usize, j: usize) -> Scalar {
        debug_assert!(b < BATCHES && c < CHANNELS && i < ROWS && j < COLS);
        let flat_index: usize =
            b * (CHANNELS * ROWS * COLS) + c * (ROWS * COLS) + i * COLS + j;
        self.data.as_flat()[flat_index]
    }
    pub fn set(self: &mut Self, b: usize, c: usize, i: usize, j: usize, value: Scalar) -> ()
    where
        S: StorageMut<Tensor4DBuffer<BATCHES, CHANNELS, ROWS, COLS>>,
    {
        debug_assert!(b < BATCHES && c < CHANNELS && i < ROWS && j < COLS);
        let flat_index: usize =
            b * (CHANNELS * ROWS * COLS) + c * (ROWS * COLS) + i * COLS + j;
        self.data.as_flat_mut()[flat_index] = value;
    }
    /// # Safety
    /// The caller guarantees b < BATCHES, c < CHANNELS, i < ROWS, j < COLS.
    pub unsafe fn get_unchecked(self: &Self, b: usize, c: usize, i: usize, j: usize) -> Scalar {
        let flat_index: usize =
            b * (CHANNELS * ROWS * COLS) + c * (ROWS * COLS) + i * COLS + j;
        *self.data.as_flat().get_unchecked(flat_index)
    }
    /// # Safety
    /// The caller guarantees b < BATCHES, c < CHANNELS, i < ROWS, j < COLS.
    pub unsafe fn set_unchecked(
        self: &mut Self,
        b: usize,
        c: usize,
        i: usize,
        j: usize,
        value: Scalar,
    ) -> ()
    where
        S: StorageMut<Tensor4DBuffer<BATCHES, CHANNELS, ROWS, COLS>>,
    {
        let flat_index: usize =
            b * (CHANNELS * ROWS * COLS) + c * (ROWS * COLS) + i * COLS + j;
        *self.data.as_flat_mut().get_unchecked_mut(flat_index) = value;
    }
    /// The flat, untransformed backing buffer. The last axis (COLS) always has
    /// stride 1, so a caller that wants to walk it can index this buffer
    /// directly with `row_offset(..) + j` instead of paying for a full
    /// `get_unchecked` (which recomputes every stride term) on each step.
    pub fn get_raw_buffer(&self) -> &[Scalar] {
        self.data.as_flat()
    }
    /// Flat offset of (b, c, i, 0) into `get_raw_buffer()`: the start of the
    /// contiguous row along the last axis.
    /// # Safety
    /// The caller guarantees b < BATCHES, c < CHANNELS, i < ROWS.
    pub unsafe fn row_offset(self: &Self, b: usize, c: usize, i: usize) -> usize {
        b * (CHANNELS * ROWS * COLS) + c * (ROWS * COLS) + i * COLS
    }
}
