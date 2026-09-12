use super::{Tensor6D, Tensor6DBuffer};
use crate::linalg::storage::{Buffer, Storage, StorageMut};
use crate::scalar::Scalar;

impl<
        const BATCHES: usize,
        const GROUPS: usize,
        const CHANNELS: usize,
        const DEPTH: usize,
        const ROWS: usize,
        const COLS: usize,
        S: Storage<Tensor6DBuffer<BATCHES, GROUPS, CHANNELS, DEPTH, ROWS, COLS>>,
    > Tensor6D<BATCHES, GROUPS, CHANNELS, DEPTH, ROWS, COLS, S>
{
    pub fn get(self: &Self, b: usize, g: usize, c: usize, d: usize, i: usize, j: usize) -> Scalar {
        debug_assert!(b < BATCHES && g < GROUPS && c < CHANNELS && d < DEPTH && i < ROWS && j < COLS);
        let flat_index: usize = b * (GROUPS * CHANNELS * DEPTH * ROWS * COLS)
            + g * (CHANNELS * DEPTH * ROWS * COLS)
            + c * (DEPTH * ROWS * COLS)
            + d * (ROWS * COLS)
            + i * COLS
            + j;
        self.data.as_flat()[flat_index]
    }
    pub fn set(
        self: &mut Self,
        b: usize,
        g: usize,
        c: usize,
        d: usize,
        i: usize,
        j: usize,
        value: Scalar,
    ) -> ()
    where
        S: StorageMut<Tensor6DBuffer<BATCHES, GROUPS, CHANNELS, DEPTH, ROWS, COLS>>,
    {
        debug_assert!(b < BATCHES && g < GROUPS && c < CHANNELS && d < DEPTH && i < ROWS && j < COLS);
        let flat_index: usize = b * (GROUPS * CHANNELS * DEPTH * ROWS * COLS)
            + g * (CHANNELS * DEPTH * ROWS * COLS)
            + c * (DEPTH * ROWS * COLS)
            + d * (ROWS * COLS)
            + i * COLS
            + j;
        self.data.as_flat_mut()[flat_index] = value;
    }
    /// # Safety
    /// The caller guarantees b < BATCHES, g < GROUPS, c < CHANNELS, d < DEPTH,
    /// i < ROWS, j < COLS.
    pub unsafe fn get_unchecked(
        self: &Self,
        b: usize,
        g: usize,
        c: usize,
        d: usize,
        i: usize,
        j: usize,
    ) -> Scalar {
        let flat_index: usize = b * (GROUPS * CHANNELS * DEPTH * ROWS * COLS)
            + g * (CHANNELS * DEPTH * ROWS * COLS)
            + c * (DEPTH * ROWS * COLS)
            + d * (ROWS * COLS)
            + i * COLS
            + j;
        *self.data.as_flat().get_unchecked(flat_index)
    }
    /// # Safety
    /// The caller guarantees b < BATCHES, g < GROUPS, c < CHANNELS, d < DEPTH,
    /// i < ROWS, j < COLS.
    pub unsafe fn set_unchecked(
        self: &mut Self,
        b: usize,
        g: usize,
        c: usize,
        d: usize,
        i: usize,
        j: usize,
        value: Scalar,
    ) -> ()
    where
        S: StorageMut<Tensor6DBuffer<BATCHES, GROUPS, CHANNELS, DEPTH, ROWS, COLS>>,
    {
        let flat_index: usize = b * (GROUPS * CHANNELS * DEPTH * ROWS * COLS)
            + g * (CHANNELS * DEPTH * ROWS * COLS)
            + c * (DEPTH * ROWS * COLS)
            + d * (ROWS * COLS)
            + i * COLS
            + j;
        *self.data.as_flat_mut().get_unchecked_mut(flat_index) = value;
    }
}
