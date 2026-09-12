use super::Tensor;
use crate::linalg::storage::{Buffer, Storage, StorageMut};
use crate::scalar::Scalar;

impl<const ROWS: usize, const COLS: usize, S: Storage<[[Scalar; COLS]; ROWS]>>
    Tensor<ROWS, COLS, S>
{
    pub fn get(self: &Self, i: usize, j: usize) -> Scalar {
        debug_assert!(i < ROWS && j < COLS);
        let flat_index: usize = i * COLS + j;
        self.data.as_flat()[flat_index]
    }
    pub fn set(self: &mut Self, i: usize, j: usize, value: Scalar) -> ()
    where
        S: StorageMut<[[Scalar; COLS]; ROWS]>,
    {
        debug_assert!(i < ROWS && j < COLS);
        let flat_index: usize = i * COLS + j;
        self.data.as_flat_mut()[flat_index] = value;
    }
    /// # Safety
    /// The caller guarantees i < ROWS, j < COLS.
    pub unsafe fn get_unchecked(self: &Self, i: usize, j: usize) -> Scalar {
        let flat_index: usize = i * COLS + j;
        *self.data.as_flat().get_unchecked(flat_index)
    }
    /// # Safety
    /// The caller guarantees i < ROWS, j < COLS.
    pub unsafe fn set_unchecked(self: &mut Self, i: usize, j: usize, value: Scalar) -> ()
    where
        S: StorageMut<[[Scalar; COLS]; ROWS]>,
    {
        let flat_index: usize = i * COLS + j;
        *self.data.as_flat_mut().get_unchecked_mut(flat_index) = value;
    }
    /// The flat, untransformed backing buffer. The last axis (COLS) always has
    /// stride 1, so a caller that wants to walk it can index this buffer
    /// directly with `row_offset(..) + j` instead of paying for a full
    /// `get_unchecked` (which recomputes every stride term) on each step.
    pub fn get_raw_buffer(&self) -> &[Scalar] {
        self.data.as_flat()
    }
    /// Flat offset of (i, 0) into `get_raw_buffer()`: the start of the
    /// contiguous row along the last axis.
    /// # Safety
    /// The caller guarantees i < ROWS.
    pub unsafe fn row_offset(self: &Self, i: usize) -> usize {
        i * COLS
    }
}
