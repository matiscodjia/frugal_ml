use super::Tensor;
use crate::linalg::storage::{Buffer, Storage, StorageMut};
use crate::scalar::Scalar;
use core::ops::{Index, IndexMut};

impl<const N: usize, S: Storage<[[Scalar; 1]; N]>> Index<usize> for Tensor<N, 1, S> {
    type Output = Scalar;
    fn index(&self, i: usize) -> &Self::Output {
        &self.data.as_flat()[i]
    }
}
impl<const N: usize, S: StorageMut<[[Scalar; 1]; N]>> IndexMut<usize> for Tensor<N, 1, S> {
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {
        &mut self.data.as_flat_mut()[i]
    }
}

impl<const ROWS: usize, const COLS: usize, S: Storage<[[Scalar; COLS]; ROWS]>>
    Index<(usize, usize)> for Tensor<ROWS, COLS, S>
{
    type Output = Scalar;
    fn index(&self, (i, j): (usize, usize)) -> &Self::Output {
        &self.data.as_flat()[i * COLS + j]
    }
}
impl<const ROWS: usize, const COLS: usize, S: StorageMut<[[Scalar; COLS]; ROWS]>>
    IndexMut<(usize, usize)> for Tensor<ROWS, COLS, S>
{
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut Self::Output {
        &mut self.data.as_flat_mut()[i * COLS + j]
    }
}
