use crate::linalg::Storage;

use super::tensor2d::Tensor;
use crate::scalar::Scalar;

impl<const ROWS: usize, const COLS: usize, S: Storage<[[Scalar; COLS]; ROWS]>>
    Tensor<ROWS, COLS, S>
{
    pub fn mean(self: &Self, dim: usize) -> Scalar {
        if dim == 0 {
            todo!()
        }
        todo!()
    }
}
