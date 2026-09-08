
use crate::linalg::Storage;

use super::tensor2d::Tensor
use super::tensor3d
use super::tensor4d
use crate::scalar::{fabs, sqrt, Scalar}
impl <const ROWS: usize, const COLS: usize, const NUMEL: usize, S:Storage<[Scalar; NUMEL]>> Tensor<ROWS, COLS, NUMEL, S>{
    
    pub fn mean(self: &Self, dim: usize) ->  {
        if (dim == 0)
    }
}
