use super::Tensor;
use crate::linalg::storage::{Buffer, OwnedStorage, Storage};
use crate::scalar::{fabs, Scalar};
use core::ops::{Add, Div, Mul, Neg, Sub};

impl<const ROWS: usize, const COLS: usize, S: OwnedStorage<[[Scalar; COLS]; ROWS]>> Add
    for Tensor<ROWS, COLS, S>
{
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        let mut result = Self::zeroed();
        let (out, a, b) = (
            result.data.as_flat_mut(),
            self.data.as_flat(),
            rhs.data.as_flat(),
        );
        for k in 0..out.len() {
            out[k] = a[k] + b[k];
        }
        result
    }
}
impl<const ROWS: usize, const COLS: usize, S: OwnedStorage<[[Scalar; COLS]; ROWS]>> Sub
    for Tensor<ROWS, COLS, S>
{
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        let mut result = Self::zeroed();
        let (out, a, b) = (
            result.data.as_flat_mut(),
            self.data.as_flat(),
            rhs.data.as_flat(),
        );
        for k in 0..out.len() {
            out[k] = a[k] - b[k];
        }
        result
    }
}
impl<const ROWS: usize, const COLS: usize, S: OwnedStorage<[[Scalar; COLS]; ROWS]>> Neg
    for Tensor<ROWS, COLS, S>
{
    type Output = Self;
    fn neg(self) -> Self::Output {
        self * -1.0
    }
}
impl<const ROWS: usize, const COLS: usize, S: OwnedStorage<[[Scalar; COLS]; ROWS]>> Mul<Scalar>
    for Tensor<ROWS, COLS, S>
{
    type Output = Self;
    fn mul(self, rhs: Scalar) -> Self::Output {
        let mut result = Self::zeroed();
        let (out, a) = (result.data.as_flat_mut(), self.data.as_flat());
        for k in 0..out.len() {
            out[k] = a[k] * rhs;
        }
        result
    }
}
impl<const ROWS: usize, const COLS: usize, S: OwnedStorage<[[Scalar; COLS]; ROWS]>> Div<Scalar>
    for Tensor<ROWS, COLS, S>
{
    type Output = Self;
    fn div(self, rhs: Scalar) -> Self::Output {
        self * (1.0 / rhs)
    }
}

impl<const ROWS: usize, const COLS: usize, S: Storage<[[Scalar; COLS]; ROWS]>> PartialEq
    for Tensor<ROWS, COLS, S>
{
    fn eq(&self, other: &Self) -> bool {
        let epsilon = 1e-5;
        let (a, b) = (self.data.as_flat(), other.data.as_flat());
        for k in 0..a.len() {
            if fabs(a[k] - b[k]) >= epsilon {
                return false;
            }
        }
        true
    }
}
