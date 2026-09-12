use super::Tensor;
use crate::linalg::storage::{Buffer, OwnedStorage, Storage};
use crate::scalar::{fabs, pow, sqrt, Scalar};
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
impl<const ROWS: usize, const COLS: usize, S: Storage<[[Scalar; COLS]; ROWS]>>
    Tensor<ROWS, COLS, S>
{
    pub fn pow(&self, n: f32) -> Self
    where
        S: OwnedStorage<[[Scalar; COLS]; ROWS]>,
    {
        let mut pow_tensor = Self::zeroed();
        for i in 0..ROWS {
            for j in 0..COLS {
                unsafe {
                    let elem = self.get_unchecked(i, j);
                    pow_tensor.set_unchecked(i, j, pow(elem, n));
                }
            }
        }
        pow_tensor
    }
    /// Elementwise square root, via `crate::scalar::sqrt` (`libm`) rather
    /// than `f32`/`f64`'s inherent `sqrt`, which only exists once `std` is
    /// linked and would otherwise break the bare-metal, `no_std` build.
    pub fn sqrt(&self) -> Self
    where
        S: OwnedStorage<[[Scalar; COLS]; ROWS]>,
    {
        let mut sqrt_tensor = Self::zeroed();
        for i in 0..ROWS {
            for j in 0..COLS {
                unsafe {
                    sqrt_tensor.set_unchecked(i, j, sqrt(self.get_unchecked(i, j)));
                }
            }
        }
        sqrt_tensor
    }
}
