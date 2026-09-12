use super::Tensor;
use crate::linalg::storage::{OwnedStorage, Storage};
use crate::scalar::{fabs, sqrt, Scalar};

impl<const ROWS: usize, const COLS: usize, S: Storage<[[Scalar; COLS]; ROWS]>>
    Tensor<ROWS, COLS, S>
{
    /// Accumulates the product of `a * b` into `self`: `self += a * b`.
    pub fn matmul_accumulate<const K: usize>(&mut self, a: &Tensor<ROWS, K>, b: &Tensor<K, COLS>)
    where
        S: OwnedStorage<[[Scalar; COLS]; ROWS]>,
    {
        for i in 0..ROWS {
            for j in 0..COLS {
                let mut sum: Scalar = 0.0;
                for k in 0..K {
                    sum += a.get(i, k) * b.get(k, j);
                }
                let prev = self.get(i, j);
                self.set(i, j, prev + sum);
            }
        }
    }
    /// Matrix product: `self` (ROWS x COLS) * `other` (COLS x P) -> (ROWS x P).
    /// Also serves as matrix-vector product once `Vector<N> = Tensor<N, 1>`.
    ///
    /// A method rather than `Mul`: `Mul::Output` can't be inferred here (a
    /// free const generic on a trait impl, unlike on a plain function/method,
    /// must be constrained by `Self`/`Rhs`, so `P` can't come from `Output`
    /// alone, `E0207`).
    pub fn multiply<const P: usize>(&self, other: &Tensor<COLS, P>) -> Tensor<ROWS, P> {
        let mut result = Tensor::<ROWS, P>::zeroed();
        for i in 0..ROWS {
            for j in 0..P {
                let mut sum: Scalar = 0.0;
                for k in 0..COLS {
                    sum += self.get(i, k) * other.get(k, j);
                }
                result.set(i, j, sum);
            }
        }
        result
    }
}

/// The column-vector shape: a "vector" is just a `Tensor` with one column.
/// See the [`super::Vector`] alias.
impl<const N: usize, S: Storage<[[Scalar; 1]; N]>> Tensor<N, 1, S> {
    /// Returns the dimension of the vector (`Vector::dim`'s equivalent).
    pub const fn dim(&self) -> usize {
        N
    }
    pub fn dot(&self, other: &Self) -> Scalar {
        let mut sum: Scalar = 0.0;
        for i in 0..N {
            sum += self.get(i, 0) * other.get(i, 0);
        }
        sum
    }
    pub fn l2_norm(&self) -> Scalar {
        sqrt(self.dot(self))
    }
    pub fn l1_norm(&self) -> Scalar {
        let mut sum: Scalar = 0.0;
        for i in 0..N {
            sum += fabs(self.get(i, 0));
        }
        sum
    }
    pub fn inf_norm(&self) -> Scalar {
        let mut max: Scalar = 0.0;
        for i in 0..N {
            let abs_val = fabs(self.get(i, 0));
            if abs_val > max {
                max = abs_val;
            }
        }
        max
    }
    pub fn orthogonal_projection(&self, other: &Self) -> Self
    where
        S: OwnedStorage<[[Scalar; 1]; N]> + Copy,
    {
        let scale_factor = other.dot(other);
        if fabs(scale_factor) < 1e-8 {
            return Self::zeroed();
        }
        let ratio = self.dot(other) / scale_factor;
        *other * ratio
    }
    pub fn sum(&self) -> Scalar {
        let mut s: Scalar = 0.0;
        for i in 0..N {
            s += self.get(i, 0);
        }
        s
    }
    pub fn hadamard(&self, other: &Self) -> Self
    where
        S: OwnedStorage<[[Scalar; 1]; N]>,
    {
        let mut result = Self::zeroed();
        for i in 0..N {
            result.set(i, 0, self.get(i, 0) * other.get(i, 0));
        }
        result
    }
}
