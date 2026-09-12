use super::{Rank6, Tensor6D, Tensor6DBuffer};
use crate::linalg::storage::{Buffer, OwnedStorage, Storage, StorageMut};
use crate::scalar::{sqrt, Scalar, STATS_EPSILON};

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
    pub fn mean(&self) -> Scalar {
        let n = BATCHES * GROUPS * CHANNELS * DEPTH * ROWS * COLS;
        let sum: Scalar = self.get_raw_buffer().iter().sum();
        sum / n as Scalar
    }

    /// Numerically stable: `E[(X - E[X])^2]` rather than `E[X^2] - E[X]^2`,
    /// which cancels catastrophically once the data sits far from zero (see
    /// `tests/tensor_stats_precision.rs`).
    pub fn variance(&self) -> Scalar {
        let mean = self.mean();
        let n = BATCHES * GROUPS * CHANNELS * DEPTH * ROWS * COLS;
        let sum_sq_dev: Scalar = self
            .get_raw_buffer()
            .iter()
            .map(|x| (x - mean) * (x - mean))
            .sum();
        sum_sq_dev / n as Scalar
    }

    pub fn std(&self) -> Scalar {
        sqrt(self.variance())
    }

    pub fn min(&self) -> Scalar {
        let buf = self.get_raw_buffer();
        let mut min = buf[0];
        for &x in &buf[1..] {
            if x < min {
                min = x;
            }
        }
        min
    }
    pub fn max(&self) -> Scalar {
        let buf = self.get_raw_buffer();
        let mut max = buf[0];
        for &x in &buf[1..] {
            if x > max {
                max = x;
            }
        }
        max
    }

    /// Z-score normalization in place, using this tensor's own `mean`/`std`.
    pub fn standardize(&mut self)
    where
        S: StorageMut<Tensor6DBuffer<BATCHES, GROUPS, CHANNELS, DEPTH, ROWS, COLS>>,
    {
        let mean = self.mean();
        let std = self.std();
        self.standardize_with(mean, std);
    }
    /// [`Self::standardize`], but returns a new tensor instead of mutating
    /// `self`.
    pub fn standardized(&self) -> Self
    where
        S: OwnedStorage<Tensor6DBuffer<BATCHES, GROUPS, CHANNELS, DEPTH, ROWS, COLS>>,
    {
        let mut result = self.copied();
        result.standardize();
        result
    }
    /// Z-score normalization in place, against externally supplied
    /// `mean`/`std` (e.g. statistics fit on a training set, then applied
    /// unchanged to validation/test data). `std` is floored by
    /// [`STATS_EPSILON`] so a constant (zero-variance) tensor scales to a
    /// large finite value instead of `NaN`/`inf`.
    pub fn standardize_with(&mut self, mean: Scalar, std: Scalar)
    where
        S: StorageMut<Tensor6DBuffer<BATCHES, GROUPS, CHANNELS, DEPTH, ROWS, COLS>>,
    {
        let std = std + STATS_EPSILON;
        for x in self.data.as_flat_mut() {
            *x = (*x - mean) / std;
        }
    }
    /// [`Self::standardize_with`], but returns a new tensor instead of
    /// mutating `self`.
    pub fn standardized_with(&self, mean: Scalar, std: Scalar) -> Self
    where
        S: OwnedStorage<Tensor6DBuffer<BATCHES, GROUPS, CHANNELS, DEPTH, ROWS, COLS>>,
    {
        let mut result = self.copied();
        result.standardize_with(mean, std);
        result
    }

    /// Min-max scaling in place, using this tensor's own `min`/`max`: rescales
    /// into `[0, 1]`.
    pub fn min_max_scale(&mut self)
    where
        S: StorageMut<Tensor6DBuffer<BATCHES, GROUPS, CHANNELS, DEPTH, ROWS, COLS>>,
    {
        let min = self.min();
        let max = self.max();
        self.min_max_scale_with(min, max);
    }
    /// [`Self::min_max_scale`], but returns a new tensor instead of mutating
    /// `self`.
    pub fn min_max_scaled(&self) -> Self
    where
        S: OwnedStorage<Tensor6DBuffer<BATCHES, GROUPS, CHANNELS, DEPTH, ROWS, COLS>>,
    {
        let mut result = self.copied();
        result.min_max_scale();
        result
    }
    /// Min-max scaling in place, against an externally supplied `min`/`max`
    /// (e.g. bounds fit on a training set, then applied unchanged to
    /// validation/test data). `max - min` is floored by [`STATS_EPSILON`]
    /// so a degenerate (constant) tensor scales to a large finite value
    /// instead of `NaN`/`inf`.
    pub fn min_max_scale_with(&mut self, min: Scalar, max: Scalar)
    where
        S: StorageMut<Tensor6DBuffer<BATCHES, GROUPS, CHANNELS, DEPTH, ROWS, COLS>>,
    {
        let range = (max - min) + STATS_EPSILON;
        for x in self.data.as_flat_mut() {
            *x = (*x - min) / range;
        }
    }
    /// [`Self::min_max_scale_with`], but returns a new tensor instead of
    /// mutating `self`.
    pub fn min_max_scaled_with(&self, min: Scalar, max: Scalar) -> Self
    where
        S: OwnedStorage<Tensor6DBuffer<BATCHES, GROUPS, CHANNELS, DEPTH, ROWS, COLS>>,
    {
        let mut result = self.copied();
        result.min_max_scale_with(min, max);
        result
    }

    /// Bitwise copy into a freshly `zeroed()` tensor: the shared first step
    /// of every `*_with`-less "new tensor" scaling variant (`standardized`,
    /// `min_max_scaled`, ...), which all need an owned clone of `self` to
    /// mutate in place before handing it back.
    fn copied(&self) -> Self
    where
        S: OwnedStorage<Tensor6DBuffer<BATCHES, GROUPS, CHANNELS, DEPTH, ROWS, COLS>>,
    {
        let mut result = Self::zeroed();
        result
            .data
            .as_flat_mut()
            .copy_from_slice(self.data.as_flat());
        result
    }
}
