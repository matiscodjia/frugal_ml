use super::{Tensor, TensorView};
use crate::linalg::storage::{Buffer, OwnedStorage, Storage};
use crate::scalar::Scalar;

impl<const ROWS: usize, const COLS: usize, S: Storage<[[Scalar; COLS]; ROWS]>>
    Tensor<ROWS, COLS, S>
{
    pub fn view<'a>(
        self: &'a Self,
        lines: (usize, usize),
        columns: (usize, usize),
    ) -> TensorView<'a> {
        let max_line = lines.1;
        let max_col = columns.1;
        assert!(max_line < ROWS && max_col < COLS && lines.0 <= lines.1 && columns.0 <= columns.1);
        let reference_index = lines.0 * COLS + columns.0;
        let view_shape = (lines.1 - lines.0 + 1, columns.1 - columns.0 + 1);
        TensorView {
            data: self.data.as_flat(),
            reference_index,
            row_stride: COLS,
            col_stride: 1,
            shape: view_shape,
        }
    }
    pub const fn rows(&self) -> usize {
        ROWS
    }
    pub const fn cols(&self) -> usize {
        COLS
    }
    /// Extracts a column as a `Tensor<ROWS, 1>` (a [`super::Vector`]).
    pub fn get_col(&self, col: usize) -> Option<Tensor<ROWS, 1>> {
        if col >= COLS {
            return None;
        }
        let mut result = Tensor::<ROWS, 1>::zeroed();
        for i in 0..ROWS {
            result.set(i, 0, self.get(i, col));
        }
        Some(result)
    }
    /// Injects a `Tensor<ROWS, 1>` (a [`super::Vector`]) into a matrix column.
    /// # Panics
    /// Panics if `col >= COLS`.
    pub fn set_col(&mut self, col: usize, vec: &Tensor<ROWS, 1>)
    where
        S: OwnedStorage<[[Scalar; COLS]; ROWS]>,
    {
        assert!(col < COLS, "Column index out of bounds");
        for i in 0..ROWS {
            self.set(i, col, vec.get(i, 0));
        }
    }
    /// Returns the transpose as a new tensor: this changes the static shape
    /// from `(ROWS, COLS)` to `(COLS, ROWS)`. Always returns default
    /// storage: `S` is tied to this tensor's own `(ROWS, COLS)` buffer shape
    /// and can't be reused for the transposed `(COLS, ROWS)` one. There is no
    /// in-place counterpart: `ROWS`/`COLS` are fixed for the tensor's whole
    /// lifetime (no runtime shape/stride state to swap), so transposing
    /// without allocating a new tensor would need a different type.
    pub fn transposed(&self) -> Tensor<COLS, ROWS> {
        let mut result = Tensor::<COLS, ROWS>::zeroed();
        for i in 0..ROWS {
            for j in 0..COLS {
                result.set(j, i, self.get(i, j));
            }
        }
        result
    }
}
