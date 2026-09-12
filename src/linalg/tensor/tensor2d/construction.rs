use super::Tensor;
use crate::linalg::storage::{Buffer, LenMismatch, OwnedStorage, Storage, StorageMut};
use crate::scalar::Scalar;

impl<const ROWS: usize, const COLS: usize, S: Storage<[[Scalar; COLS]; ROWS]>>
    Tensor<ROWS, COLS, S>
{
    /// Zero-initialized accumulator for the crate's own algorithms (`identity`,
    /// `multiply`, `qr_decomposition`, `tensordot_*`...), which build a result
    /// element by element and can't hand `new` the full data upfront. Kept out
    /// of the public API on purpose: an external caller of the crate should
    /// never be able to get a silently-zeroed tensor by omitting a load step.
    pub(crate) fn zeroed() -> Self
    where
        S: OwnedStorage<[[Scalar; COLS]; ROWS]>,
    {
        Self { data: S::zeroed() }
    }
    /// Builds a tensor from data known upfront, the caller never needs `mut`
    /// or a separate load step.
    ///
    /// Takes the buffer's own nested shape (`[[Scalar; COLS]; ROWS]`, row by
    /// row) rather than a flat `[Scalar; ROWS * COLS]`: that flat form can't
    /// be spelled as an array length here without the unstable
    /// `generic_const_exprs`, and the nested one is exactly [`Buffer`]'s
    /// shape, so the length is verified at compile time, not checked at
    /// runtime.
    pub fn new(data: [[Scalar; COLS]; ROWS]) -> Self
    where
        S: OwnedStorage<[[Scalar; COLS]; ROWS]>,
    {
        let mut t = Self::zeroed();
        t.load_data(data);
        t
    }
    /// Loads a full buffer's worth of data, for small tensors and known
    /// data. For a tensor too big to build `[[Scalar; COLS]; ROWS]` on the
    /// stack, see [`Self::load_slice`] (`no_std`, no `alloc`) or
    /// [`Self::from_vec`].
    pub fn load_data(self: &mut Self, data: [[Scalar; COLS]; ROWS]) -> ()
    where
        S: StorageMut<[[Scalar; COLS]; ROWS]>,
    {
        self.data.as_flat_mut().copy_from_slice(data.as_flat());
    }
    /// Copies from a runtime-sized slice, never materializes a full
    /// `ROWS * COLS` array on the stack, so it's the door for a large
    /// tensor without `alloc` (a table already sitting in flash or in a
    /// driver-filled buffer).
    pub fn load_slice(self: &mut Self, data: &[Scalar]) -> Result<(), LenMismatch>
    where
        S: StorageMut<[[Scalar; COLS]; ROWS]>,
    {
        if data.len() != ROWS * COLS {
            return Err(LenMismatch);
        }
        self.data.as_flat_mut().copy_from_slice(data);
        Ok(())
    }
    /// Builds a tensor straight from a `Vec`, the door for the `.npy`
    /// pipeline, where the data doesn't exist as a compile-time-sized array in
    /// the first place.
    ///
    /// Hands the `Vec` back intact if its length doesn't match `ROWS * COLS`,
    /// rather than dumping it into a panic message.
    #[cfg(feature = "alloc")]
    pub fn from_vec(data: alloc::vec::Vec<Scalar>) -> Result<Self, alloc::vec::Vec<Scalar>>
    where
        S: OwnedStorage<[[Scalar; COLS]; ROWS]>,
    {
        if data.len() != ROWS * COLS {
            return Err(data);
        }
        let mut t = Self::zeroed();
        t.data.as_flat_mut().copy_from_slice(&data);
        Ok(t)
    }
    /// Builds a tensor from an array of column vectors.
    pub fn from_cols(cols: [Tensor<ROWS, 1>; COLS]) -> Self
    where
        S: OwnedStorage<[[Scalar; COLS]; ROWS]>,
    {
        let mut mat = Self::zeroed();
        for j in 0..COLS {
            mat.set_col(j, &cols[j]);
        }
        mat
    }
}

impl<const SIZE: usize, S: Storage<[[Scalar; SIZE]; SIZE]>> Tensor<SIZE, SIZE, S> {
    pub fn identity() -> Self
    where
        S: OwnedStorage<[[Scalar; SIZE]; SIZE]>,
    {
        let mut result = Self::zeroed();
        for i in 0..SIZE {
            result.set(i, i, 1.0);
        }
        result
    }
}

/// The column-vector shape: a "vector" is just a `Tensor` with one column.
/// See the [`super::Vector`] alias.
impl<const N: usize, S: Storage<[[Scalar; 1]; N]>> Tensor<N, 1, S> {
    /// Builds a column tensor from a flat array, the `Vector::new(data)`
    /// equivalent (can't reuse the name `new`, already taken by the general
    /// impl's data constructor with a slightly different signature).
    pub fn from_data(data: [Scalar; N]) -> Self
    where
        S: OwnedStorage<[[Scalar; 1]; N]>,
    {
        let mut t = Self::zeroed();
        for i in 0..N {
            t.set(i, 0, data[i]);
        }
        t
    }
}
