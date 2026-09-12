use super::Tensor3D;
use crate::linalg::storage::{Buffer, LenMismatch, OwnedStorage, Storage, StorageMut};
use crate::scalar::Scalar;

impl<
        const CHANNELS: usize,
        const ROWS: usize,
        const COLS: usize,
        S: Storage<[[[Scalar; COLS]; ROWS]; CHANNELS]>,
    > Tensor3D<CHANNELS, ROWS, COLS, S>
{
    /// Zero-initialized accumulator for the crate's own algorithms (e.g.
    /// `sp::kernels::replicate`), which build a result element by element.
    /// Kept out of the public API on purpose: an external caller should never
    /// get a silently-zeroed tensor by omitting a load step.
    pub(crate) fn zeroed() -> Self
    where
        S: OwnedStorage<[[[Scalar; COLS]; ROWS]; CHANNELS]>,
    {
        Self { data: S::zeroed() }
    }
    /// Builds a tensor from data known upfront, the caller never needs `mut`
    /// or a separate load step.
    ///
    /// Takes the buffer's own nested shape (channel by channel, row by row)
    /// rather than a flat array: the flat form can't be spelled as an array
    /// length here without the unstable `generic_const_exprs`, and the
    /// nested one is exactly [`Buffer`]'s shape, so the length is verified
    /// at compile time, not checked at runtime.
    pub fn new(data: [[[Scalar; COLS]; ROWS]; CHANNELS]) -> Self
    where
        S: OwnedStorage<[[[Scalar; COLS]; ROWS]; CHANNELS]>,
    {
        let mut t = Self::zeroed();
        t.load_data(data);
        t
    }
    /// Loads a full buffer's worth of data, for small tensors and known
    /// data. For a tensor too big to build the full nested array on the
    /// stack, see [`Self::load_slice`] (`no_std`, no `alloc`) or
    /// [`Self::from_vec`].
    pub fn load_data(self: &mut Self, data: [[[Scalar; COLS]; ROWS]; CHANNELS]) -> ()
    where
        S: StorageMut<[[[Scalar; COLS]; ROWS]; CHANNELS]>,
    {
        self.data.as_flat_mut().copy_from_slice(data.as_flat());
    }
    /// Copies from a runtime-sized slice, never materializes a full
    /// `CHANNELS * ROWS * COLS` array on the stack.
    pub fn load_slice(self: &mut Self, data: &[Scalar]) -> Result<(), LenMismatch>
    where
        S: StorageMut<[[[Scalar; COLS]; ROWS]; CHANNELS]>,
    {
        if data.len() != CHANNELS * ROWS * COLS {
            return Err(LenMismatch);
        }
        self.data.as_flat_mut().copy_from_slice(data);
        Ok(())
    }
    /// Builds a tensor straight from a `Vec`, no compile-time-sized array
    /// ever materialized.
    #[cfg(feature = "alloc")]
    pub fn from_vec(data: alloc::vec::Vec<Scalar>) -> Result<Self, alloc::vec::Vec<Scalar>>
    where
        S: OwnedStorage<[[[Scalar; COLS]; ROWS]; CHANNELS]>,
    {
        if data.len() != CHANNELS * ROWS * COLS {
            return Err(data);
        }
        let mut t = Self::zeroed();
        t.data.as_flat_mut().copy_from_slice(&data);
        Ok(t)
    }
}
