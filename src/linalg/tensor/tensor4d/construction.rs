use super::{Tensor4D, Tensor4DBuffer};
use crate::linalg::storage::{Buffer, LenMismatch, OwnedStorage, Storage, StorageMut};
use crate::scalar::Scalar;

impl<
        const BATCHES: usize,
        const CHANNELS: usize,
        const ROWS: usize,
        const COLS: usize,
        S: Storage<Tensor4DBuffer<BATCHES, CHANNELS, ROWS, COLS>>,
    > Tensor4D<BATCHES, CHANNELS, ROWS, COLS, S>
{
    /// Zero-initialized accumulator for the crate's own algorithms
    /// (`tensordot_3`, `sp::kernels::filter_bank`), which build a result
    /// element by element. Kept out of the public API on purpose: an external
    /// caller should never get a silently-zeroed tensor by omitting a load
    /// step.
    pub(crate) fn zeroed() -> Self
    where
        S: OwnedStorage<Tensor4DBuffer<BATCHES, CHANNELS, ROWS, COLS>>,
    {
        Self { data: S::zeroed() }
    }
    /// Builds a tensor from data known upfront, the caller never needs `mut`
    /// or a separate load step.
    ///
    /// Takes the buffer's own nested shape ([`Tensor4DBuffer`]: batch by
    /// batch, channel by channel, row by row) rather than a flat array: the
    /// flat form can't be spelled as an array length here without the
    /// unstable `generic_const_exprs`, and the nested one is exactly
    /// [`Buffer`]'s shape, so the length is verified at compile time, not
    /// checked at runtime.
    pub fn new(data: Tensor4DBuffer<BATCHES, CHANNELS, ROWS, COLS>) -> Self
    where
        S: OwnedStorage<Tensor4DBuffer<BATCHES, CHANNELS, ROWS, COLS>>,
    {
        let mut t = Self::zeroed();
        t.load_data(data);
        t
    }
    /// Loads a full buffer's worth of data, for small tensors and known
    /// data. For a tensor too big to build the full nested array on the
    /// stack, see [`Self::load_slice`] (`no_std`, no `alloc`) or
    /// [`Self::from_vec`].
    pub fn load_data(self: &mut Self, data: Tensor4DBuffer<BATCHES, CHANNELS, ROWS, COLS>) -> ()
    where
        S: StorageMut<Tensor4DBuffer<BATCHES, CHANNELS, ROWS, COLS>>,
    {
        self.data.as_flat_mut().copy_from_slice(data.as_flat());
    }
    /// Copies from a runtime-sized slice, never materializes a full
    /// `BATCHES * CHANNELS * ROWS * COLS` array on the stack, so it's the
    /// door for a large tensor without `alloc`.
    pub fn load_slice(self: &mut Self, data: &[Scalar]) -> Result<(), LenMismatch>
    where
        S: StorageMut<Tensor4DBuffer<BATCHES, CHANNELS, ROWS, COLS>>,
    {
        if data.len() != BATCHES * CHANNELS * ROWS * COLS {
            return Err(LenMismatch);
        }
        self.data.as_flat_mut().copy_from_slice(data);
        Ok(())
    }
    /// Loads a dynamically-sized buffer, never materializing
    /// `[Scalar; BATCHES * CHANNELS * ROWS * COLS]` on the stack: the entry
    /// point for large tensors.
    ///
    /// Hands the `Vec` back intact if its length doesn't match, rather than
    /// dumping it into a panic message.
    #[cfg(feature = "alloc")]
    pub fn load_vec(
        self: &mut Self,
        data: alloc::vec::Vec<Scalar>,
    ) -> Result<(), alloc::vec::Vec<Scalar>>
    where
        S: OwnedStorage<Tensor4DBuffer<BATCHES, CHANNELS, ROWS, COLS>>,
    {
        if data.len() != BATCHES * CHANNELS * ROWS * COLS {
            return Err(data);
        }
        self.data.as_flat_mut().copy_from_slice(&data);
        Ok(())
    }
    /// Builds a tensor straight from a `Vec` in one step: the `.npy`
    /// pipeline's door, without the caller needing a `mut` local for the
    /// zeroed()+load_vec() two-step.
    #[cfg(feature = "alloc")]
    pub fn from_vec(data: alloc::vec::Vec<Scalar>) -> Result<Self, alloc::vec::Vec<Scalar>>
    where
        S: OwnedStorage<Tensor4DBuffer<BATCHES, CHANNELS, ROWS, COLS>>,
    {
        let mut t = Self::zeroed();
        t.load_vec(data)?;
        Ok(t)
    }
}
