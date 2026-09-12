use super::{Tensor6D, Tensor6DBuffer};
use crate::linalg::storage::{Buffer, LenMismatch, OwnedStorage, Storage, StorageMut};
use crate::scalar::Scalar;

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
    /// Zero-initialized accumulator for the crate's own algorithms. Kept out
    /// of the public API on purpose: an external caller should never get a
    /// silently-zeroed tensor by omitting a load step.
    pub(crate) fn zeroed() -> Self
    where
        S: OwnedStorage<Tensor6DBuffer<BATCHES, GROUPS, CHANNELS, DEPTH, ROWS, COLS>>,
    {
        Self { data: S::zeroed() }
    }
    /// Builds a tensor from data known upfront, the caller never needs `mut`
    /// or a separate load step.
    ///
    /// Takes the buffer's own nested shape ([`Tensor6DBuffer`]) rather than a
    /// flat array: the flat form can't be spelled as an array length here
    /// without the unstable `generic_const_exprs`, and the nested one is
    /// exactly [`Buffer`]'s shape, so the length is verified at compile
    /// time, not checked at runtime. In practice this type is always built
    /// by the crate itself (im2col patches, `tensordot_3` operands), so the
    /// six levels of nesting are never handwritten at a call site.
    pub fn new(data: Tensor6DBuffer<BATCHES, GROUPS, CHANNELS, DEPTH, ROWS, COLS>) -> Self
    where
        S: OwnedStorage<Tensor6DBuffer<BATCHES, GROUPS, CHANNELS, DEPTH, ROWS, COLS>>,
    {
        let mut t = Self::zeroed();
        t.load_data(data);
        t
    }
    /// Loads a full buffer's worth of data, for small tensors and known
    /// data. For a tensor too big to build the full nested array on the
    /// stack, see [`Self::load_slice`] (`no_std`, no `alloc`) or
    /// [`Self::from_vec`].
    pub fn load_data(
        self: &mut Self,
        data: Tensor6DBuffer<BATCHES, GROUPS, CHANNELS, DEPTH, ROWS, COLS>,
    ) -> ()
    where
        S: StorageMut<Tensor6DBuffer<BATCHES, GROUPS, CHANNELS, DEPTH, ROWS, COLS>>,
    {
        self.data.as_flat_mut().copy_from_slice(data.as_flat());
    }
    /// Copies from a runtime-sized slice, never materializes the full array
    /// on the stack.
    pub fn load_slice(self: &mut Self, data: &[Scalar]) -> Result<(), LenMismatch>
    where
        S: StorageMut<Tensor6DBuffer<BATCHES, GROUPS, CHANNELS, DEPTH, ROWS, COLS>>,
    {
        if data.len() != BATCHES * GROUPS * CHANNELS * DEPTH * ROWS * COLS {
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
        S: OwnedStorage<Tensor6DBuffer<BATCHES, GROUPS, CHANNELS, DEPTH, ROWS, COLS>>,
    {
        if data.len() != BATCHES * GROUPS * CHANNELS * DEPTH * ROWS * COLS {
            return Err(data);
        }
        let mut t = Self::zeroed();
        t.data.as_flat_mut().copy_from_slice(&data);
        Ok(t)
    }
}
