use core::{
    array::{from_fn, from_mut, from_ref},
    ops::{Deref, DerefMut},
};

use crate::Scalar;

#[derive(Debug, Clone, PartialEq, Copy, Eq)]
pub struct LenMismatch;
pub unsafe trait Buffer {
    const LEN: usize;
    fn as_flat(&self) -> &[Scalar];
    fn as_flat_mut(&mut self) -> &mut [Scalar];
    fn zeroed_inline() -> Self;
}
unsafe impl Buffer for Scalar {
    const LEN: usize = 1;
    fn as_flat(&self) -> &[Scalar] {
        from_ref(self)
    }
    fn as_flat_mut(&mut self) -> &mut [Scalar] {
        from_mut(self)
    }
    fn zeroed_inline() -> Self {
        0.0
    }
}

unsafe impl<B: Buffer, const N: usize> Buffer for [B; N] {
    const LEN: usize = N * B::LEN;
    fn as_flat(&self) -> &[Scalar] {
        unsafe { core::slice::from_raw_parts(self.as_ptr().cast::<Scalar>(), Self::LEN) }
    }
    fn as_flat_mut(&mut self) -> &mut [Scalar] {
        unsafe { core::slice::from_raw_parts_mut(self.as_mut_ptr().cast::<Scalar>(), Self::LEN) }
    }
    fn zeroed_inline() -> Self {
        from_fn(|_| B::zeroed_inline())
    }
}

pub trait Storage<B: Buffer>: Deref<Target = B> {}
pub trait StorageMut<B: Buffer>: Storage<B> + DerefMut {}
#[derive(Clone, Copy, Debug)]
pub struct StackStorage<B: Buffer> {
    buffer: B,
}
impl<B: Buffer> Deref for StackStorage<B> {
    type Target = B;
    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}
impl<B: Buffer> DerefMut for StackStorage<B> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.buffer
    }
}
impl<B: Buffer> Storage<B> for StackStorage<B> {}
impl<B: Buffer> StorageMut<B> for StackStorage<B> {}

pub trait OwnedStorage<B: Buffer>: StorageMut<B> {
    fn zeroed() -> Self;
}
impl<B: Buffer> OwnedStorage<B> for StackStorage<B> {
    fn zeroed() -> Self {
        StackStorage {
            buffer: B::zeroed_inline(),
        }
    }
}

#[cfg(feature = "alloc")]
pub use heap::HeapStorage;
#[cfg(feature = "alloc")]
pub mod heap {
    use alloc::alloc::{alloc_zeroed, handle_alloc_error};
    use core::alloc::Layout;

    use super::*;
    use alloc::boxed::Box;
    pub struct HeapStorage<B: Buffer> {
        buffer_ref: Box<B>,
    }
    impl<B: Buffer> Deref for HeapStorage<B> {
        type Target = B;
        fn deref(&self) -> &Self::Target {
            &self.buffer_ref
        }
    }
    impl<B: Buffer> DerefMut for HeapStorage<B> {
        fn deref_mut(&mut self) -> &mut Self::Target {
            &mut self.buffer_ref
        }
    }

    impl<B: Buffer> Storage<B> for HeapStorage<B> {}
    impl<B: Buffer> StorageMut<B> for HeapStorage<B> {}

    impl<B: Buffer> OwnedStorage<B> for HeapStorage<B> {
        fn zeroed() -> Self {
            let layout = Layout::new::<B>();
            if layout.size() == 0 {
                HeapStorage {
                    buffer_ref: unsafe { Box::from_raw(core::ptr::NonNull::dangling().as_ptr()) },
                }
            } else {
                let ptr = unsafe { alloc_zeroed(layout) };
                if ptr.is_null() {
                    handle_alloc_error(layout)
                } else {
                    HeapStorage {
                        buffer_ref: unsafe { Box::from_raw(ptr.cast::<B>()) },
                    }
                }
            }
        }
    }
}
