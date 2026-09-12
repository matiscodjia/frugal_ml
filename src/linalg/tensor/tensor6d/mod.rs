use crate::linalg::storage::{StackStorage, Storage};
use crate::scalar::Scalar;

mod access;
mod construction;
mod rank6;
mod stats;
mod view;

pub use rank6::Rank6;

/// The nested-array buffer shape backing a `Tensor6D<..>`: named so its
/// submodules (`construction`, `access`, `rank6`) don't have to spell the
/// six levels of nesting out at every bound, and so `Tensor6D::new`'s
/// parameter type (also this shape) has a name to show up in its docs.
pub type Tensor6DBuffer<
    const BATCHES: usize,
    const GROUPS: usize,
    const CHANNELS: usize,
    const DEPTH: usize,
    const ROWS: usize,
    const COLS: usize,
> = [[[[[[Scalar; COLS]; ROWS]; DEPTH]; CHANNELS]; GROUPS]; BATCHES];

/// `BATCHES`/`GROUPS`/`CHANNELS`/`DEPTH`/`ROWS`/`COLS` fix the shape for
/// good: there is no runtime shape/stride state, unlike [`TensorView6D`]
/// (which really does move over an arbitrary source buffer). Strides are
/// folded into every flat index at the call site instead.
pub struct Tensor6D<
    const BATCHES: usize,
    const GROUPS: usize,
    const CHANNELS: usize,
    const DEPTH: usize,
    const ROWS: usize,
    const COLS: usize,
    S: Storage<Tensor6DBuffer<BATCHES, GROUPS, CHANNELS, DEPTH, ROWS, COLS>> = StackStorage<
        Tensor6DBuffer<BATCHES, GROUPS, CHANNELS, DEPTH, ROWS, COLS>,
    >,
> {
    data: S,
}

/// An im2col view over a (N x C x H x W) tensor: one (C x KH x KW) receptive
/// field per output pixel, laid out as (N x H_OUT x W_OUT x C x KH x KW).
///
/// The const parameters go source tensor first (N, C, H, W), then window
/// geometry (H_OUT, W_OUT, KH, KW). No data is copied: the view only remaps
/// strides onto the input buffer, so the same element is aliased by every
/// window that overlaps it.
pub struct TensorView6D<
    'a,
    const N: usize,
    const C: usize,
    const H: usize,
    const W: usize,
    const H_OUT: usize,
    const W_OUT: usize,
    const KH: usize,
    const KW: usize,
> {
    pub(super) data: &'a [Scalar],
    pub(super) reference_index: usize,
    pub(super) n_stride: usize,
    pub(super) h_out_stride: usize,
    pub(super) w_out_stride: usize,
    pub(super) channel_stride: usize,
    pub(super) kh_stride: usize,
    pub(super) kw_stride: usize,
    pub(super) shape: [usize; 6],
}
