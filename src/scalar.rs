#[cfg(feature = "f64")]
pub type Scalar = f64;
#[cfg(not(feature = "f64"))]
pub type Scalar = f32;

/// Floor added to statistics-derived divisors (`std`, `max - min`, ...)
/// before dividing, so a degenerate (constant) tensor yields a large finite
/// value instead of `NaN`/`inf`.
pub const STATS_EPSILON: Scalar = 1e-8;

#[cfg(feature = "f64")]
pub use libm::{exp, fabs, log, pow, sqrt, tanh};
#[cfg(not(feature = "f64"))]
pub use libm::{
    expf as exp, fabsf as fabs, logf as log, powf as pow, sqrtf as sqrt, tanhf as tanh,
};
