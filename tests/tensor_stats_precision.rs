//! Precision checks for the tensor statistics (`mean`/`variance`/`std`/
//! `min`/`max`), across every tensor rank.
//!
//! `variance` used to be computed as `E[X^2] - E[X]^2`: correct in exact
//! arithmetic, but the two terms grow with the data's offset from zero while
//! their difference (the actual variance) doesn't, so floating-point
//! cancellation swallows the signal once the offset is large enough. These
//! tests pin the fix: `variance` is now `E[(X - E[X])^2]`, computed on
//! deviations from the mean rather than on the raw values, which stays
//! accurate regardless of how far the data sits from zero.

use frugal_ml::linalg::tensor::{Tensor, Tensor3D, Tensor4D, Tensor6D};
use frugal_ml::scalar::Scalar;

fn assert_close(actual: Scalar, expected: Scalar, epsilon: Scalar) {
    assert!(
        (actual - expected).abs() < epsilon,
        "expected {expected}, got {actual} (epsilon {epsilon})"
    );
}

/// Large enough that `offset + offset` cancellation in the old `E[X^2] -
/// E[X]^2` formula loses essentially all precision, on `f32` and `f64`
/// alike: at this magnitude `offset`'s own ulp is 1.0, i.e. right at the
/// scale of the data's spread.
fn unstable_offset() -> Scalar {
    1.0 / Scalar::EPSILON
}

// ---- Tensor (2D) ----------------------------------------------------------

#[test]
fn variance_matches_hand_computed_value() {
    // mean = 2.5; deviations are -1.5, -0.5, 0.5, 1.5 -> mean of squares = 1.25.
    let m = Tensor::<2, 2>::new([[1.0, 2.0], [3.0, 4.0]]);
    assert_close(m.mean(), 2.5, 1e-5);
    assert_close(m.variance(), 1.25, 1e-5);
    assert_close(m.std(), 1.118_033_988_749_895, 1e-5);
}

#[test]
fn variance_is_shift_invariant_far_from_zero() {
    let offset = unstable_offset();
    let m = Tensor::<2, 2>::new([[offset, offset + 1.0], [offset + 2.0, offset + 3.0]]);
    // True variance of [0, 1, 2, 3] is 1.25 and does not depend on the
    // additive offset; only the offset's own representable granularity at
    // this magnitude limits how exactly that value can be recovered.
    assert_close(m.variance(), 1.25, 5.0);
}

#[test]
fn stable_variance_survives_where_the_naive_formula_does_not() {
    let offset = unstable_offset();
    let data = [offset, offset + 1.0, offset + 2.0, offset + 3.0];
    let m = Tensor::<2, 2>::new([[data[0], data[1]], [data[2], data[3]]]);

    let stable = m.variance();

    // The formula this replaced: E[X^2] - E[X]^2.
    let mean = m.mean();
    let mean_of_squares: Scalar = data.iter().map(|x| x * x).sum::<Scalar>() / data.len() as Scalar;
    let naive = mean_of_squares - mean * mean;

    assert_close(stable, 1.25, 5.0);
    // The naive formula's cancellation error dwarfs the true variance
    // (1.25) by many orders of magnitude at this offset.
    assert!(
        (naive - 1.25).abs() > 1000.0,
        "naive formula unexpectedly precise: {naive}"
    );
}

#[test]
fn rows_and_cols_variance_are_shift_invariant() {
    let offset = unstable_offset();
    let m = Tensor::<2, 2>::new([[offset, offset + 10.0], [offset + 2.0, offset + 3.0]]);

    // Row 0: [offset, offset+10] -> variance 25.0. Row 1: [offset+2, offset+3] -> variance 0.25.
    let rows_var = m.rows_variance();
    assert_close(rows_var.get(0, 0), 1.0, 5.0); // col 0: [offset, offset+2] -> variance 1.0
    assert_close(rows_var.get(0, 1), 12.25, 5.0); // col 1: [offset+10, offset+3] -> variance 12.25

    let cols_var = m.cols_variance();
    assert_close(cols_var.get(0, 0), 25.0, 5.0);
    assert_close(cols_var.get(1, 0), 0.25, 5.0);
}

#[test]
fn min_and_max() {
    let m = Tensor::<2, 3>::new([[3.0, -1.0, 4.0], [1.0, 5.0, -9.0]]);
    assert_eq!(m.min(), -9.0);
    assert_eq!(m.max(), 5.0);
}

#[test]
fn standardize_zeroes_mean_and_unit_std() {
    let mut m = Tensor::<2, 2>::new([[1.0, 2.0], [3.0, 4.0]]);
    m.standardize();
    assert_close(m.mean(), 0.0, 1e-4);
    assert_close(m.std(), 1.0, 1e-4);
}

#[test]
fn standardize_with_applies_external_stats_verbatim() {
    // Deliberately not this tensor's own mean/std: standardize_with must use
    // exactly what it's given, not recompute anything.
    let mut m = Tensor::<2, 2>::new([[1.0, 2.0], [3.0, 4.0]]);
    m.standardize_with(10.0, 2.0);
    assert_close(m.get(0, 0), (1.0 - 10.0) / 2.0, 1e-5);
    assert_close(m.get(0, 1), (2.0 - 10.0) / 2.0, 1e-5);
    assert_close(m.get(1, 0), (3.0 - 10.0) / 2.0, 1e-5);
    assert_close(m.get(1, 1), (4.0 - 10.0) / 2.0, 1e-5);
}

#[test]
fn min_max_scale_maps_into_unit_range() {
    let mut m = Tensor::<2, 3>::new([[3.0, -1.0, 4.0], [1.0, 5.0, -9.0]]);
    m.min_max_scale();
    assert_close(m.min(), 0.0, 1e-5);
    assert_close(m.max(), 1.0, 1e-5);
}

#[test]
fn min_max_scale_with_applies_external_bounds_verbatim() {
    let mut m = Tensor::<2, 2>::new([[1.0, 2.0], [3.0, 4.0]]);
    m.min_max_scale_with(0.0, 10.0);
    assert_close(m.get(0, 0), 0.1, 1e-5);
    assert_close(m.get(0, 1), 0.2, 1e-5);
    assert_close(m.get(1, 0), 0.3, 1e-5);
    assert_close(m.get(1, 1), 0.4, 1e-5);
}

#[test]
fn standardize_with_zero_std_does_not_divide_by_zero() {
    // A constant tensor has std == 0: without STATS_EPSILON this would
    // divide by zero and produce NaN/inf.
    let mut m = Tensor::<2, 2>::new([[7.0, 7.0], [7.0, 7.0]]);
    m.standardize_with(7.0, 0.0);
    for &x in m.get_raw_buffer() {
        assert_eq!(x, 0.0);
        assert!(x.is_finite());
    }
}

#[test]
fn min_max_scale_with_zero_range_does_not_divide_by_zero() {
    // min == max (a constant tensor) makes the range 0: without
    // STATS_EPSILON this would divide by zero and produce NaN/inf.
    let mut m = Tensor::<2, 2>::new([[7.0, 7.0], [7.0, 7.0]]);
    m.min_max_scale_with(7.0, 7.0);
    for &x in m.get_raw_buffer() {
        assert_eq!(x, 0.0);
        assert!(x.is_finite());
    }
}

#[test]
fn scaling_new_tensor_variants_leave_the_original_untouched() {
    let original = Tensor::<2, 2>::new([[1.0, 2.0], [3.0, 4.0]]);

    let standardized = original.standardized();
    assert_eq!(original, Tensor::<2, 2>::new([[1.0, 2.0], [3.0, 4.0]]));
    assert_close(standardized.mean(), 0.0, 1e-4);
    assert_close(standardized.std(), 1.0, 1e-4);

    let standardized_with = original.standardized_with(10.0, 2.0);
    assert_eq!(original, Tensor::<2, 2>::new([[1.0, 2.0], [3.0, 4.0]]));
    assert_close(standardized_with.get(0, 0), (1.0 - 10.0) / 2.0, 1e-5);

    let min_max_scaled = original.min_max_scaled();
    assert_eq!(original, Tensor::<2, 2>::new([[1.0, 2.0], [3.0, 4.0]]));
    assert_close(min_max_scaled.min(), 0.0, 1e-5);
    assert_close(min_max_scaled.max(), 1.0, 1e-5);

    let min_max_scaled_with = original.min_max_scaled_with(0.0, 10.0);
    assert_eq!(original, Tensor::<2, 2>::new([[1.0, 2.0], [3.0, 4.0]]));
    assert_close(min_max_scaled_with.get(0, 0), 0.1, 1e-5);
}

#[test]
fn scaling_new_tensor_variants_match_the_in_place_ones() {
    let original = Tensor::<2, 2>::new([[1.0, 2.0], [3.0, 4.0]]);

    let mut in_place = original;
    in_place.standardize();
    assert_eq!(in_place, original.standardized());

    let mut in_place = original;
    in_place.min_max_scale();
    assert_eq!(in_place, original.min_max_scaled());
}

// ---- Tensor3D / Tensor4D / Tensor6D ---------------------------------------
// Same stable formula, generalized to every other rank.

#[test]
fn tensor3d_stats_are_generalized() {
    let m = Tensor3D::<1, 2, 2>::new([[[1.0, 2.0], [3.0, 4.0]]]);
    assert_close(m.mean(), 2.5, 1e-5);
    assert_close(m.variance(), 1.25, 1e-5);
    assert_close(m.std(), 1.118_033_988_749_895, 1e-5);
    assert_eq!(m.min(), 1.0);
    assert_eq!(m.max(), 4.0);
}

#[test]
fn tensor3d_standardize_and_min_max_scale_are_generalized() {
    let mut m = Tensor3D::<1, 2, 2>::new([[[1.0, 2.0], [3.0, 4.0]]]);
    m.standardize();
    assert_close(m.mean(), 0.0, 1e-4);
    assert_close(m.std(), 1.0, 1e-4);

    let mut m = Tensor3D::<1, 2, 2>::new([[[1.0, 2.0], [3.0, 4.0]]]);
    m.min_max_scale();
    assert_close(m.min(), 0.0, 1e-5);
    assert_close(m.max(), 1.0, 1e-5);

    let original = Tensor3D::<1, 2, 2>::new([[[1.0, 2.0], [3.0, 4.0]]]);
    assert_close(original.standardized().mean(), 0.0, 1e-4);
    assert_close(original.min_max_scaled().min(), 0.0, 1e-5);
    assert_close(original.mean(), 2.5, 1e-5); // the original is untouched
}

#[test]
fn tensor3d_variance_stays_accurate_far_from_zero() {
    let offset = unstable_offset();
    let m = Tensor3D::<1, 2, 2>::from_vec(vec![offset, offset + 1.0, offset + 2.0, offset + 3.0])
        .unwrap();
    assert_close(m.variance(), 1.25, 5.0);
}

#[test]
fn tensor4d_stats_are_generalized() {
    let m = Tensor4D::<1, 1, 2, 2>::new([[[[1.0, 2.0], [3.0, 4.0]]]]);
    assert_close(m.mean(), 2.5, 1e-5);
    assert_close(m.variance(), 1.25, 1e-5);
    assert_eq!(m.min(), 1.0);
    assert_eq!(m.max(), 4.0);
}

#[test]
fn tensor4d_variance_stays_accurate_far_from_zero() {
    let offset = unstable_offset();
    let m =
        Tensor4D::<1, 1, 2, 2>::from_vec(vec![offset, offset + 1.0, offset + 2.0, offset + 3.0])
            .unwrap();
    assert_close(m.variance(), 1.25, 5.0);
}

#[test]
fn tensor6d_stats_are_generalized() {
    let m = Tensor6D::<1, 1, 1, 1, 2, 2>::from_vec(vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    assert_close(m.mean(), 2.5, 1e-5);
    assert_close(m.variance(), 1.25, 1e-5);
    assert_eq!(m.min(), 1.0);
    assert_eq!(m.max(), 4.0);
}

#[test]
fn tensor6d_variance_stays_accurate_far_from_zero() {
    let offset = unstable_offset();
    let m = Tensor6D::<1, 1, 1, 1, 2, 2>::from_vec(vec![
        offset,
        offset + 1.0,
        offset + 2.0,
        offset + 3.0,
    ])
    .unwrap();
    assert_close(m.variance(), 1.25, 5.0);
}
