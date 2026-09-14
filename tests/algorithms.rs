use frugal_ml::linalg::decomposition::{
    gram_schmidt, inverse, jacobi_rotation, qr_decomposition, solve_linear_system,
    solve_upper_triangular, svd, svd_2x2,
};
use frugal_ml::linalg::tensor::{Tensor, Vector};
use frugal_ml::Scalar;

#[test]
fn test_gram_schmidt_2d() {
    let v1 = Vector::from_data([1.0, 1.0]);
    let v2 = Vector::from_data([0.0, 1.0]);
    let ortho = gram_schmidt::<2, 2>(&[v1, v2]);
    assert!(ortho[0].dot(&ortho[1]).abs() < 1e-6);
    assert!((ortho[0].l2_norm() - 1.0).abs() < 1e-6);
    assert!((ortho[1].l2_norm() - 1.0).abs() < 1e-6);
}

#[test]
fn test_qr_decomposition_simple() {
    let mut a = Tensor::<2, 2>::new([[0.0; 2]; 2]);
    a[(0, 0)] = 1.0;
    a[(0, 1)] = 1.0;
    a[(1, 0)] = 0.0;
    a[(1, 1)] = 1.0;
    let (q, r) = qr_decomposition::<2, 2>(&a);
    let qr: Tensor<2, 2> = q.multiply(&r);
    assert_eq!(qr, a);
    let qtq: Tensor<2, 2> = q.transposed().multiply(&q);
    assert_eq!(qtq, Tensor::<2, 2>::identity());
}

#[test]
fn test_solve_upper_triangular() {
    let mut r = Tensor::<2, 2>::new([[0.0; 2]; 2]);
    r[(0, 0)] = 2.0;
    r[(0, 1)] = 1.0;
    r[(1, 1)] = 1.0;
    let b = Vector::from_data([5.0, 1.0]);
    let x = solve_upper_triangular(&r, &b).unwrap();
    assert_eq!(x, Vector::from_data([2.0, 1.0]));
}

#[test]
fn test_solve_linear_system_2d() {
    let mut a = Tensor::<2, 2>::new([[0.0; 2]; 2]);
    a[(0, 0)] = 1.0;
    a[(0, 1)] = 1.0;
    a[(1, 0)] = 1.0;
    a[(1, 1)] = -1.0;
    let b = Vector::from_data([3.0, 1.0]);
    let x = solve_linear_system::<2, 2>(&a, &b).unwrap();
    assert_eq!(x, Vector::from_data([2.0, 1.0]));
}

#[test]
fn test_singular_system() {
    let a = Tensor::<2, 2>::new([[0.0; 2]; 2]);
    let b = Vector::from_data([1.0, 1.0]);
    assert!(solve_linear_system::<2, 2>(&a, &b).is_none());
}

#[test]
fn test_gram_schmidt_dependent() {
    let v1 = Vector::from_data([1.0, 0.0]);
    let v2 = Vector::from_data([2.0, 0.0]);
    let ortho = gram_schmidt::<2, 2>(&[v1, v2]);
    assert_eq!(ortho[1], Vector::from_data([0.0, 0.0]));
}

#[test]
fn test_qr_3x2_matrix() {
    let mut a = Tensor::<3, 2>::new([[0.0; 2]; 3]);
    a[(0, 0)] = 12.0;
    a[(0, 1)] = -51.0;
    a[(1, 0)] = 6.0;
    a[(1, 1)] = 167.0;
    a[(2, 0)] = -4.0;
    a[(2, 1)] = 24.0;
    let (q, r) = qr_decomposition::<3, 2>(&a);
    let qr: Tensor<3, 2> = q.multiply(&r);
    assert_eq!(qr, a);
}

#[test]
fn test_back_substitution_3d() {
    let mut r = Tensor::<3, 3>::new([[0.0; 3]; 3]);
    r[(0, 0)] = 1.0;
    r[(0, 1)] = 2.0;
    r[(0, 2)] = 3.0;
    r[(1, 1)] = 1.0;
    r[(1, 2)] = 2.0;
    r[(2, 2)] = 1.0;
    let b = Vector::from_data([6.0, 3.0, 1.0]);
    let x = solve_upper_triangular(&r, &b).unwrap();
    assert_eq!(x, Vector::from_data([1.0, 1.0, 1.0]));
}

#[test]
fn test_identity_solver() {
    let a = Tensor::<3, 3>::identity();
    let b = Vector::from_data([1.0, 2.0, 3.0]);
    let x = solve_linear_system::<3, 3>(&a, &b).unwrap();
    assert_eq!(x, b);
}

#[test]
fn test_orthogonal_projection_consistency() {
    let v = Vector::from_data([1.0, 2.0, 3.0]);
    let u = Vector::from_data([1.0, 0.0, 0.0]);
    let proj = v.orthogonal_projection(&u);
    assert_eq!(proj, Vector::from_data([1.0, 0.0, 0.0]));
}

#[test]
fn test_svd_2x2() {
    let mut a = Tensor::<2, 2>::new([[0.0; 2]; 2]);
    a[(0, 0)] = 2.0;
    a[(0, 1)] = 1.0;
    a[(1, 0)] = 1.0;
    a[(1, 1)] = 2.0;
    let (u, _, _) = svd_2x2(&a);
    assert!(u.get_col(0).unwrap().dot(&u.get_col(1).unwrap()).abs() < 1e-5);
}

#[test]
fn test_svd_reconstruction_3x3() {
    let mut a = Tensor::<3, 3>::new([[0.0; 3]; 3]);
    a[(0, 0)] = 4.0;
    a[(0, 1)] = 2.0;
    a[(0, 2)] = 1.0;
    a[(1, 0)] = 2.0;
    a[(1, 1)] = 3.0;
    a[(1, 2)] = 1.0;
    a[(2, 0)] = 1.0;
    a[(2, 1)] = 1.0;
    a[(2, 2)] = 2.0;
    let (u, sigma, v): (Tensor<3, 3>, Vector<3>, Tensor<3, 3>) = svd(&a);
    let mut sigma_mat = Tensor::<3, 3>::new([[0.0; 3]; 3]);
    sigma_mat[(0, 0)] = sigma[0];
    sigma_mat[(1, 1)] = sigma[1];
    sigma_mat[(2, 2)] = sigma[2];
    let u_sigma: Tensor<3, 3> = u.multiply(&sigma_mat);
    let reconstructed: Tensor<3, 3> = u_sigma.multiply(&v.transposed());
    assert_eq!(reconstructed, a);
    let utu: Tensor<3, 3> = u.transposed().multiply(&u);
    assert_eq!(utu, Tensor::<3, 3>::identity());
    let vtv: Tensor<3, 3> = v.transposed().multiply(&v);
    assert_eq!(vtv, Tensor::<3, 3>::identity());
}

#[test]
fn test_svd_identity_3x3() {
    let a = Tensor::<3, 3>::identity();
    let (_, sigma, _): (Tensor<3, 3>, Vector<3>, Tensor<3, 3>) = svd(&a);
    for i in 0..3 {
        assert!((sigma[i] - 1.0).abs() < 1e-5, "sigma[{i}] = {}", sigma[i]);
    }
}

#[test]
fn test_svd_reconstruction_4x4() {
    let mut a = Tensor::<4, 4>::new([[0.0; 4]; 4]);
    a[(0, 0)] = 5.0;
    a[(0, 1)] = 1.0;
    a[(0, 2)] = 2.0;
    a[(0, 3)] = 0.0;
    a[(1, 0)] = 1.0;
    a[(1, 1)] = 4.0;
    a[(1, 2)] = 1.0;
    a[(1, 3)] = 1.0;
    a[(2, 0)] = 2.0;
    a[(2, 1)] = 1.0;
    a[(2, 2)] = 3.0;
    a[(2, 3)] = 0.0;
    a[(3, 0)] = 0.0;
    a[(3, 1)] = 1.0;
    a[(3, 2)] = 0.0;
    a[(3, 3)] = 2.0;
    let (u, sigma, v): (Tensor<4, 4>, Vector<4>, Tensor<4, 4>) = svd(&a);
    let mut sigma_mat = Tensor::<4, 4>::new([[0.0; 4]; 4]);
    for i in 0..4 {
        sigma_mat[(i, i)] = sigma[i];
    }
    let u_sigma: Tensor<4, 4> = u.multiply(&sigma_mat);
    let reconstructed: Tensor<4, 4> = u_sigma.multiply(&v.transposed());
    assert_eq!(reconstructed, a);
    let utu: Tensor<4, 4> = u.transposed().multiply(&u);
    assert_eq!(utu, Tensor::<4, 4>::identity());
    let vtv: Tensor<4, 4> = v.transposed().multiply(&v);
    assert_eq!(vtv, Tensor::<4, 4>::identity());
}

#[test]
fn bench_matmul() {
    let mut a2 = Tensor::<2, 2>::new([[0.0; 2]; 2]);
    a2[(0, 0)] = 1.0;
    a2[(0, 1)] = 2.0;
    a2[(1, 0)] = 3.0;
    a2[(1, 1)] = 4.0;
    let mut a3 = Tensor::<3, 3>::new([[0.0; 3]; 3]);
    for i in 0..3 {
        for j in 0..3 {
            a3[(i, j)] = (i * 3 + j + 1) as Scalar;
        }
    }
    let mut a4 = Tensor::<4, 4>::new([[0.0; 4]; 4]);
    for i in 0..4 {
        for j in 0..4 {
            a4[(i, j)] = (i * 4 + j + 1) as Scalar;
        }
    }
    let n = 100_000u32;
    let t = std::time::Instant::now();
    for _ in 0..n {
        let _: Tensor<2, 2> = a2.multiply(&a2);
    }
    println!(
        "matmul 2x2 x{n}: {:?} ({:.1}ns/iter)",
        t.elapsed(),
        t.elapsed().as_nanos() as f64 / n as f64
    );
    let t = std::time::Instant::now();
    for _ in 0..n {
        let _: Tensor<3, 3> = a3.multiply(&a3);
    }
    println!(
        "matmul 3x3 x{n}: {:?} ({:.1}ns/iter)",
        t.elapsed(),
        t.elapsed().as_nanos() as f64 / n as f64
    );
    let t = std::time::Instant::now();
    for _ in 0..n {
        let _: Tensor<4, 4> = a4.multiply(&a4);
    }
    println!(
        "matmul 4x4 x{n}: {:?} ({:.1}ns/iter)",
        t.elapsed(),
        t.elapsed().as_nanos() as f64 / n as f64
    );
}

#[test]
fn bench_svd() {
    let mut a2 = Tensor::<2, 2>::new([[0.0; 2]; 2]);
    a2[(0, 0)] = 2.0;
    a2[(0, 1)] = 1.0;
    a2[(1, 0)] = 1.0;
    a2[(1, 1)] = 2.0;
    let mut a3 = Tensor::<3, 3>::new([[0.0; 3]; 3]);
    a3[(0, 0)] = 4.0;
    a3[(0, 1)] = 2.0;
    a3[(0, 2)] = 1.0;
    a3[(1, 0)] = 2.0;
    a3[(1, 1)] = 3.0;
    a3[(1, 2)] = 1.0;
    a3[(2, 0)] = 1.0;
    a3[(2, 1)] = 1.0;
    a3[(2, 2)] = 2.0;
    let mut a4 = Tensor::<4, 4>::new([[0.0; 4]; 4]);
    a4[(0, 0)] = 5.0;
    a4[(0, 1)] = 1.0;
    a4[(0, 2)] = 2.0;
    a4[(0, 3)] = 0.0;
    a4[(1, 0)] = 1.0;
    a4[(1, 1)] = 4.0;
    a4[(1, 2)] = 1.0;
    a4[(1, 3)] = 1.0;
    a4[(2, 0)] = 2.0;
    a4[(2, 1)] = 1.0;
    a4[(2, 2)] = 3.0;
    a4[(2, 3)] = 0.0;
    a4[(3, 0)] = 0.0;
    a4[(3, 1)] = 1.0;
    a4[(3, 2)] = 0.0;
    a4[(3, 3)] = 2.0;
    let n = 10_000u32;
    let t = std::time::Instant::now();
    for _ in 0..n {
        let _ = svd_2x2(&a2);
    }
    println!(
        "svd 2x2 x{n}: {:?} ({:.1}ns/iter)",
        t.elapsed(),
        t.elapsed().as_nanos() as f64 / n as f64
    );
    let t = std::time::Instant::now();
    for _ in 0..n {
        let _: (Tensor<3, 3>, Vector<3>, Tensor<3, 3>) = svd(&a3);
    }
    println!(
        "svd 3x3 x{n}: {:?} ({:.1}ns/iter)",
        t.elapsed(),
        t.elapsed().as_nanos() as f64 / n as f64
    );
    let t = std::time::Instant::now();
    for _ in 0..n {
        let _: (Tensor<4, 4>, Vector<4>, Tensor<4, 4>) = svd(&a4);
    }
    println!(
        "svd 4x4 x{n}: {:?} ({:.1}ns/iter)",
        t.elapsed(),
        t.elapsed().as_nanos() as f64 / n as f64
    );
}

#[test]
fn test_svd_recomposition() {
    let mut a = Tensor::<2, 2>::new([[0.0; 2]; 2]);
    a[(0, 0)] = 3.0;
    a[(0, 1)] = 1.0;
    a[(1, 0)] = 1.0;
    a[(1, 1)] = 3.0;
    let (u, sigma, v) = svd_2x2(&a);
    let mut s = Tensor::<2, 2>::new([[0.0; 2]; 2]);
    s[(0, 0)] = sigma[0];
    s[(1, 1)] = sigma[1];
    let u_s: Tensor<2, 2> = u.multiply(&s);
    let reconstructed: Tensor<2, 2> = u_s.multiply(&v.transposed());
    assert_eq!(reconstructed, a);
}

#[test]
fn test_jacobi_rotation_zero() {
    let (cos, sin) = jacobi_rotation(1.0, 1.0, 0.0);
    assert_eq!(cos, 1.0);
    assert_eq!(sin, 0.0);
}

#[test]
fn test_inverse_identity() {
    let a = Tensor::<3, 3>::identity();
    let inv = inverse(&a).unwrap();
    assert_eq!(inv, a);
}

#[test]
fn test_inverse_2x2() {
    let mut a = Tensor::<2, 2>::new([[0.0; 2]; 2]);
    a[(0, 0)] = 4.0;
    a[(0, 1)] = 7.0;
    a[(1, 0)] = 2.0;
    a[(1, 1)] = 6.0;
    let inv = inverse(&a).unwrap();

    // Known closed-form inverse: 1/det * [[d, -b], [-c, a]], det = 10.
    let mut expected = Tensor::<2, 2>::new([[0.0; 2]; 2]);
    expected[(0, 0)] = 0.6;
    expected[(0, 1)] = -0.7;
    expected[(1, 0)] = -0.2;
    expected[(1, 1)] = 0.4;
    assert_eq!(inv, expected);

    let identity: Tensor<2, 2> = a.multiply(&inv);
    assert_eq!(identity, Tensor::<2, 2>::identity());
    let identity_rev: Tensor<2, 2> = inv.multiply(&a);
    assert_eq!(identity_rev, Tensor::<2, 2>::identity());
}

#[test]
fn test_inverse_3x3() {
    let mut a = Tensor::<3, 3>::new([[0.0; 3]; 3]);
    a[(0, 0)] = 2.0;
    a[(0, 1)] = 0.0;
    a[(0, 2)] = 1.0;
    a[(1, 0)] = 1.0;
    a[(1, 1)] = 3.0;
    a[(1, 2)] = 2.0;
    a[(2, 0)] = 1.0;
    a[(2, 1)] = 0.0;
    a[(2, 2)] = 1.0;
    let inv = inverse(&a).unwrap();

    let identity: Tensor<3, 3> = a.multiply(&inv);
    assert_eq!(identity, Tensor::<3, 3>::identity());
    let identity_rev: Tensor<3, 3> = inv.multiply(&a);
    assert_eq!(identity_rev, Tensor::<3, 3>::identity());
}

#[test]
fn test_inverse_singular_returns_none() {
    // Rank-deficient: row 2 is a multiple of row 1.
    let mut a = Tensor::<3, 3>::new([[0.0; 3]; 3]);
    a[(0, 0)] = 1.0;
    a[(0, 1)] = 2.0;
    a[(0, 2)] = 3.0;
    a[(1, 0)] = 2.0;
    a[(1, 1)] = 4.0;
    a[(1, 2)] = 6.0;
    a[(2, 0)] = 1.0;
    a[(2, 1)] = 0.0;
    a[(2, 2)] = 1.0;
    assert!(inverse(&a).is_none());
}

#[test]
fn test_inverse_zero_matrix_returns_none() {
    let a = Tensor::<3, 3>::new([[0.0; 3]; 3]);
    assert!(inverse(&a).is_none());
}

#[test]
fn test_inverse_diagonal_matrix() {
    let mut a = Tensor::<3, 3>::new([[0.0; 3]; 3]);
    a[(0, 0)] = 2.0;
    a[(1, 1)] = 4.0;
    a[(2, 2)] = 0.5;
    let inv = inverse(&a).unwrap();

    let mut expected = Tensor::<3, 3>::new([[0.0; 3]; 3]);
    expected[(0, 0)] = 0.5;
    expected[(1, 1)] = 0.25;
    expected[(2, 2)] = 2.0;
    assert_eq!(inv, expected);
}

/// Numerical stability under increasing ill-conditioning: as the matrix's
/// condition number grows, `inverse` should either degrade gracefully
/// (A * A^-1 stays close to I within the loosened tolerance below) or
/// report failure via `None` — it must never panic or silently overflow.
#[test]
fn test_inverse_stability_ill_conditioned() {
    for &ratio in &[1e2, 1e3, 1e4, 1e5] {
        let mut a = Tensor::<3, 3>::new([[0.0; 3]; 3]);
        a[(0, 0)] = 1.0;
        a[(1, 1)] = 1.0;
        a[(2, 2)] = 1.0 / ratio as Scalar;

        if let Some(inv) = inverse(&a) {
            let identity: Tensor<3, 3> = a.multiply(&inv);
            for i in 0..3 {
                for j in 0..3 {
                    let expected = if i == j { 1.0 } else { 0.0 };
                    let err = (identity[(i, j)] - expected).abs();
                    assert!(
                        err < 1e-2,
                        "ratio {ratio}: |A*A^-1 - I|[{i},{j}] = {err} (got {})",
                        identity[(i, j)]
                    );
                }
            }
        }
        // else: singular-enough to be rejected, which is an acceptable outcome.
    }
}

/// A Hilbert matrix is the classic ill-conditioned example (condition
/// number grows exponentially with N). This documents how far the f32/f64
/// Gram-Schmidt-based QR inverse can be pushed before it stops being
/// trustworthy.
#[test]
fn test_inverse_hilbert_4x4_stability() {
    let mut h = Tensor::<4, 4>::new([[0.0; 4]; 4]);
    for i in 0..4 {
        for j in 0..4 {
            h[(i, j)] = 1.0 / (i + j + 1) as Scalar;
        }
    }
    match inverse(&h) {
        Some(inv) => {
            let identity: Tensor<4, 4> = h.multiply(&inv);
            let mut max_err: Scalar = 0.0;
            for i in 0..4 {
                for j in 0..4 {
                    let expected = if i == j { 1.0 } else { 0.0 };
                    max_err = max_err.max((identity[(i, j)] - expected).abs());
                }
            }
            println!("Hilbert 4x4 inverse: max |A*A^-1 - I| = {max_err}");
        }
        None => {
            println!("Hilbert 4x4 inverse: rejected as singular (acceptable)");
        }
    }
}

/// Regression test for the relative-threshold fix: 1e-8 is bigger than the
/// old absolute `1e-10` cutoff (so it used to be treated as safe to divide
/// by), but it is far below what f32 can resolve relative to a matrix
/// scaled around 1.0 (`Scalar::EPSILON` ~1.19e-7 for f32). The old absolute
/// threshold let this through as if well-conditioned; it must now be
/// rejected as singular.
#[test]
fn test_inverse_relative_threshold_rejects_f32_noise_pivot() {
    let mut a = Tensor::<3, 3>::new([[0.0; 3]; 3]);
    a[(0, 0)] = 1.0;
    a[(1, 1)] = 1.0;
    a[(2, 2)] = 1e-8;
    assert!(inverse(&a).is_none());
}
