//! Stress and strain tensors for continuum mechanics.
//!
//! Provides Cauchy stress tensor and infinitesimal strain tensor representations
//! using 3×3 symmetric matrices from nalgebra.

use nalgebra::{Matrix3, Vector3};
use serde::{Deserialize, Serialize};

/// Cauchy stress tensor — a symmetric 3×3 tensor representing the state of stress at a point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressTensor {
    /// The 3×3 symmetric stress matrix [σ_ij]
    pub matrix: Matrix3<f64>,
}

impl StressTensor {
    /// Create a new stress tensor from a 3×3 matrix.
    /// The matrix is symmetrized by averaging off-diagonal components.
    pub fn new(matrix: Matrix3<f64>) -> Self {
        let sym = (matrix + matrix.transpose()) * 0.5;
        Self { matrix: sym }
    }

    /// Create a stress tensor from individual components.
    /// σ = [[σ_xx, τ_xy, τ_xz], [τ_xy, σ_yy, τ_yz], [τ_xz, τ_yz, σ_zz]]
    pub fn from_components(
        sxx: f64, syy: f64, szz: f64,
        txy: f64, txz: f64, tyz: f64,
    ) -> Self {
        Self {
            matrix: Matrix3::new(
                sxx, txy, txz,
                txy, syy, tyz,
                txz, tyz, szz,
            ),
        }
    }

    /// Hydrostatic (mean) stress: σ_m = (σ_xx + σ_yy + σ_zz) / 3
    pub fn hydrostatic(&self) -> f64 {
        self.matrix.trace() / 3.0
    }

    /// Deviatoric stress tensor: s_ij = σ_ij - σ_m * δ_ij
    pub fn deviatoric(&self) -> StressTensor {
        let p = self.hydrostatic();
        let dev = self.matrix - p * Matrix3::identity();
        StressTensor { matrix: dev }
    }

    /// Von Mises equivalent stress.
    pub fn von_mises(&self) -> f64 {
        let s = self.deviatoric().matrix;
        ((3.0 / 2.0) * (
            s[(0, 0)] * s[(0, 0)] + s[(1, 1)] * s[(1, 1)] + s[(2, 2)] * s[(2, 2)]
            + 2.0 * (s[(0, 1)] * s[(0, 1)] + s[(0, 2)] * s[(0, 2)] + s[(1, 2)] * s[(1, 2)])
        )).sqrt()
    }

    /// First invariant (trace): I₁ = σ_xx + σ_yy + σ_zz
    pub fn first_invariant(&self) -> f64 {
        self.matrix.trace()
    }

    /// Second invariant: I₂ = σ_xx*σ_yy + σ_yy*σ_zz + σ_zz*σ_xx - τ_xy² - τ_yz² - τ_xz²
    pub fn second_invariant(&self) -> f64 {
        let m = &self.matrix;
        m[(0, 0)] * m[(1, 1)] + m[(1, 1)] * m[(2, 2)] + m[(2, 2)] * m[(0, 0)]
            - m[(0, 1)] * m[(0, 1)] - m[(1, 2)] * m[(1, 2)] - m[(0, 2)] * m[(0, 2)]
    }

    /// Third invariant (determinant): I₃ = det(σ)
    pub fn third_invariant(&self) -> f64 {
        self.matrix.determinant()
    }

    /// Principal stresses (eigenvalues of the stress tensor), returned in descending order.
    pub fn principal_stresses(&self) -> Vector3<f64> {
        let eig = self.matrix.symmetric_eigen();
        let mut vals = eig.eigenvalues.data.0[0];
        vals.sort_by(|a, b| b.partial_cmp(a).unwrap());
        Vector3::new(vals[0], vals[1], vals[2])
    }

    /// Maximum shear stress: τ_max = (σ₁ - σ₃) / 2
    pub fn max_shear(&self) -> f64 {
        let p = self.principal_stresses();
        (p[0] - p[2]) / 2.0
    }

    /// Traction vector on a plane with normal n: t = σ · n
    pub fn traction(&self, normal: &Vector3<f64>) -> Vector3<f64> {
        self.matrix * normal
    }

    /// Normal stress on a plane: σ_n = n · (σ · n)
    pub fn normal_stress_on_plane(&self, normal: &Vector3<f64>) -> f64 {
        let t = self.traction(normal);
        normal.dot(&t)
    }

    /// Shear stress on a plane: τ = |t - σ_n · n|
    pub fn shear_stress_on_plane(&self, normal: &Vector3<f64>) -> f64 {
        let t = self.traction(normal);
        let sn = self.normal_stress_on_plane(normal);
        let tau_vec = t - sn * normal;
        tau_vec.norm()
    }

    /// Create a uniaxial stress state (σ_xx = σ, all others zero).
    pub fn uniaxial(sigma: f64) -> Self {
        Self::from_components(sigma, 0.0, 0.0, 0.0, 0.0, 0.0)
    }

    /// Create a biaxial stress state.
    pub fn biaxial(sxx: f64, syy: f64) -> Self {
        Self::from_components(sxx, syy, 0.0, 0.0, 0.0, 0.0)
    }

    /// Create a pure shear stress state.
    pub fn pure_shear(txy: f64) -> Self {
        Self::from_components(0.0, 0.0, 0.0, txy, 0.0, 0.0)
    }
}

impl std::ops::Add for StressTensor {
    type Output = StressTensor;
    fn add(self, rhs: StressTensor) -> Self::Output {
        StressTensor { matrix: self.matrix + rhs.matrix }
    }
}

impl std::ops::Mul<f64> for StressTensor {
    type Output = StressTensor;
    fn mul(self, scalar: f64) -> Self::Output {
        StressTensor { matrix: self.matrix * scalar }
    }
}

/// Infinitesimal strain tensor — symmetric 3×3 tensor for small deformations.
/// ε_ij = 0.5 * (∂u_i/∂x_j + ∂u_j/∂x_i)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrainTensor {
    pub matrix: Matrix3<f64>,
}

impl StrainTensor {
    /// Create from a 3×3 matrix (symmetrized).
    pub fn new(matrix: Matrix3<f64>) -> Self {
        let sym = (matrix + matrix.transpose()) * 0.5;
        Self { matrix: sym }
    }

    /// Create from individual components.
    pub fn from_components(
        exx: f64, eyy: f64, ezz: f64,
        exy: f64, exz: f64, eyz: f64,
    ) -> Self {
        // Engineering shear strain: γ = 2ε
        Self {
            matrix: Matrix3::new(
                exx, exy, exz,
                exy, eyy, eyz,
                exz, eyz, ezz,
            ),
        }
    }

    /// Create from engineering shear strains (γ_xy = 2ε_xy, etc.).
    pub fn from_engineering(
        exx: f64, eyy: f64, ezz: f64,
        gamma_xy: f64, gamma_xz: f64, gamma_yz: f64,
    ) -> Self {
        Self::from_components(exx, eyy, ezz, gamma_xy / 2.0, gamma_xz / 2.0, gamma_yz / 2.0)
    }

    /// Volumetric strain: ε_v = ε_xx + ε_yy + ε_zz
    pub fn volumetric(&self) -> f64 {
        self.matrix.trace()
    }

    /// Deviatoric strain tensor.
    pub fn deviatoric(&self) -> StrainTensor {
        let ev = self.volumetric() / 3.0;
        let dev = self.matrix - ev * Matrix3::identity();
        StrainTensor { matrix: dev }
    }

    /// Principal strains (eigenvalues), descending order.
    pub fn principal_strains(&self) -> Vector3<f64> {
        let eig = self.matrix.symmetric_eigen();
        let mut vals = eig.eigenvalues.data.0[0];
        vals.sort_by(|a, b| b.partial_cmp(a).unwrap());
        Vector3::new(vals[0], vals[1], vals[2])
    }

    /// Maximum engineering shear strain.
    pub fn max_shear_strain(&self) -> f64 {
        let p = self.principal_strains();
        p[0] - p[2]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_stress_tensor_symmetry() {
        let s = StressTensor::new(Matrix3::new(
            10.0, 5.0, 3.0,
            4.0, 20.0, 6.0,
            7.0, 8.0, 30.0,
        ));
        assert_relative_eq!(s.matrix[(0, 1)], s.matrix[(1, 0)]);
        assert_relative_eq!(s.matrix[(0, 2)], s.matrix[(2, 0)]);
        assert_relative_eq!(s.matrix[(1, 2)], s.matrix[(2, 1)]);
    }

    #[test]
    fn test_hydrostatic_stress() {
        let s = StressTensor::from_components(30.0, 30.0, 30.0, 0.0, 0.0, 0.0);
        assert_relative_eq!(s.hydrostatic(), 30.0);
    }

    #[test]
    fn test_deviatoric_trace_zero() {
        let s = StressTensor::from_components(10.0, 20.0, 30.0, 5.0, 0.0, 0.0);
        let dev = s.deviatoric();
        assert_relative_eq!(dev.matrix.trace(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_principal_stresses_ordering() {
        let s = StressTensor::from_components(30.0, 10.0, 20.0, 0.0, 0.0, 0.0);
        let p = s.principal_stresses();
        assert!(p[0] >= p[1]);
        assert!(p[1] >= p[2]);
        assert_relative_eq!(p[0], 30.0);
        assert_relative_eq!(p[1], 20.0);
        assert_relative_eq!(p[2], 10.0);
    }

    #[test]
    fn test_max_shear() {
        let s = StressTensor::from_components(100.0, 0.0, -50.0, 0.0, 0.0, 0.0);
        assert_relative_eq!(s.max_shear(), 75.0);
    }

    #[test]
    fn test_uniaxial_von_mises() {
        let s = StressTensor::uniaxial(100.0);
        assert_relative_eq!(s.von_mises(), 100.0);
    }

    #[test]
    fn test_strain_volumetric() {
        let e = StrainTensor::from_components(0.001, 0.002, 0.003, 0.0, 0.0, 0.0);
        assert_relative_eq!(e.volumetric(), 0.006);
    }

    #[test]
    fn test_strain_deviatoric_trace_zero() {
        let e = StrainTensor::from_components(0.001, 0.002, 0.003, 0.0, 0.0, 0.0);
        let dev = e.deviatoric();
        assert_relative_eq!(dev.matrix.trace(), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_traction_vector() {
        let s = StressTensor::from_components(10.0, 0.0, 0.0, 5.0, 0.0, 0.0);
        let n = Vector3::new(1.0, 0.0, 0.0);
        let t = s.traction(&n);
        assert_relative_eq!(t[0], 10.0);
        assert_relative_eq!(t[1], 5.0);
    }
}
