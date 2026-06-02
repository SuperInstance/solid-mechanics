//! Constitutive relations — Hooke's law and elasticity tensors.

use nalgebra::{Matrix3, Matrix6, Vector6};
use serde::{Deserialize, Serialize};

use crate::tensor::{StressTensor, StrainTensor};

/// Isotropic linear elastic material using Hooke's law.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HookeIsotropic {
    /// Young's modulus (Pa)
    pub youngs_modulus: f64,
    /// Poisson's ratio
    pub poissons_ratio: f64,
}

impl HookeIsotropic {
    /// Create a new isotropic elastic material.
    pub fn new(e: f64, nu: f64) -> Self {
        Self { youngs_modulus: e, poissons_ratio: nu }
    }

    /// Shear modulus: G = E / (2(1 + ν))
    pub fn shear_modulus(&self) -> f64 {
        self.youngs_modulus / (2.0 * (1.0 + self.poissons_ratio))
    }

    /// Bulk modulus: K = E / (3(1 - 2ν))
    pub fn bulk_modulus(&self) -> f64 {
        self.youngs_modulus / (3.0 * (1.0 - 2.0 * self.poissons_ratio))
    }

    /// Lamé's first parameter: λ = νE / ((1+ν)(1-2ν))
    pub fn lame_lambda(&self) -> f64 {
        let nu = self.poissons_ratio;
        nu * self.youngs_modulus / ((1.0 + nu) * (1.0 - 2.0 * nu))
    }

    /// Lamé's second parameter (shear modulus): μ = G
    pub fn lame_mu(&self) -> f64 {
        self.shear_modulus()
    }

    /// Compute stress from strain using Hooke's law (3D).
    /// σ_ij = λ ε_kk δ_ij + 2μ ε_ij
    pub fn stress_from_strain(&self, strain: &StrainTensor) -> StressTensor {
        let lam = self.lame_lambda();
        let mu = self.lame_mu();
        let tr = strain.matrix.trace();
        let sigma = lam * tr * Matrix3::identity() + 2.0 * mu * strain.matrix;
        StressTensor { matrix: sigma }
    }

    /// Compute strain from stress using compliance (3D).
    pub fn strain_from_stress(&self, stress: &StressTensor) -> StrainTensor {
        let e = self.youngs_modulus;
        let nu = self.poissons_ratio;
        let s = &stress.matrix;
        let tr = s.trace();
        let eps = ((1.0 + nu) / e) * *s - (nu / e) * tr * Matrix3::identity();
        StrainTensor { matrix: eps }
    }

    /// 6×6 stiffness matrix (Voigt notation): [σ_xx, σ_yy, σ_zz, τ_yz, τ_xz, τ_xy]
    pub fn stiffness_matrix_voigt(&self) -> Matrix6<f64> {
        let nu = self.poissons_ratio;
        let e = self.youngs_modulus;
        let factor = e / ((1.0 + nu) * (1.0 - 2.0 * nu));
        let lam_nu = 1.0 - nu;
        let c12 = factor * nu;
        let c11 = factor * lam_nu;
        let c44 = e / (2.0 * (1.0 + nu));
        Matrix6::from_row_slice(&[
            c11, c12, c12, 0.0, 0.0, 0.0,
            c12, c11, c12, 0.0, 0.0, 0.0,
            c12, c12, c11, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, c44, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, c44, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, c44,
        ])
    }

    /// 6×6 compliance matrix (Voigt notation).
    pub fn compliance_matrix_voigt(&self) -> Matrix6<f64> {
        let e = self.youngs_modulus;
        let nu = self.poissons_ratio;
        let s11 = 1.0 / e;
        let s12 = -nu / e;
        let s44 = 2.0 * (1.0 + nu) / e;
        Matrix6::from_row_slice(&[
            s11, s12, s12, 0.0, 0.0, 0.0,
            s12, s11, s12, 0.0, 0.0, 0.0,
            s12, s12, s11, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, s44, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, s44, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, s44,
        ])
    }

    /// Axial stiffness EA.
    pub fn axial_stiffness(&self, area: f64) -> f64 {
        self.youngs_modulus * area
    }

    /// Flexural stiffness EI.
    pub fn flexural_stiffness(&self, moment_of_inertia: f64) -> f64 {
        self.youngs_modulus * moment_of_inertia
    }
}

/// General anisotropic elasticity tensor (6×6 Voigt).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElasticityTensor {
    /// Full 6×6 stiffness matrix in Voigt notation
    pub c: Matrix6<f64>,
}

impl ElasticityTensor {
    /// Create from a 6×6 Voigt stiffness matrix.
    pub fn new(c: Matrix6<f64>) -> Self {
        Self { c }
    }

    /// Create an orthotropic material from 9 independent constants.
    pub fn orthotropic(
        e1: f64, e2: f64, e3: f64,
        nu12: f64, nu13: f64, nu23: f64,
        g12: f64, g13: f64, g23: f64,
    ) -> Self {
        // Build compliance first, then invert
        let nu21 = nu12 * e2 / e1;
        let nu31 = nu13 * e3 / e1;
        let nu32 = nu23 * e3 / e2;
        let delta = 1.0 - nu12 * nu21 - nu23 * nu32 - nu31 * nu13 - 2.0 * nu21 * nu32 * nu13;

        let _c11 = (1.0 - nu23 * nu32) / (e2 * e3 * delta);
        let _c22 = (1.0 - nu13 * nu31) / (e1 * e3 * delta);
        let _c33 = (1.0 - nu12 * nu21) / (e1 * e2 * delta);
        let _c12 = (nu21 + nu31 * nu23) / (e1 * e3 * delta);
        let _c13 = (nu31 + nu21 * nu32) / (e1 * e2 * delta);
        let _c23 = (nu32 + nu12 * nu31) / (e2 * e1 * delta);

        // Simplified: build stiffness from compliance inversion
        let s = Matrix6::from_row_slice(&[
            1.0/e1, -nu12/e1, -nu13/e1, 0.0, 0.0, 0.0,
            -nu21/e2, 1.0/e2, -nu23/e2, 0.0, 0.0, 0.0,
            -nu31/e3, -nu32/e3, 1.0/e3, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0/g12, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 1.0/g13, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0/g23,
        ]);

        Self { c: s.try_inverse().unwrap_or(Matrix6::zeros()) }
    }

    /// Compute stress from strain (Voigt vectors).
    pub fn stress_from_strain(&self, strain_voigt: &Vector6<f64>) -> Vector6<f64> {
        self.c * strain_voigt
    }

    /// Compliance matrix (inverse of stiffness).
    pub fn compliance(&self) -> Option<Matrix6<f64>> {
        self.c.try_inverse()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_hooke_shear_modulus() {
        let h = HookeIsotropic::new(200e9, 0.3);
        let g = 200e9 / (2.0 * 1.3);
        assert_relative_eq!(h.shear_modulus(), g);
    }

    #[test]
    fn test_hooke_bulk_modulus() {
        let h = HookeIsotropic::new(200e9, 0.3);
        let k = 200e9 / (3.0 * 0.4);
        assert_relative_eq!(h.bulk_modulus(), k);
    }

    #[test]
    fn test_hooke_roundtrip() {
        let h = HookeIsotropic::new(200e9, 0.3);
        let strain = StrainTensor::from_components(1e-3, 0.5e-3, -0.5e-3, 0.2e-3, 0.0, 0.0);
        let stress = h.stress_from_strain(&strain);
        let strain2 = h.strain_from_stress(&stress);
        assert_relative_eq!(strain.matrix[(0, 0)], strain2.matrix[(0, 0)], epsilon = 1e-6);
        assert_relative_eq!(strain.matrix[(1, 1)], strain2.matrix[(1, 1)], epsilon = 1e-6);
    }

    #[test]
    fn test_stiffness_compliance_inverse() {
        let h = HookeIsotropic::new(200e9, 0.3);
        let c = h.stiffness_matrix_voigt();
        let s = h.compliance_matrix_voigt();
        let product = c * s;
        let identity = Matrix6::identity();
        for i in 0..6 {
            for j in 0..6 {
                assert_relative_eq!(product[(i, j)], identity[(i, j)], epsilon = 1e-3);
            }
        }
    }

    #[test]
    fn test_orthotropic_creation() {
        let ortho = ElasticityTensor::orthotropic(
            140e9, 10e9, 10e9,  // E1, E2, E3
            0.3, 0.3, 0.4,      // nu12, nu13, nu23
            5e9, 5e9, 3e9,      // G12, G13, G23
        );
        // Verify it's not zero (i.e., inversion succeeded)
        assert!(ortho.c.norm() > 0.0);
    }
}
