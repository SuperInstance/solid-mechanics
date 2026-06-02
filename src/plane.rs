//! Plane stress and plane strain conditions.

use serde::{Deserialize, Serialize};

use crate::constitutive::HookeIsotropic;

/// Plane stress condition (σ_zz = τ_xz = τ_yz = 0).
/// Thin structures loaded in-plane.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlaneStress {
    pub material: HookeIsotropic,
}

impl PlaneStress {
    pub fn new(material: HookeIsotropic) -> Self {
        Self { material }
    }

    /// 2D stress → 2D strain under plane stress.
    /// ε_xx = (σ_xx - ν·σ_yy) / E
    /// ε_yy = (σ_yy - ν·σ_xx) / E
    /// γ_xy = τ_xy / G
    pub fn strain_from_stress_2d(&self, sxx: f64, syy: f64, txy: f64) -> (f64, f64, f64) {
        let e = self.material.youngs_modulus;
        let nu = self.material.poissons_ratio;
        let g = self.material.shear_modulus();
        let exx = (sxx - nu * syy) / e;
        let eyy = (syy - nu * sxx) / e;
        let gxy = txy / g;
        (exx, eyy, gxy)
    }

    /// 2D strain → 2D stress under plane stress.
    pub fn stress_from_strain_2d(&self, exx: f64, eyy: f64, gxy: f64) -> (f64, f64, f64) {
        let e = self.material.youngs_modulus;
        let nu = self.material.poissons_ratio;
        let factor = e / (1.0 - nu * nu);
        let sxx = factor * (exx + nu * eyy);
        let syy = factor * (eyy + nu * exx);
        let txy = self.material.shear_modulus() * gxy;
        (sxx, syy, txy)
    }

    /// Out-of-plane strain under plane stress: ε_zz = -ν(σ_xx + σ_yy)/E
    pub fn out_of_plane_strain(&self, sxx: f64, syy: f64) -> f64 {
        let e = self.material.youngs_modulus;
        let nu = self.material.poissons_ratio;
        -nu * (sxx + syy) / e
    }

    /// Plane stress stiffness matrix (3×3 for [ε_xx, ε_yy, γ_xy]).
    pub fn stiffness_matrix_2d(&self) -> [[f64; 3]; 3] {
        let e = self.material.youngs_modulus;
        let nu = self.material.poissons_ratio;
        let factor = e / (1.0 - nu * nu);
        let g = e / (2.0 * (1.0 + nu));
        [
            [factor, factor * nu, 0.0],
            [factor * nu, factor, 0.0],
            [0.0, 0.0, g],
        ]
    }
}

/// Plane strain condition (ε_zz = γ_xz = γ_yz = 0).
/// Thick structures or constrained in z-direction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlaneStrain {
    pub material: HookeIsotropic,
}

impl PlaneStrain {
    pub fn new(material: HookeIsotropic) -> Self {
        Self { material }
    }

    /// 2D stress → 2D strain under plane strain.
    pub fn strain_from_stress_2d(&self, sxx: f64, syy: f64, txy: f64) -> (f64, f64, f64) {
        let e = self.material.youngs_modulus;
        let nu = self.material.poissons_ratio;
        let g = self.material.shear_modulus();
        let factor = (1.0 + nu) * (1.0 - 2.0 * nu);
        let exx = ((1.0 - nu) * sxx - nu * syy) / (e * factor / (1.0 + nu));
        let eyy = ((1.0 - nu) * syy - nu * sxx) / (e * factor / (1.0 + nu));
        let gxy = txy / g;
        (exx, eyy, gxy)
    }

    /// 2D strain → 2D stress under plane strain.
    pub fn stress_from_strain_2d(&self, exx: f64, eyy: f64, gxy: f64) -> (f64, f64, f64) {
        let e = self.material.youngs_modulus;
        let nu = self.material.poissons_ratio;
        let factor = e / ((1.0 + nu) * (1.0 - 2.0 * nu));
        let sxx = factor * ((1.0 - nu) * exx + nu * eyy);
        let syy = factor * (nu * exx + (1.0 - nu) * eyy);
        let txy = self.material.shear_modulus() * gxy;
        (sxx, syy, txy)
    }

    /// Out-of-plane stress under plane strain: σ_zz = ν(σ_xx + σ_yy)
    pub fn out_of_plane_stress(&self, sxx: f64, syy: f64) -> f64 {
        self.material.poissons_ratio * (sxx + syy)
    }

    /// Plane strain stiffness matrix (3×3).
    pub fn stiffness_matrix_2d(&self) -> [[f64; 3]; 3] {
        let e = self.material.youngs_modulus;
        let nu = self.material.poissons_ratio;
        let factor = e / ((1.0 + nu) * (1.0 - 2.0 * nu));
        let g = e / (2.0 * (1.0 + nu));
        [
            [factor * (1.0 - nu), factor * nu, 0.0],
            [factor * nu, factor * (1.0 - nu), 0.0],
            [0.0, 0.0, g],
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_plane_stress_roundtrip() {
        let mat = HookeIsotropic::new(200e9, 0.3);
        let ps = PlaneStress::new(mat);
        let (sxx, syy, txy) = (100e6, 50e6, 20e6);
        let (exx, eyy, gxy) = ps.strain_from_stress_2d(sxx, syy, txy);
        let (sxx2, syy2, txy2) = ps.stress_from_strain_2d(exx, eyy, gxy);
        assert_relative_eq!(sxx, sxx2, epsilon = 1e-3);
        assert_relative_eq!(syy, syy2, epsilon = 1e-3);
        assert_relative_eq!(txy, txy2, epsilon = 1e-3);
    }

    #[test]
    fn test_plane_strain_out_of_plane() {
        let mat = HookeIsotropic::new(200e9, 0.3);
        let pe = PlaneStrain::new(mat);
        let s_zz = pe.out_of_plane_stress(100e6, 50e6);
        assert_relative_eq!(s_zz, 0.3 * 150e6);
    }

    #[test]
    fn test_plane_stress_out_of_plane_strain() {
        let mat = HookeIsotropic::new(200e9, 0.3);
        let ps = PlaneStress::new(mat);
        let e_zz = ps.out_of_plane_strain(100e6, 50e6);
        assert_relative_eq!(e_zz, -0.3 * 150e6 / 200e9);
    }

    #[test]
    fn test_plane_stress_stiffness_symmetry() {
        let mat = HookeIsotropic::new(200e9, 0.3);
        let ps = PlaneStress::new(mat);
        let c = ps.stiffness_matrix_2d();
        assert_relative_eq!(c[0][1], c[1][0]);
    }
}
