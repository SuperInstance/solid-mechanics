//! Energy methods — strain energy, Castigliano's theorem.

use crate::constitutive::HookeIsotropic;
use crate::tensor::{StressTensor, StrainTensor};

/// Strain energy calculations.
pub struct StrainEnergy;

impl StrainEnergy {
    /// Strain energy density (per unit volume) from stress and strain:
    /// u = 0.5 * σ_ij * ε_ij = 0.5 * tr(σ · ε)
    pub fn density_3d(stress: &StressTensor, strain: &StrainTensor) -> f64 {
        0.5 * (stress.matrix.component_mul(&strain.matrix)).sum()
    }

    /// Strain energy density from stress only (using compliance):
    /// u = σ_ij * S_ijkl * σ_kl / 2
    pub fn density_from_stress(stress: &StressTensor, material: &HookeIsotropic) -> f64 {
        let strain = material.strain_from_stress(stress);
        Self::density_3d(stress, &strain)
    }

    /// Strain energy density from strain only (using stiffness):
    /// u = ε_ij * C_ijkl * ε_kl / 2
    pub fn density_from_strain(strain: &StrainTensor, material: &HookeIsotropic) -> f64 {
        let stress = material.stress_from_strain(strain);
        Self::density_3d(&stress, strain)
    }

    /// Total strain energy = ∫ u dV ≈ u * volume (uniform field).
    pub fn total_uniform(stress: &StressTensor, strain: &StrainTensor, volume: f64) -> f64 {
        Self::density_3d(stress, strain) * volume
    }

    /// Strain energy for a bar under axial load: U = F²L / (2EA)
    pub fn bar_axial(force: f64, length: f64, e: f64, area: f64) -> f64 {
        force.powi(2) * length / (2.0 * e * area)
    }

    /// Strain energy for a beam in bending: U = ∫ M²/(2EI) dx
    /// For simply-supported beam with centered point load P:
    /// U = P²L³/(96EI)
    pub fn beam_bending_simply_supported_centered(p: f64, l: f64, e: f64, i: f64) -> f64 {
        p.powi(2) * l.powi(3) / (96.0 * e * i)
    }

    /// Strain energy for a beam in bending (cantilever, end load):
    /// U = P²L³/(6EI)
    pub fn beam_bending_cantilever_end(p: f64, l: f64, e: f64, i: f64) -> f64 {
        p.powi(2) * l.powi(3) / (6.0 * e * i)
    }

    /// Castigliano's first theorem: displacement = ∂U/∂F
    /// For axial bar: δ = FL/(EA)
    pub fn castigliano_displacement_bar(force: f64, length: f64, e: f64, area: f64) -> f64 {
        force * length / (e * area)
    }

    /// Castigliano's theorem for simply-supported beam with centered point load:
    /// δ = PL²/(48EI)
    pub fn castigliano_deflection_beam_ss_centered(p: f64, l: f64, e: f64, i: f64) -> f64 {
        p * l.powi(2) / (48.0 * e * i)
    }

    /// Complementary strain energy density: u* = 0.5 * σ : S : σ
    /// For linear elastic materials, u* = u.
    pub fn complementary_density(stress: &StressTensor, material: &HookeIsotropic) -> f64 {
        Self::density_from_stress(stress, material)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_strain_energy_uniaxial() {
        let stress = StressTensor::uniaxial(100e6);
        let mat = HookeIsotropic::new(200e9, 0.3);
        let strain = mat.strain_from_stress(&stress);
        let u = StrainEnergy::density_3d(&stress, &strain);
        // u = 0.5 * σ * ε = 0.5 * 100e6 * 100e6/200e9 = 25000
        assert_relative_eq!(u, 0.5 * 100e6 * 100e6 / 200e9);
    }

    #[test]
    fn test_energy_from_stress_equals_from_strain() {
        let mat = HookeIsotropic::new(200e9, 0.3);
        let strain = StrainTensor::from_components(1e-3, 0.5e-3, -0.3e-3, 0.1e-3, 0.0, 0.0);
        let u1 = StrainEnergy::density_from_strain(&strain, &mat);
        let stress = mat.stress_from_strain(&strain);
        let u2 = StrainEnergy::density_from_stress(&stress, &mat);
        assert_relative_eq!(u1, u2, epsilon = 1e-3);
    }

    #[test]
    fn test_bar_axial_energy() {
        let u = StrainEnergy::bar_axial(10000.0, 1.0, 200e9, 1e-4);
        assert_relative_eq!(u, 1e8 / (2.0 * 200e9 * 1e-4));
    }

    #[test]
    fn test_beam_bending_energy_cantilever() {
        let u = StrainEnergy::beam_bending_cantilever_end(5000.0, 3.0, 200e9, 1e-4);
        let expected = 5000.0_f64.powi(2) * 27.0 / (6.0 * 200e9 * 1e-4);
        assert_relative_eq!(u, expected);
    }

    #[test]
    fn test_castigliano_bar() {
        let delta = StrainEnergy::castigliano_displacement_bar(10000.0, 1.0, 200e9, 1e-4);
        let expected = 10000.0 / (200e9 * 1e-4);
        assert_relative_eq!(delta, expected);
    }

    #[test]
    fn test_complementary_equals_strain_energy_linear() {
        let mat = HookeIsotropic::new(200e9, 0.3);
        let stress = StressTensor::from_components(100e6, 50e6, 0.0, 20e6, 0.0, 0.0);
        let u = StrainEnergy::density_from_stress(&stress, &mat);
        let uc = StrainEnergy::complementary_density(&stress, &mat);
        assert_relative_eq!(u, uc, epsilon = 1e-6);
    }

    #[test]
    fn test_total_strain_energy_volume() {
        let stress = StressTensor::uniaxial(100e6);
        let mat = HookeIsotropic::new(200e9, 0.3);
        let strain = mat.strain_from_stress(&stress);
        let volume = 0.01; // m³
        let u_total = StrainEnergy::total_uniform(&stress, &strain, volume);
        let u_density = StrainEnergy::density_3d(&stress, &strain);
        assert_relative_eq!(u_total, u_density * volume);
    }
}
