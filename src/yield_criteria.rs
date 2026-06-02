//! Yield criteria — von Mises and Tresca.

use crate::tensor::StressTensor;
use serde::{Deserialize, Serialize};

/// Von Mises yield criterion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VonMises {
    /// Yield stress σ_y (Pa)
    pub yield_stress: f64,
}

impl VonMises {
    /// Create a new von Mises criterion with given yield stress.
    pub fn new(yield_stress: f64) -> Self {
        Self { yield_stress }
    }

    /// Evaluate the von Mises equivalent stress.
    pub fn equivalent_stress(stress: &StressTensor) -> f64 {
        stress.von_mises()
    }

    /// Check if the stress state has yielded.
    pub fn has_yielded(&self, stress: &StressTensor) -> bool {
        Self::equivalent_stress(stress) >= self.yield_stress
    }

    /// Safety factor: σ_y / σ_vm
    pub fn safety_factor(&self, stress: &StressTensor) -> f64 {
        self.yield_stress / Self::equivalent_stress(stress)
    }

    /// Yield ratio: σ_vm / σ_y (1.0 = yielding, >1.0 = post-yield)
    pub fn yield_ratio(&self, stress: &StressTensor) -> f64 {
        Self::equivalent_stress(stress) / self.yield_stress
    }
}

/// Tresca (maximum shear stress) yield criterion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tresca {
    /// Yield stress in shear (or tensile yield / 2 for von Mises equivalence)
    pub yield_stress_shear: f64,
}

impl Tresca {
    /// Create from tensile yield stress: τ_y = σ_y / 2.
    pub fn from_tensile(yield_stress: f64) -> Self {
        Self { yield_stress_shear: yield_stress / 2.0 }
    }

    /// Create from shear yield stress directly.
    pub fn new(yield_stress_shear: f64) -> Self {
        Self { yield_stress_shear }
    }

    /// Maximum shear stress.
    pub fn max_shear(stress: &StressTensor) -> f64 {
        stress.max_shear()
    }

    /// Check if yielded: τ_max ≥ τ_y
    pub fn has_yielded(&self, stress: &StressTensor) -> bool {
        Self::max_shear(stress) >= self.yield_stress_shear
    }

    /// Safety factor: τ_y / τ_max
    pub fn safety_factor(&self, stress: &StressTensor) -> f64 {
        self.yield_stress_shear / Self::max_shear(stress)
    }

    /// Yield ratio: τ_max / τ_y
    pub fn yield_ratio(&self, stress: &StressTensor) -> f64 {
        Self::max_shear(stress) / self.yield_stress_shear
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_von_mises_uniaxial() {
        let stress = StressTensor::uniaxial(250e6);
        let vm = VonMises::equivalent_stress(&stress);
        assert_relative_eq!(vm, 250e6);
    }

    #[test]
    fn test_von_mises_pure_shear() {
        let stress = StressTensor::pure_shear(100e6);
        let vm = VonMises::equivalent_stress(&stress);
        // σ_vm = √3 * τ for pure shear
        assert_relative_eq!(vm, 100e6 * 3.0_f64.sqrt(), epsilon = 1e-3);
    }

    #[test]
    fn test_von_mises_yield_check() {
        let vm = VonMises::new(250e6);
        let safe = StressTensor::uniaxial(200e6);
        let yielded = StressTensor::uniaxial(300e6);
        assert!(!vm.has_yielded(&safe));
        assert!(vm.has_yielded(&yielded));
    }

    #[test]
    fn test_von_mises_safety_factor() {
        let vm = VonMises::new(250e6);
        let stress = StressTensor::uniaxial(125e6);
        assert_relative_eq!(vm.safety_factor(&stress), 2.0);
    }

    #[test]
    fn test_von_mises_biaxial() {
        // Equal biaxial: σ_vm = σ (same as uniaxial)
        let stress = StressTensor::biaxial(100e6, 100e6);
        let vm = VonMises::equivalent_stress(&stress);
        assert_relative_eq!(vm, 100e6);
    }

    #[test]
    fn test_tresca_uniaxial() {
        let stress = StressTensor::uniaxial(250e6);
        let tau = Tresca::max_shear(&stress);
        assert_relative_eq!(tau, 125e6);
    }

    #[test]
    fn test_tresca_yield_check() {
        let tresca = Tresca::from_tensile(250e6);
        let safe = StressTensor::uniaxial(200e6);
        let yielded = StressTensor::uniaxial(300e6);
        assert!(!tresca.has_yielded(&safe));
        assert!(tresca.has_yielded(&yielded));
    }

    #[test]
    fn test_tresca_safety_factor() {
        let tresca = Tresca::from_tensile(250e6);
        let stress = StressTensor::uniaxial(125e6);
        assert_relative_eq!(tresca.safety_factor(&stress), 2.0);
    }

    #[test]
    fn test_von_mises_vs_tresca() {
        // For pure shear: VM equivalent = √3·τ, Tresca equivalent = 2·τ
        // Tresca gives higher equivalent stress (more conservative)
        let stress = StressTensor::pure_shear(100e6);
        let vm_stress = VonMises::equivalent_stress(&stress);
        let tresca_max_shear = Tresca::max_shear(&stress);
        let tresca_equiv = 2.0 * tresca_max_shear; // Tresca equivalent
        assert!(tresca_equiv >= vm_stress - 1e-3);
    }
}
