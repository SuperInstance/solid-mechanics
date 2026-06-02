//! Mohr's circle for stress transformation and principal stress analysis.

use serde::{Deserialize, Serialize};

use crate::tensor::StressTensor;

/// Mohr's circle representation for a 2D stress state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MohrCircle {
    /// Center of Mohr's circle (average normal stress): C = (σ_xx + σ_yy) / 2
    pub center: f64,
    /// Radius of Mohr's circle: R = √(((σ_xx - σ_yy)/2)² + τ_xy²)
    pub radius: f64,
    /// Principal stresses (σ₁ ≥ σ₂)
    pub sigma1: f64,
    pub sigma2: f64,
    /// Maximum in-plane shear stress
    pub tau_max: f64,
    /// Angle to principal plane (radians)
    pub theta_p: f64,
}

impl MohrCircle {
    /// Construct Mohr's circle from a 2D stress state (σ_xx, σ_yy, τ_xy).
    pub fn from_2d(sxx: f64, syy: f64, txy: f64) -> Self {
        let center = (sxx + syy) / 2.0;
        let radius = (((sxx - syy) / 2.0).powi(2) + txy.powi(2)).sqrt();
        let sigma1 = center + radius;
        let sigma2 = center - radius;
        let tau_max = radius;
        let theta_p = 0.5 * (2.0 * txy).atan2(sxx - syy);

        Self { center, radius, sigma1, sigma2, tau_max, theta_p }
    }

    /// Construct from a 3D stress tensor (uses in-plane components).
    pub fn from_stress_tensor_2d(stress: &StressTensor) -> Self {
        Self::from_2d(
            stress.matrix[(0, 0)],
            stress.matrix[(1, 1)],
            stress.matrix[(0, 1)],
        )
    }

    /// Full 3D Mohr's circle — returns three circles characterized by principal stresses.
    pub fn from_3d(stress: &StressTensor) -> [MohrCircle; 3] {
        let p = stress.principal_stresses();
        let s1 = p[0]; // max
        let s2 = p[1];
        let s3 = p[2]; // min
        [
            MohrCircle::from_2d(s1, s2, 0.0), // C12 circle
            MohrCircle::from_2d(s2, s3, 0.0), // C23 circle
            MohrCircle::from_2d(s1, s3, 0.0), // C13 circle (max)
        ]
    }

    /// Stress state at a given angle θ (from x-axis) on Mohr's circle.
    /// Returns (σ_n, τ_n) at that angle.
    pub fn stress_at_angle(&self, theta: f64) -> (f64, f64) {
        let sn = self.center + self.radius * (2.0 * theta).cos();
        let tn = -self.radius * (2.0 * theta).sin(); // negative per Mohr convention
        (sn, tn)
    }

    /// Maximum shear stress from 3D state (half the difference of max and min principal stresses).
    pub fn absolute_max_shear(stress: &StressTensor) -> f64 {
        let p = stress.principal_stresses();
        (p[0] - p[2]) / 2.0
    }
}

/// Stress transformation at an angle θ.
/// Rotates the 2D stress state by angle θ.
pub fn stress_transform_2d(sxx: f64, syy: f64, txy: f64, theta: f64) -> (f64, f64, f64) {
    let c = theta.cos();
    let s = theta.sin();
    let c2 = c * c;
    let s2 = s * s;
    let cs = c * s;

    let sx_x = sxx * c2 + syy * s2 + 2.0 * txy * cs;
    let sy_y = sxx * s2 + syy * c2 - 2.0 * txy * cs;
    let tx_y = -(sxx - syy) * cs + txy * (c2 - s2);

    (sx_x, sy_y, tx_y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_mohr_circle_pure_shear() {
        let mc = MohrCircle::from_2d(0.0, 0.0, 50.0);
        assert_relative_eq!(mc.center, 0.0);
        assert_relative_eq!(mc.radius, 50.0);
        assert_relative_eq!(mc.sigma1, 50.0);
        assert_relative_eq!(mc.sigma2, -50.0);
        assert_relative_eq!(mc.tau_max, 50.0);
    }

    #[test]
    fn test_mohr_circle_uniaxial() {
        let mc = MohrCircle::from_2d(100.0, 0.0, 0.0);
        assert_relative_eq!(mc.center, 50.0);
        assert_relative_eq!(mc.radius, 50.0);
        assert_relative_eq!(mc.sigma1, 100.0);
        assert_relative_eq!(mc.sigma2, 0.0);
    }

    #[test]
    fn test_mohr_circle_biaxial() {
        let mc = MohrCircle::from_2d(80.0, -40.0, 30.0);
        let c = (80.0 + (-40.0)) / 2.0;
        assert_relative_eq!(mc.center, c);
        let r = ((80.0_f64 - (-40.0_f64)) / 2.0).hypot(30.0);
        assert_relative_eq!(mc.radius, r);
        assert_relative_eq!(mc.sigma1, c + r);
        assert_relative_eq!(mc.sigma2, c - r);
    }

    #[test]
    fn test_stress_transform_principal_angle() {
        let mc = MohrCircle::from_2d(80.0, -40.0, 30.0);
        let (_sx, _sy, txy) = stress_transform_2d(80.0, -40.0, 30.0, mc.theta_p);
        // At principal angle, shear should be ~0
        assert_relative_eq!(txy, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_mohr_3d() {
        let stress = StressTensor::from_components(100.0, 50.0, -20.0, 0.0, 0.0, 0.0);
        let circles = MohrCircle::from_3d(&stress);
        // Largest circle should be between σ₁=100 and σ₃=-20
        assert_relative_eq!(circles[2].sigma1, 100.0);
        assert_relative_eq!(circles[2].sigma2, -20.0);
        assert_relative_eq!(circles[2].radius, 60.0);
    }

    #[test]
    fn test_stress_at_45_degrees() {
        let mc = MohrCircle::from_2d(100.0, 0.0, 0.0);
        let (sn, tn) = mc.stress_at_angle(std::f64::consts::FRAC_PI_4);
        assert_relative_eq!(sn, 50.0);
        assert_relative_eq!(tn.abs(), 50.0);
    }

    #[test]
    fn test_absolute_max_shear() {
        let stress = StressTensor::from_components(100.0, 50.0, -20.0, 0.0, 0.0, 0.0);
        let tau = MohrCircle::absolute_max_shear(&stress);
        assert_relative_eq!(tau, 60.0);
    }
}
