//! Euler-Bernoulli beam mechanics — deflection, bending moments, shear forces.

use serde::{Deserialize, Serialize};

/// Euler-Bernoulli beam model for a prismatic beam.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EulerBernoulliBeam {
    /// Young's modulus (Pa)
    pub e: f64,
    /// Second moment of area (m⁴)
    pub i: f64,
    /// Beam length (m)
    pub length: f64,
    /// Applied loads
    pub loads: Vec<BeamLoad>,
}

/// A load applied to the beam.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BeamLoad {
    /// Point load: (magnitude in N, position along beam in m)
    Point { p: f64, a: f64 },
    /// Uniformly distributed load: (magnitude in N/m, starts at, ends at)
    Distributed { w: f64, a_start: f64, a_end: f64 },
    /// Point moment: (magnitude in N·m, position along beam in m)
    Moment { m: f64, a: f64 },
}

impl EulerBernoulliBeam {
    /// Create a new simply-supported beam.
    pub fn new(e: f64, i: f64, length: f64) -> Self {
        Self { e, i, length, loads: Vec::new() }
    }

    /// Flexural rigidity EI.
    pub fn flexural_rigidity(&self) -> f64 {
        self.e * self.i
    }

    /// Add a point load.
    pub fn with_point_load(mut self, p: f64, a: f64) -> Self {
        self.loads.push(BeamLoad::Point { p, a });
        self
    }

    /// Add a uniformly distributed load.
    pub fn with_distributed_load(mut self, w: f64, a_start: f64, a_end: f64) -> Self {
        self.loads.push(BeamLoad::Distributed { w, a_start, a_end });
        self
    }

    /// Add a point moment.
    pub fn with_moment(mut self, m: f64, a: f64) -> Self {
        self.loads.push(BeamLoad::Moment { m, a });
        self
    }

    /// Deflection at position x for a simply-supported beam with a single centered point load P.
    /// v(x) = Px(L² - 4a²)/(48EI) for a = L/2
    /// General: v(x) = Pb(L² - b²)^(1/2) x / (9√3 EI L) for max deflection
    ///
    /// For a simply supported beam with point load P at distance a from left support:
    /// v(x) = Pb/(6LEI) * [L²/b - b²)x - x³/L] for x ≤ a  (simplified version)
    pub fn deflection_simply_supported_point(&self, x: f64, p: f64, a: f64) -> f64 {
        let l = self.length;
        let ei = self.flexural_rigidity();
        let b = l - a;
        if x <= a {
            p * b / (6.0 * l * ei) * ((l * l - b * b) * x - x.powi(3))
        } else {
            let x2 = l - x; // mirror
            p * a / (6.0 * l * ei) * ((l * l - a * a) * x2 - x2.powi(3))
        }
    }

    /// Maximum deflection for a simply-supported beam with centered point load P.
    /// δ_max = PL³ / (48EI)
    pub fn max_deflection_centered_point(&self, p: f64) -> f64 {
        let l = self.length;
        let ei = self.flexural_rigidity();
        p * l.powi(3) / (48.0 * ei)
    }

    /// Bending moment at position x for a simply-supported beam with point load P at a.
    /// M(x) = Pb·x/L for x ≤ a, M(x) = Pa(L-x)/L for x > a
    pub fn bending_moment_simply_supported_point(&self, x: f64, p: f64, a: f64) -> f64 {
        let l = self.length;
        let b = l - a;
        if x <= a {
            p * b * x / l
        } else {
            p * a * (l - x) / l
        }
    }

    /// Shear force at position x for a simply-supported beam with point load P at a.
    pub fn shear_force_simply_supported_point(&self, x: f64, p: f64, a: f64) -> f64 {
        let l = self.length;
        let b = l - a;
        if x < a {
            p * b / l
        } else {
            -p * a / l
        }
    }

    /// Maximum bending moment (at load point) for simply-supported beam with point load at a.
    pub fn max_bending_moment_point(&self, p: f64, a: f64) -> f64 {
        let l = self.length;
        let b = l - a;
        p * a * b / l
    }

    /// Deflection at position x for a cantilever beam with point load P at the free end.
    /// v(x) = P/(6EI) * (3Lx² - x³)
    pub fn deflection_cantilever_point_end(&self, x: f64, p: f64) -> f64 {
        let l = self.length;
        let ei = self.flexural_rigidity();
        p / (6.0 * ei) * (3.0 * l * x * x - x.powi(3))
    }

    /// Maximum deflection of cantilever with end point load: δ = PL³ / (3EI)
    pub fn max_deflection_cantilever_point_end(&self, p: f64) -> f64 {
        let l = self.length;
        let ei = self.flexural_rigidity();
        p * l.powi(3) / (3.0 * ei)
    }

    /// Bending moment at position x for cantilever with end point load.
    /// M(x) = P(L - x)
    pub fn bending_moment_cantilever_point_end(&self, x: f64, p: f64) -> f64 {
        let l = self.length;
        p * (l - x)
    }

    /// Deflection at position x for a cantilever with uniformly distributed load w (N/m).
    /// v(x) = w/(24EI) * (x⁴ - 4Lx³ + 6L²x²)
    pub fn deflection_cantilever_udl(&self, x: f64, w: f64) -> f64 {
        let l = self.length;
        let ei = self.flexural_rigidity();
        w / (24.0 * ei) * (x.powi(4) - 4.0 * l * x.powi(3) + 6.0 * l * l * x * x)
    }

    /// Max deflection of cantilever with UDL: δ = wL⁴/(8EI)
    pub fn max_deflection_cantilever_udl(&self, w: f64) -> f64 {
        let l = self.length;
        let ei = self.flexural_rigidity();
        w * l.powi(4) / (8.0 * ei)
    }

    /// Bending stress at position (x, y): σ = My/I
    pub fn bending_stress(&self, moment: f64, y: f64) -> f64 {
        moment * y / self.i
    }

    /// Maximum bending stress for given moment and section height h: σ = M(h/2)/I
    pub fn max_bending_stress(&self, moment: f64, section_height: f64) -> f64 {
        self.bending_stress(moment, section_height / 2.0)
    }

    /// Reaction forces for simply-supported beam with point load P at a.
    /// Returns (R_left, R_right).
    pub fn reactions_simply_supported_point(&self, p: f64, a: f64) -> (f64, f64) {
        let l = self.length;
        let b = l - a;
        (p * b / l, p * a / l)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_simply_supported_centered_deflection() {
        let beam = EulerBernoulliBeam::new(200e9, 1e-4, 5.0);
        // P at center: δ = PL³/(48EI)
        let p = 10000.0;
        let expected = p * 125.0 / (48.0 * 200e9 * 1e-4);
        let actual = beam.max_deflection_centered_point(p);
        assert_relative_eq!(actual, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_cantilever_end_deflection() {
        let beam = EulerBernoulliBeam::new(200e9, 1e-4, 3.0);
        let p = 5000.0;
        let expected = p * 27.0 / (3.0 * 200e9 * 1e-4);
        let actual = beam.max_deflection_cantilever_point_end(p);
        assert_relative_eq!(actual, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_bending_moment_simply_supported() {
        let beam = EulerBernoulliBeam::new(200e9, 1e-4, 4.0);
        let p = 10000.0;
        let a = 2.0;
        let m = beam.max_bending_moment_point(p, a);
        assert_relative_eq!(m, 10000.0, epsilon = 1e-6);
    }

    #[test]
    fn test_reactions_sum_to_load() {
        let beam = EulerBernoulliBeam::new(200e9, 1e-4, 4.0);
        let p = 10000.0;
        let (rl, rr) = beam.reactions_simply_supported_point(p, 1.0);
        assert_relative_eq!(rl + rr, p);
        assert_relative_eq!(rl, 7500.0);
        assert_relative_eq!(rr, 2500.0);
    }

    #[test]
    fn test_bending_stress() {
        let beam = EulerBernoulliBeam::new(200e9, 1e-4, 4.0);
        // σ = My/I = 5000 * 0.05 / 1e-4 = 2_500_000
        let stress = beam.bending_stress(5000.0, 0.05);
        assert_relative_eq!(stress, 2_500_000.0);
    }

    #[test]
    fn test_cantilever_udl_deflection() {
        let beam = EulerBernoulliBeam::new(200e9, 1e-4, 2.0);
        let w = 5000.0;
        let expected = w * 16.0 / (8.0 * 200e9 * 1e-4);
        let actual = beam.max_deflection_cantilever_udl(w);
        assert_relative_eq!(actual, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_deflection_zero_at_supports() {
        let beam = EulerBernoulliBeam::new(200e9, 1e-4, 5.0);
        let p = 10000.0;
        let a = 2.5;
        let d0 = beam.deflection_simply_supported_point(0.0, p, a);
        let d_l = beam.deflection_simply_supported_point(5.0, p, a);
        assert_relative_eq!(d0, 0.0, epsilon = 1e-10);
        assert_relative_eq!(d_l, 0.0, epsilon = 1e-10);
    }
}
