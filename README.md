# solid-mechanics

**Solid mechanics in Rust.** Stress, strain, and structural analysis.

## What This Does

- **Stress and strain tensors** — Cauchy stress and infinitesimal strain as symmetric 3×3 matrices, with hydrostatic/deviatoric decomposition, principal values, invariants, traction vectors
- **Constitutive relations** — isotropic Hooke's law, Lamé parameters, 6×6 Voigt stiffness/compliance, orthotropic elasticity
- **Mohr's circle** — 2D and 3D construction, stress transformation, principal stresses
- **Beam mechanics** — Euler-Bernoulli beam: deflection, bending moment, shear force for simply-supported and cantilever beams
- **Finite elements** — 1D bar elements, global stiffness assembly, displacement solution, stress recovery
- **Yield criteria** — von Mises and Tresca
- **Plane stress and plane strain** — 2D stress-strain conversion, out-of-plane response, 3×3 stiffness matrices
- **Energy methods** — strain energy density, Castigliano's theorem, complementary energy

## Install

```toml
[dependencies]
solid-mechanics = "0.1.0"
```

Requires Rust 2021 edition. Depends on `nalgebra` and `serde`.

## Quick Start

### Stress Tensor and Principal Stresses

```rust
use solid_mechanics::tensor::StressTensor;

let sigma = StressTensor::from_components(80.0, -40.0, 20.0, 30.0, 0.0, 0.0);
println!("Hydrostatic stress: {}", sigma.hydrostatic());
println!("Von Mises: {}", sigma.von_mises());
println!("Principal stresses: {:?}", sigma.principal_stresses());
```

### Hooke's Law

```rust
use solid_mechanics::{StressTensor, StrainTensor, HookeIsotropic};

let steel = HookeIsotropic::new(200e9, 0.3);
let stress = StressTensor::uniaxial(100e6);
let strain = steel.strain_from_stress(&stress);
```

### Mohr's Circle

```rust
use solid_mechanics::MohrCircle;

let mc = MohrCircle::from_2d(80.0, -40.0, 30.0);
println!("σ₁ = {}, σ₂ = {}", mc.sigma1, mc.sigma2);
```

### Beam Deflection

```rust
use solid_mechanics::EulerBernoulliBeam;

let beam = EulerBernoulliBeam::new(200e9, 1e-4, 5.0);
println!("Max deflection: {}", beam.max_deflection_centered_point(10000.0));
```

### 1D Finite Element Analysis

```rust
use solid_mechanics::fem::{BarElement1D, FemAssembler1D};

let fem = FemAssembler1D::uniform_bar(200e9, 1e-4, 5.0, 10);
let mut forces = vec![0.0; 11];
forces[10] = 50000.0;
let u = fem.solve(&forces, &[0]).unwrap();
```

### Yield Criteria

```rust
use solid_mechanics::{VonMises, Tresca, StressTensor};

let vm = VonMises::new(250e6);
let stress = StressTensor::from_components(200e6, 100e6, -50e6, 60e6, 0.0, 0.0);
println!("Von Mises equivalent: {}", VonMises::equivalent_stress(&stress));
println!("Safety factor: {}", vm.safety_factor(&stress));
```

## License

MIT OR Apache-2.0
