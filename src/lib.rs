//! # solid-mechanics
//!
//! Solid mechanics library implementing stress, strain, and deformation of solids.
//!
//! ## Modules
//! - **tensor**: Stress and strain tensors (Cauchy stress, infinitesimal strain)
//! - **constitutive**: Constitutive relations (Hooke's law, isotropic/anisotropic elasticity)
//! - **mohr**: Mohr's circle (principal stresses, maximum shear)
//! - **beam**: Beam mechanics (Euler-Bernoulli beam, deflection, bending moments)
//! - **fem**: Finite element basics (1D bar elements, assembly, boundary conditions)
//! - **yield_criteria**: Yield criteria (von Mises, Tresca)
//! - **plane**: Plane stress and plane strain
//! - **energy**: Energy methods (strain energy, Castigliano's theorem)

pub mod tensor;
pub mod constitutive;
pub mod mohr;
pub mod beam;
pub mod fem;
pub mod yield_criteria;
pub mod plane;
pub mod energy;

pub use tensor::{StressTensor, StrainTensor};
pub use constitutive::{HookeIsotropic, ElasticityTensor};
pub use mohr::MohrCircle;
pub use beam::EulerBernoulliBeam;
pub use fem::{BarElement1D, FemAssembler1D};
pub use yield_criteria::{VonMises, Tresca};
pub use plane::{PlaneStress, PlaneStrain};
pub use energy::StrainEnergy;
