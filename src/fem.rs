//! Finite element basics — 1D bar elements, assembly, boundary conditions.

use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

/// A 1D two-node bar element with constant cross-section.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BarElement1D {
    /// Young's modulus (Pa)
    pub e: f64,
    /// Cross-sectional area (m²)
    pub a: f64,
    /// Element length (m)
    pub length: f64,
    /// Global node indices [node_i, node_j]
    pub nodes: [usize; 2],
}

impl BarElement1D {
    /// Create a new 1D bar element.
    pub fn new(e: f64, a: f64, length: f64, node_i: usize, node_j: usize) -> Self {
        Self { e, a, length, nodes: [node_i, node_j] }
    }

    /// Element stiffness: k = EA/L
    pub fn stiffness(&self) -> f64 {
        self.e * self.a / self.length
    }

    /// Local 2×2 element stiffness matrix.
    pub fn local_stiffness_matrix(&self) -> [[f64; 2]; 2] {
        let k = self.stiffness();
        [[k, -k], [-k, k]]
    }

    /// Axial stress in the element given nodal displacements.
    pub fn axial_stress(&self, u_i: f64, u_j: f64) -> f64 {
        self.e * (u_j - u_i) / self.length
    }

    /// Axial force in the element.
    pub fn axial_force(&self, u_i: f64, u_j: f64) -> f64 {
        self.stiffness() * (u_j - u_i)
    }

    /// Strain energy in the element.
    pub fn strain_energy(&self, u_i: f64, u_j: f64) -> f64 {
        let du = u_j - u_i;
        0.5 * self.stiffness() * du * du
    }
}

/// 1D FEM assembler for bar elements.
#[derive(Debug, Clone)]
pub struct FemAssembler1D {
    /// Number of nodes
    pub num_nodes: usize,
    /// Elements
    pub elements: Vec<BarElement1D>,
}

impl FemAssembler1D {
    /// Create a new assembler with a given number of nodes.
    pub fn new(num_nodes: usize) -> Self {
        Self { num_nodes, elements: Vec::new() }
    }

    /// Add an element.
    pub fn add_element(&mut self, element: BarElement1D) {
        self.elements.push(element);
    }

    /// Create a uniform bar discretized into n elements.
    pub fn uniform_bar(e: f64, a: f64, total_length: f64, n_elements: usize) -> Self {
        let n_nodes = n_elements + 1;
        let le = total_length / n_elements as f64;
        let mut assembler = Self::new(n_nodes);
        for i in 0..n_elements {
            assembler.add_element(BarElement1D::new(e, a, le, i, i + 1));
        }
        assembler
    }

    /// Assemble the global stiffness matrix.
    pub fn assemble_global_stiffness(&self) -> DMatrix<f64> {
        let n = self.num_nodes;
        let mut k_global = DMatrix::zeros(n, n);

        for elem in &self.elements {
            let ke = elem.local_stiffness_matrix();
            let ni = elem.nodes[0];
            let nj = elem.nodes[1];

            k_global[(ni, ni)] += ke[0][0];
            k_global[(ni, nj)] += ke[0][1];
            k_global[(nj, ni)] += ke[1][0];
            k_global[(nj, nj)] += ke[1][1];
        }

        k_global
    }

    /// Apply boundary conditions by removing fixed DOFs.
    /// `fixed_nodes` contains indices of nodes with zero displacement.
    /// `forces` is the full force vector.
    /// Returns (reduced stiffness, reduced force vector, mapping from free DOF to global DOF).
    pub fn apply_boundary_conditions(
        &self,
        k_global: &DMatrix<f64>,
        forces: &[f64],
        fixed_nodes: &[usize],
    ) -> (DMatrix<f64>, Vec<f64>, Vec<usize>) {
        let fixed_set: std::collections::HashSet<usize> = fixed_nodes.iter().cloned().collect();
        let free_dofs: Vec<usize> = (0..self.num_nodes)
            .filter(|i| !fixed_set.contains(i))
            .collect();

        let nf = free_dofs.len();
        let mut k_reduced = DMatrix::zeros(nf, nf);
        let mut f_reduced = vec![0.0; nf];

        for (ir, &i) in free_dofs.iter().enumerate() {
            f_reduced[ir] = forces[i];
            for (jc, &j) in free_dofs.iter().enumerate() {
                k_reduced[(ir, jc)] = k_global[(i, j)];
            }
        }

        (k_reduced, f_reduced, free_dofs)
    }

    /// Solve the FEM system for displacements.
    /// Returns the full displacement vector (including zeros at fixed nodes).
    pub fn solve(
        &self,
        forces: &[f64],
        fixed_nodes: &[usize],
    ) -> Option<Vec<f64>> {
        let k_global = self.assemble_global_stiffness();
        let (k_red, f_red, free_dofs) = self.apply_boundary_conditions(&k_global, forces, fixed_nodes);

        // Solve K_red * u_red = f_red
        let u_red = k_red.lu().solve(&nalgebra::DVector::from_vec(f_red))?;

        // Reconstruct full displacement vector
        let mut u_full = vec![0.0; self.num_nodes];
        for (i, &dof) in free_dofs.iter().enumerate() {
            u_full[dof] = u_red[i];
        }

        Some(u_full)
    }

    /// Compute element stresses from the full displacement vector.
    pub fn element_stresses(&self, displacements: &[f64]) -> Vec<f64> {
        self.elements.iter().map(|elem| {
            let ui = displacements[elem.nodes[0]];
            let uj = displacements[elem.nodes[1]];
            elem.axial_stress(ui, uj)
        }).collect()
    }

    /// Compute total strain energy.
    pub fn total_strain_energy(&self, displacements: &[f64]) -> f64 {
        self.elements.iter().map(|elem| {
            let ui = displacements[elem.nodes[0]];
            let uj = displacements[elem.nodes[1]];
            elem.strain_energy(ui, uj)
        }).sum()
    }

    /// Verify equilibrium: K * u = F (external force check).
    pub fn check_equilibrium(
        &self,
        displacements: &[f64],
        forces: &[f64],
    ) -> f64 {
        let k = self.assemble_global_stiffness();
        let u = nalgebra::DVector::from_vec(displacements.to_vec());
        let f = nalgebra::DVector::from_vec(forces.to_vec());
        (k * u - f).norm()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_bar_element_stiffness() {
        let bar = BarElement1D::new(200e9, 1e-4, 1.0, 0, 1);
        assert_relative_eq!(bar.stiffness(), 2e7);
    }

    #[test]
    fn test_local_stiffness_symmetry() {
        let bar = BarElement1D::new(200e9, 1e-4, 1.0, 0, 1);
        let k = bar.local_stiffness_matrix();
        assert_relative_eq!(k[0][1], -k[0][0]);
        assert_relative_eq!(k[1][0], -k[1][1]);
    }

    #[test]
    fn test_single_bar_fixed_free() {
        let bar = BarElement1D::new(200e9, 1e-4, 1.0, 0, 1);
        let mut fem = FemAssembler1D::new(2);
        fem.add_element(bar);
        let forces = vec![0.0, 10000.0];
        let fixed = vec![0];
        let u = fem.solve(&forces, &fixed).unwrap();
        // u1 = FL/(EA) = 10000 / (200e9 * 1e-4) = 5e-4
        assert_relative_eq!(u[1], 5e-4, epsilon = 1e-8);
    }

    #[test]
    fn test_uniform_bar_convergence() {
        // Fixed-free bar with end load: analytical u = FL/(EA)
        let e = 200e9;
        let a = 1e-4;
        let l = 5.0;
        let f = 50000.0;
        let analytical = f * l / (e * a);

        // Convergence: more elements → same result
        for n in [1, 5, 10, 50] {
            let fem = FemAssembler1D::uniform_bar(e, a, l, n);
            let mut forces = vec![0.0; n + 1];
            forces[n] = f;
            let u = fem.solve(&forces, &[0]).unwrap();
            assert_relative_eq!(u[n], analytical, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_stress_uniform_in_uniform_bar() {
        let e = 200e9;
        let a = 1e-4;
        let l = 5.0;
        let f = 50000.0;
        let fem = FemAssembler1D::uniform_bar(e, a, l, 5);
        let mut forces = vec![0.0; 6];
        forces[5] = f;
        let u = fem.solve(&forces, &[0]).unwrap();
        let stresses = fem.element_stresses(&u);
        // All elements should have same stress: σ = F/A
        let expected_stress = f / a;
        for &s in &stresses {
            assert_relative_eq!(s, expected_stress, epsilon = 1e-3);
        }
    }

    #[test]
    fn test_strain_energy_conservation() {
        let e = 200e9;
        let a = 1e-4;
        let l = 3.0;
        let f = 30000.0;
        let fem = FemAssembler1D::uniform_bar(e, a, l, 10);
        let mut forces = vec![0.0; 11];
        forces[10] = f;
        let u = fem.solve(&forces, &[0]).unwrap();
        let se = fem.total_strain_energy(&u);
        // Strain energy = 0.5 * F * δ = 0.5 * F * FL/(EA) = F²L/(2EA)
        let expected = f * f * l / (2.0 * e * a);
        assert_relative_eq!(se, expected, epsilon = 1e-2);
    }

    #[test]
    fn test_equilibrium_check() {
        let e = 200e9;
        let a = 1e-4;
        let l = 2.0;
        let fem = FemAssembler1D::uniform_bar(e, a, l, 4);
        let f_end = 10000.0;
        let mut forces = vec![0.0; 5];
        forces[4] = f_end;
        let u = fem.solve(&forces, &[0]).unwrap();

        // Verify K*u = F at free DOFs only (skip fixed DOF 0)
        let k = fem.assemble_global_stiffness();
        for i in 1..5 {
            let mut ku_i = 0.0_f64;
            for j in 0..5 {
                ku_i += k[(i, j)] * u[j];
            }
            assert_relative_eq!(ku_i, forces[i], epsilon = 0.1);
        }
    }
}
