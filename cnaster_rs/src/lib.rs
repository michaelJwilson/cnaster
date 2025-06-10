use pyo3::prelude::*;
use std::collections::HashMap;
use ndarray::{Array1, Array2, Array3};
use rand::Rng;

/*
pub struct Graph<W> {
    pub adjacency_list: HashMap<usize, Vec<(usize, W)>>,
    pub positions: Array3<f64>, // shape: (num_nodes, 3)
    pub labels: Array1<i32>,
    pub coverage: Array2<f64>, // shape: (num_nodes, 2)

    // NB derived.
    pub num_nodes: usize,
    pub num_unique_labels: usize,
}

impl<W: Clone> Graph<W> {
    /// Create a new graph with given positions (shape: [num_nodes, 3]) and coverage (shape: [num_nodes, 2]).
    /// Initial labels are set according to the coverage array: label = argmax(coverage[i, :])
    pub fn new(positions: Array3<f64>, coverage: Array2<f64>) -> Self {
        let num_nodes = positions.shape()[0];
        assert_eq!(positions.shape()[1], 3, "positions must have shape [num_nodes, 3]");
        assert_eq!(coverage.nrows(), num_nodes, "coverage must have same number of rows as positions");
        assert_eq!(coverage.ncols(), 2, "coverage must have 2 columns (for 2 labels)");

        // Assign label as argmax of coverage for each node
        let labels = Array1::from(
            (0..num_nodes)
                .map(|i| {
                    if coverage[[i, 0]] >= coverage[[i, 1]] {
                        0
                    } else {
                        1
                    }
                })
                .collect::<Vec<_>>()
        );

        Graph {
            adjacency_list: HashMap::new(),
            positions,
            labels,
            coverage,
            num_nodes,
            num_unique_labels: 2,
        }
    }

    pub fn add_edge(&mut self, from: usize, to: usize, weight: W) {
        self.adjacency_list
            .entry(from)
            .or_insert_with(Vec::new)
            .push((to, weight.clone()));

        self.adjacency_list
            .entry(to)
            .or_insert_with(Vec::new)
            .push((from, weight));
    }

    pub fn neighbors(&self, node: &usize) -> Option<&Vec<(usize, W)>> {
        self.adjacency_list.get(node)
    }

    /// Potts cost: sum over edges of J * (label_i != label_j) * weight_ij
    /// plus sum over nodes of H[node, label_i]
    pub fn potts_cost(&self, J: f64, H: &ndarray::Array2<f64>) -> f64 {
        assert_eq!(H.nrows(), self.num_nodes, "H must have shape [num_nodes, num_unique_labels]");

        let mut cost = 0.0;
        let mut seen = std::collections::HashSet::new();

        // Edge costs
        for (node, neighbors) in &self.adjacency_list {
            let label_i = self.labels[*node];

            for (neighbor, weight) in neighbors {
                if seen.contains(&(*neighbor, *node)) {
                    continue;
                }

                let label_j = self.labels[*neighbor];

                if label_i != label_j {
                    cost += J * (*weight).clone().into();
                }

                seen.insert((*node, *neighbor));
            }
        }

        // External field cost: H[node, label_i]
        for node in 0..self.num_nodes {
            let label_idx = self.labels[node] as usize;
            cost += H[[node, label_idx]];
        }

        cost
    }
}
*/

pub fn get_triangular_lattice(nx: usize, ny: usize, z: f64) -> Array3<f64> {
    //  NB  triangular lattice may be indexed as 2D with two additional edges.
    //      Positions are sheared at 60 degrees to create the triangular structure.
    let mut positions = Array3::<f64>::zeros((nx * ny, 3, 1));

    //  NB  Bravais lattice vectors for triangular lattice:
    let a1 = vec![1.0, 0.0];
    let a2 = vec![0.5, (3.0_f64).sqrt() / 2.0]; // 60 degrees shear

    let x0 = vec![0.0, 0.0];

    for i in 0..ny {
        for j in 0..nx {
            let x = x0[0] + (j as f64) * a1[0] + (i as f64) * a2[0];
            let y = x0[1] + (j as f64) * a1[1] + (i as f64) * a2[1];

            positions[[i * nx + j, 0, 0]] = x;
            positions[[i * nx + j, 1, 0]] = y;
            positions[[i * nx + j, 2, 0]] = z;
        }
    }

    positions
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triangular_lattice_positions() {
        let nx = 4;
        let ny = 3;
        let z = 2.5;
        let positions = get_triangular_lattice(nx, ny, z);

        // Check shape
        assert_eq!(positions.shape(), &[nx * ny, 3, 1]);

        // Check z values
        for i in 0..(nx * ny) {
            assert!((positions[[i, 2, 0]] - z).abs() < 1e-12);
        }

        // Check some known positions
        // (0,0)
        assert!((positions[[0, 0, 0]] - 0.0).abs() < 1e-12);
        assert!((positions[[0, 1, 0]] - 0.0).abs() < 1e-12);
        // (1,0)
        assert!((positions[[1, 0, 0]] - 1.0).abs() < 1e-12);
        assert!((positions[[1, 1, 0]] - 0.0).abs() < 1e-12);
        // (0,1)
        let idx = 1 * nx + 0;
        assert!((positions[[idx, 0, 0]] - 0.5).abs() < 1e-12);
        assert!((positions[[idx, 1, 0]] - ((3.0_f64).sqrt() / 2.0)).abs() < 1e-12);
    }
}