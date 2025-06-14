mod ellipse;
mod config;
mod sim_config;

use ellipse::Ellipse;
use config::Config;
use sim_config::SimConfig;
use itertools::iproduct;
use ndarray::{Array1, Array2, Array3};
use numpy::{IntoPyArray, PyArray1, PyArray2, ToPyArray};
use pyo3::prelude::*;
use rand::prelude::SliceRandom;
use rand::Rng;
use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt;

pub struct Cnaster_Graph {
    // NB input on initialization
    pub positions: Array2<f64>, // shape: (num_nodes, 3)
    pub coverage: Array2<f64>,  // shape: (num_nodes, 2)

    // NB assigned by method
    pub labels: Array1<i32>,
    pub adjacency_list: HashMap<usize, Vec<(usize, f64)>>,

    // NB derived.
    pub num_nodes: usize,
    pub max_label: usize,
    pub max_edge_weight: f64,
    pub mean_cov: Vec<f64>,
}

impl fmt::Debug for Cnaster_Graph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let num_edges: usize = self.adjacency_list.values().map(|v| v.len()).sum();
        write!(
            f,
            "Cnaster_Graph {{ num_nodes: {}, num_edges: {}, max_label: {}, mean_coverage: {:?} }}",
            self.num_nodes, num_edges, self.max_label, self.mean_cov
        )
    }
}

impl Cnaster_Graph {
    pub fn new(positions: Array2<f64>, coverage: Array2<f64>, max_label: usize) -> Self {
        let num_nodes = positions.shape()[0];

        assert_eq!(
            positions.shape()[1],
            3,
            "positions must have shape [num_nodes, 3]"
        );

        assert_eq!(
            coverage.nrows(),
            num_nodes,
            "coverage must have same number of rows as positions"
        );

        assert_eq!(
            coverage.ncols(),
            2,
            "coverage must have 2 columns (for 2 labels)"
        );

        let mean_cov: Vec<f64> = (0..coverage.ncols())
            .map(|i| coverage.column(i).mean().unwrap_or(0.0))
            .collect();

        let labels = Array1::<i32>::zeros(num_nodes);

        let mut graph = Cnaster_Graph {
            positions,
            coverage,
            labels,
            adjacency_list: HashMap::new(),
            num_nodes,
            max_label,
            max_edge_weight: 0.0,
            mean_cov,
        };

        graph.reset_node_labels();
        graph
    }

    pub fn reset_node_labels(&mut self) {
        let mut rng = rand::thread_rng();

        for label in self.labels.iter_mut() {
            *label = rng.gen_range(0..=self.max_label as i32);
        }
    }

    pub fn update_adjacency_list(&mut self, edges: Vec<(usize, usize)>, weights: Option<Vec<f64>>) {
        for (i, (from, to)) in edges.iter().enumerate() {
            let weight = if let Some(ref w) = weights {
                w[i]
            } else {
                1.0 // Default weight if not provided
            };

            // Check if edge already exists
            if let Some(neighbors) = self.adjacency_list.get(from) {
                if let Some((_, existing_weight)) = neighbors.iter().find(|(nbr, _)| nbr == to) {
                    if (*existing_weight - weight).abs() > 1e-12 {
                        panic!(
                            "Edge ({},{}) already exists with incompatible weight: {} vs {}",
                            from, to, existing_weight, weight
                        );
                    } else {
                        continue; // Edge exists with same weight, skip adding
                    }
                }
            }

            if weight > self.max_edge_weight {
                self.max_edge_weight = weight;
            }

            self.adjacency_list
                .entry(*from)
                .or_insert_with(Vec::new)
                .push((*to, weight));
        }
    }

    pub fn neighbors(&self, node: &usize) -> Option<&Vec<(usize, f64)>> {
        self.adjacency_list.get(node)
    }

    /// Potts cost: sum over edges of J * (label_i != label_j) * weight_ij
    /// plus sum over nodes of H[node, label_i]
    pub fn potts_energy(&self, J: f64, H: Option<&Array2<f64>>) -> f64 {
        if let Some(H) = H {
            assert_eq!(
                H.nrows(),
                self.num_nodes,
                "H must have shape [num_nodes, 1 + max_label]"
            );

            assert_eq!(
                H.ncols(),
                1 + self.max_label,
                "H must have shape [num_nodes, 1 + max_label]"
            );
        }

        let mut energy = 0.0;
        let mut seen = std::collections::HashSet::new();

        for (node, neighbors) in &self.adjacency_list {
            let label_i = self.labels[*node];

            for (neighbor, weight) in neighbors {
                if seen.contains(&(*neighbor, *node)) {
                    continue;
                }
                let label_j = self.labels[*neighbor];
                if label_i != label_j {
                    energy += J * *weight;
                }
                seen.insert((*node, *neighbor));
            }
        }
        if let Some(H) = H {
            for node in 0..self.num_nodes {
                let label_idx = self.labels[node] as usize;

                energy -= H[[node, label_idx]];
            }
        }

        energy
    }

    pub fn icm_sweep(&mut self, J: f64, epsilon: f64, H: Option<&Array2<f64>>) -> f64 {
        let mut rng = rand::thread_rng();

        for node in 0..self.num_nodes {
            let current_label = self.labels[node];
            let mut best_label = current_label;

            let explore = rng.gen::<f64>() < epsilon;

            if explore {
                let mut new_label = current_label;

                while new_label == current_label {
                    new_label = rng.gen_range(0..=self.max_label as i32);
                }

                best_label = new_label;
            } else {
                let mut best_energy = std::f64::INFINITY;

                for label in 0..=self.max_label as i32 {
                    let mut energy = 0.0;

                    if let Some(neighbors) = self.adjacency_list.get(&node) {
                        for (neighbor, weight) in neighbors {
                            if label != self.labels[*neighbor] {
                                energy += J * *weight;
                            }
                        }
                    }

                    if let Some(H) = H {
                        energy -= H[[node, label as usize]];
                    }

                    if energy < best_energy {
                        best_energy = energy;
                        best_label = label;
                    }
                }
            }

            if best_label != current_label {
                self.labels[node] = best_label;
            }
        }

        self.potts_energy(J, H)
    }

    pub fn icm_learn(&mut self, J: f64, max_iters: usize, H: Option<&Array2<f64>>) -> f64 {
        let mut prev_energy = self.potts_energy(J, H);
        let mut t = 1;
        let mut changed = true;

        while changed && t <= max_iters {
            let epsilon = 1.0 / (t as f64);
            let old_labels = self.labels.clone();

            self.icm_sweep(J, epsilon, H);

            changed = old_labels != self.labels;

            let energy = self.potts_energy(J, H);

            if !changed || (energy - prev_energy).abs() < 1e-12 {
                return energy;
            }

            prev_energy = energy;
            t += 1;
        }

        self.potts_energy(J, H)
    }

    pub fn metropolis_hastings_sweep(&mut self, J: f64, beta: f64, H: Option<&Array2<f64>>) -> f64 {
        let mut rng = rand::thread_rng();
        let mut total_energy = self.potts_energy(J, H);

        for node in 0..self.num_nodes {
            let current_label = self.labels[node];
            let mut proposed_label = current_label;

            while proposed_label == current_label {
                proposed_label = rng.gen_range(0..=self.max_label as i32);
            }

            let mut delta_energy = 0.0;

            if let Some(H) = H {
                delta_energy =
                    H[[node, current_label as usize]] - H[[node, proposed_label as usize]];
            }

            if let Some(neighbors) = self.adjacency_list.get(&node) {
                for (neighbor, weight) in neighbors {
                    if self.labels[*neighbor] == current_label {
                        delta_energy += J * weight;
                    } else if self.labels[*neighbor] == proposed_label {
                        delta_energy -= J * weight;
                    }
                }
            }

            if delta_energy < 0.0 || rng.gen::<f64>() < (-delta_energy * beta).exp() {
                self.labels[node] = proposed_label;

                total_energy += delta_energy;
            }
        }

        total_energy
    }
}

#[pyclass]
#[pyo3(name = "CnasterGraph")]
pub struct pyCnaster_Graph {
    inner: Cnaster_Graph,
}

#[pymethods]
impl pyCnaster_Graph {
    #[new]
    pub fn new(positions: &PyArray2<f64>, coverage: &PyArray2<f64>, max_label: usize) -> Self {
        let positions = positions.readonly().as_array().to_owned();
        let coverage = coverage.readonly().as_array().to_owned();

        let inner = Cnaster_Graph::new(positions, coverage, max_label);

        pyCnaster_Graph { inner }
    }

    pub fn __repr__(&self) -> String {
        let num_edges: usize = self.inner.adjacency_list.values().map(|v| v.len()).sum();

        format!(
            "CnasterGraph(num_nodes={}, num_edges={}, max_label={}, mean_coverage={:?})",
            self.inner.num_nodes, num_edges, self.inner.max_label, self.inner.mean_cov
        )
    }

    #[getter]
    pub fn positions<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {
        self.inner.positions.view().to_pyarray(py)
    }

    #[getter]
    pub fn coverage<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {
        self.inner.coverage.view().to_pyarray(py)
    }

    #[getter]
    pub fn labels<'py>(&self, py: Python<'py>) -> &'py PyArray1<i32> {
        self.inner.labels.view().to_pyarray(py)
    }

    #[getter]
    pub fn adjacency_list(&self) -> HashMap<usize, Vec<(usize, f64)>> {
        self.inner.adjacency_list.clone()
    }

    #[getter]
    pub fn num_nodes(&self) -> usize {
        self.inner.num_nodes
    }

    #[getter]
    pub fn max_label(&self) -> usize {
        self.inner.max_label
    }

    #[getter]
    pub fn max_edge_weight(&self) -> f64 {
        self.inner.max_edge_weight
    }

    #[getter]
    pub fn mean_cov(&self) -> Vec<f64> {
        self.inner.mean_cov.clone()
    }

    pub fn reset_node_labels(&mut self) {
        self.inner.reset_node_labels();
    }

    pub fn update_adjacency_list(
        &mut self,
        edges: &PyArray2<usize>,
        weights: Option<&PyArray1<f64>>,
    ) {
        let binding = edges.readonly();
        let edges = binding.as_array();

        let edges_vec: Vec<(usize, usize)> = edges
            .rows()
            .into_iter()
            .map(|row| (row[0], row[1]))
            .collect();

        let weights_vec = if let Some(weights) = weights {
            Some(weights.readonly().as_array().to_owned().to_vec())
        } else {
            None
        };

        self.inner.update_adjacency_list(edges_vec, weights_vec);
    }

    pub fn neighbors(&self, node: usize) -> Option<Vec<(usize, f64)>> {
        self.inner.neighbors(&node).cloned()
    }

    pub fn potts_energy(&self, J: f64, H: Option<&PyArray2<f64>>) -> f64 {
        let mut h_owned = None;

        let h_ref = if let Some(H) = H {
            h_owned = Some(H.readonly().as_array().to_owned());
            h_owned.as_ref()
        } else {
            None
        };

        self.inner.potts_energy(J, h_ref)
    }

    pub fn icm_sweep(&mut self, J: f64, epsilon: f64, H: Option<&PyArray2<f64>>) -> f64 {
        let mut h_owned = None;
        let h_ref = if let Some(H) = H {
            h_owned = Some(H.readonly().as_array().to_owned());
            h_owned.as_ref()
        } else {
            None
        };

        self.inner.icm_sweep(J, epsilon, h_ref)
    }

    pub fn icm_learn(&mut self, J: f64, max_iters: usize, H: Option<&PyArray2<f64>>) -> f64 {
        let mut h_owned = None;
        let h_ref = if let Some(H) = H {
            h_owned = Some(H.readonly().as_array().to_owned());
            h_owned.as_ref()
        } else {
            None
        };

        self.inner.icm_learn(J, max_iters, h_ref)
    }

    pub fn metropolis_hastings_sweep(
        &mut self,
        J: f64,
        beta: f64,
        H: Option<&PyArray2<f64>>,
    ) -> f64 {
        let mut h_owned = None;
        let h_ref = if let Some(H) = H {
            h_owned = Some(H.readonly().as_array().to_owned());
            h_owned.as_ref()
        } else {
            None
        };

        self.inner.metropolis_hastings_sweep(J, beta, h_ref)
    }
}

pub fn get_triangular_lattice(nx: usize, ny: usize, x0: Vec<f64>, z: f64) -> Array2<f64> {
    //  NB  triangular lattice may be indexed as 2D with two additional edges.
    //      Positions are sheared at 60 degrees to create the triangular structure.
    let mut positions = Array2::<f64>::zeros((nx * ny, 3));

    //  NB  Bravais lattice vectors for triangular lattice:
    let a1 = vec![1.0, 0.0];

    //  NB  a2 is sheared at 30 degrees clockwise from square lattice.
    let a2 = vec![0.5, (3.0_f64).sqrt() / 2.0]; // 60 degrees shear

    for i in 0..ny {
        for j in 0..nx {
            let x = x0[0] + (j as f64) * a1[0] + (i as f64) * a2[0];
            let y = x0[1] + (j as f64) * a1[1] + (i as f64) * a2[1];

            positions[[i * nx + j, 0]] = x;
            positions[[i * nx + j, 1]] = y;
            positions[[i * nx + j, 2]] = z;
        }
    }

    positions
}

pub fn get_triangular_lattice_edges(nx: usize, ny: usize) -> Vec<(usize, usize)> {
    let mut edges = Vec::new();

    for y in 0..ny {
        for x in 0..nx {
            let idx = y * nx + x;

            // right neighbor, (i+1, j)
            if x + 1 < nx {
                let nbr = y * nx + (x + 1);
                edges.push((idx, nbr));
            }
            // left neighbor, (i-1, j)
            if x >= 1 {
                let nbr = y * nx + (x - 1);
                edges.push((idx, nbr));
            }
            // top neighbor, (i, j+1)
            if y + 1 < ny {
                let nbr = (y + 1) * nx + x;
                edges.push((idx, nbr));
            }
            // bottom neighbor, (i, j-1)
            if y >= 1 {
                let nbr = (y - 1) * nx + x;
                edges.push((idx, nbr));
            }
            // triangular specific, (i+1, j-1)
            if x + 1 < nx && y >= 1 {
                let nbr = (y - 1) * nx + (x + 1);
                edges.push((idx, nbr));
            }
            // triangular specific, (i-1, j+1)
            if x >= 1 && y + 1 < ny {
                let nbr = (y + 1) * nx + (x - 1);
                edges.push((idx, nbr));
            }
        }
    }

    edges
}

pub fn get_slices_triangular_lattice_edges(
    positions_slices: &Vec<Array2<f64>>,
    nxy_vec: &Vec<(usize, usize)>,
) -> Vec<(usize, usize)> {
    let mut edges = Vec::new();

    let mut idx_bases: Vec<usize> = Vec::with_capacity(positions_slices.len());
    let mut acc = 0;

    for slice in positions_slices {
        idx_bases.push(acc);
        acc += slice.nrows();
    }

    for ((slice, &(nx, ny)), &idx_base) in
        positions_slices.iter().zip(nxy_vec).zip(idx_bases.iter())
    {
        let local_edges = get_triangular_lattice_edges(nx, ny);

        for (from, to) in local_edges {
            edges.push((idx_base + from, idx_base + to));
        }
    }

    edges
}

pub fn nearest_neighbor_edges(
    positions_slices: &Vec<Array2<f64>>,
    n_closest: usize,
    max_distance: Option<f64>,
) -> (Array2<usize>, Array1<f64>) {
    let nslice = positions_slices.len();
    if nslice == 0 {
        return (Array2::<usize>::zeros((0, 2)), Array1::<f64>::zeros(0));
    }

    let mut from_to = Vec::new();
    let mut weights = Vec::new();

    // NB idx by cumulative node number by slice, in slice order.
    let idx_bases: Vec<usize> = positions_slices
        .iter()
        .scan(0, |acc, arr| {
            let base = *acc;
            *acc += arr.nrows();
            Some(base)
        })
        .collect();

    for z in 0..nslice {
        let this_slice = &positions_slices[z];
        let nthis_slice = this_slice.nrows();

        for &dz in &[-1isize, 1] {
            let z_adj = z as isize + dz;
            if z_adj < 0 || z_adj >= nslice as isize {
                continue;
            }
            let adj_slice = &positions_slices[z_adj as usize];
            let nadj_slice = adj_slice.nrows();

            for i in 0..nthis_slice {
                let pos_i = this_slice.row(i);

                let mut dists: Vec<(usize, f64)> = (0..nadj_slice)
                    .map(|j| {
                        let pos_j = adj_slice.row(j);
                        let dist = ((pos_i[0] - pos_j[0]).powi(2)
                            + (pos_i[1] - pos_j[1]).powi(2)
                            + (pos_i[2] - pos_j[2]).powi(2))
                        .sqrt();
                        (j, dist)
                    })
                    .collect();

                dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

                let mut count = 0;
                for &(j, dist) in dists.iter() {
                    if let Some(max_d) = max_distance {
                        if dist > max_d {
                            continue;
                        }
                    }
                    from_to.push([idx_bases[z] + i, idx_bases[z_adj as usize] + j]);
                    weights.push(dist);
                    count += 1;
                    if count >= n_closest {
                        break;
                    }
                }
            }
        }
    }

    let num_edges = from_to.len();
    let from_to_arr = if num_edges > 0 {
        Array2::from_shape_vec((num_edges, 2), from_to.into_iter().flatten().collect()).unwrap()
    } else {
        Array2::<usize>::zeros((0, 2))
    };
    let weights_arr = Array1::from(weights);

    (from_to_arr, weights_arr)
}

#[pyfunction]
#[pyo3(name = "get_triangular_lattice")]
fn py_get_triangular_lattice<'py>(
    py: Python<'py>,
    nx: usize,
    ny: usize,
    z: f64,
    x0: Option<(f64, f64)>,
) -> PyResult<&'py PyArray2<f64>> {
    let x0 = x0.unwrap_or((0.0, 0.0));
    let positions = get_triangular_lattice(nx, ny, vec![x0.0, x0.1], z);

    Ok(positions.into_pyarray(py))
}

#[pyfunction]
#[pyo3(name = "get_slices_triangular_lattice_edges")]
fn py_get_slices_triangular_lattice_edges<'py>(
    py: Python<'py>,
    positions_slices: Vec<&PyArray2<f64>>,
    nxy_vec: &PyArray2<usize>, // shape: (num_slices, 2)
) -> PyResult<&'py PyArray2<usize>> {
    let positions_vec: Vec<Array2<f64>> = positions_slices
        .iter()
        .map(|arr| arr.readonly().as_array().to_owned())
        .collect();

    let nxy_vec: Vec<(usize, usize)> = nxy_vec
        .readonly()
        .as_array()
        .rows()
        .into_iter()
        .map(|row| (row[0], row[1]))
        .collect();

    let edges = get_slices_triangular_lattice_edges(&positions_vec, &nxy_vec);

    let num_edges = edges.len();
    let mut edges_arr = Array2::<usize>::zeros((num_edges, 2));

    for (i, (from, to)) in edges.iter().enumerate() {
        edges_arr[[i, 0]] = *from;
        edges_arr[[i, 1]] = *to;
    }

    Ok(edges_arr.into_pyarray(py))
}

#[pyfunction]
#[pyo3(name = "nearest_neighbor_edges")]
fn py_nearest_neighbor_edges<'py>(
    py: Python<'py>,
    positions_slices: Vec<&PyArray2<f64>>,
    n_closest: usize,
    max_distance: Option<f64>,
) -> PyResult<(&'py PyArray2<usize>, &'py PyArray1<f64>)> {
    let positions_vec: Vec<Array2<f64>> = positions_slices
        .iter()
        .map(|arr| arr.readonly().as_array().to_owned())
        .collect();

    let (from_to, weights) = nearest_neighbor_edges(&positions_vec, n_closest, max_distance);

    Ok((from_to.into_pyarray(py), weights.into_pyarray(py)))
}

#[pymodule]
#[pyo3(name = "cnaster_rs")]
fn cnaster_rs(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_get_triangular_lattice, m)?)?;
    m.add_function(wrap_pyfunction!(py_get_slices_triangular_lattice_edges, m)?)?;
    m.add_function(wrap_pyfunction!(py_nearest_neighbor_edges, m)?)?;
    m.add_class::<pyCnaster_Graph>()?;
    // m.add_class::<pyEllipse>()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triangular_lattice_positions_large() {
        let nx = 52;
        let ny = 96;

        assert_eq!(nx, ny);

        let x0 = vec![0.0, 0.0];
        let z = 2.5;

        let positions = get_triangular_lattice(nx, ny, x0, z);

        println!("Positions: {:?}", positions);

        assert_eq!(positions.shape(), &[nx * ny, 3]);
        assert_eq!(positions.shape(), &[4992, 3]);

        assert!(positions.column(2).iter().all(|&zi| (zi - z).abs() < 1e-12));

        assert!((positions[[0, 0]] - 0.0).abs() < 1e-12);
        assert!((positions[[0, 1]] - 0.0).abs() < 1e-12);

        assert!((positions[[1, 0]] - 1.0).abs() < 1e-12);
        assert!((positions[[1, 1]] - 0.0).abs() < 1e-12);

        let idx = 1 * nx + 0;

        assert!((positions[[idx, 0]] - 0.5).abs() < 1e-12);
        assert!((positions[[idx, 1]] - ((3.0_f64).sqrt() / 2.0)).abs() < 1e-12);
    }
}
