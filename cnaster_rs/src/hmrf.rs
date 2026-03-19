use plotters::prelude::*;
use rand::distributions::WeightedIndex;
use rand::prelude::Distribution;
use rand::{rngs::StdRng, Rng, SeedableRng};
use rayon::prelude::*;
use std::collections::VecDeque;

pub struct HMRF {
    pub labels: Vec<usize>,
    pub adj_list: Vec<Vec<(usize, f64)>>,
    pub h_field: Vec<Vec<f64>>,
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub num_colors: usize,
    pub width: usize,
    pub height: usize,
    pub min_h: f64,
}

impl HMRF {
    /// Calculate the proportions of each label (color) currently on the grid.
    pub fn clone_proportions(&self) -> Vec<f64> {
        let mut counts = vec![0; self.num_colors];
        for &label in &self.labels {
            counts[label] += 1;
        }
        let total = self.labels.len() as f64;
        counts.into_iter().map(|c| c as f64 / total).collect()
    }

    pub fn potts_energy(&self, beta: f64) -> f64 {
        let mut energy = 0.0;

        // External field contribution: -H_{i, c_i}
        for (i, &c_i) in self.labels.iter().enumerate() {
            energy += self.h_field[i][c_i];
        }

        // Pairwise interaction contribution: -J_{ij} delta(c_i, c_j)
        // We assume adj_list contains symmetric directed edges (i->j and j->i).
        // To avoid doubling, we only add when i < j, or divide by 2.
        let mut interaction_energy = 0.0;
        for (i, &c_i) in self.labels.iter().enumerate() {
            for &(j, edge_weight) in &self.adj_list[i] {
                if i < j && c_i != self.labels[j] {
                    interaction_energy += edge_weight;
                }
            }
        }

        energy += interaction_energy;
        beta * energy
    }

    /// Iterated Conditional Modes (ICM) to find a local minimum of the MRF energy.
    pub fn icm(&mut self, beta: f64, max_iters: usize) {
        if beta == 0.0 {
            let mut rng = rand::thread_rng();
            for label in &mut self.labels {
                *label = rng.gen_range(0..self.num_colors);
            }
            return;
        }

        for _ in 0..max_iters {
            let mut changed = false;

            for i in 0..self.labels.len() {
                let old_c = self.labels[i];
                let mut best_c = old_c;
                let mut min_cost = f64::INFINITY;

                for c in 0..self.num_colors {
                    // Cost contribution from external field
                    let mut local_cost = self.h_field[i][c];

                    // Pairwise interaction contribution
                    for &(j, weight) in &self.adj_list[i] {
                        if c != self.labels[j] {
                            local_cost += weight;
                        }
                    }

                    local_cost *= beta;

                    // NB defauls to last if equal, which is allowable for symmetry breaking.
                    if local_cost < min_cost {
                        min_cost = local_cost;
                        best_c = c;
                    }
                }

                if best_c != old_c {
                    self.labels[i] = best_c;
                    changed = true;
                }
            }

            if !changed {
                break;
            }
        }
    }

    /// Gibbs sampling to sample from the MRF distribution and optionally find low energy states.
    pub fn gibbs_sample(&mut self, beta: f64, max_iters: usize) {
        let mut rng = rand::thread_rng();

        for _ in 0..max_iters {
            for i in 0..self.labels.len() {
                let mut local_energies = vec![0.0; self.num_colors];
                let mut min_energy = f64::INFINITY;

                for c in 0..self.num_colors {
                    // Energy contribution from external field
                    let mut local_energy = self.h_field[i][c];

                    // Pairwise interaction energy
                    for &(j, weight) in &self.adj_list[i] {
                        if c != self.labels[j] {
                            local_energy += weight;
                        }
                    }

                    local_energies[c] = local_energy;
                    if local_energy < min_energy {
                        min_energy = local_energy;
                    }
                }

                // Compute probabilities: P \propto exp(-beta * energy)
                // avoiding overflow by subtracting min_energy
                let mut probs = vec![0.0; self.num_colors];
                for c in 0..self.num_colors {
                    probs[c] = (-beta * (local_energies[c] - min_energy)).exp();
                }

                // Sample a new color based on the computed probabilities using WeightedIndex
                let dist = WeightedIndex::new(&probs).unwrap();
                self.labels[i] = dist.sample(&mut rng);
            }
        }
    }

    /// Wolff cluster sampling algorithm with Metropolis acceptance for the external field.
    pub fn wolff_sample(&mut self, beta: f64, num_clusters: usize) {
        let mut rng = rand::thread_rng();
        let n_nodes = self.labels.len();

        for _ in 0..num_clusters {
            // Sample a node at random
            let start_node = rng.gen_range(0..n_nodes);
            let m_prime = self.labels[start_node];

            // Select a new color m uniformly from the (num_colors - 1) other choices
            let mut m = rng.gen_range(0..(self.num_colors - 1));
            if m >= m_prime {
                m += 1;
            }

            let mut cluster = Vec::new();
            let mut queue = VecDeque::new();
            let mut in_cluster = vec![false; n_nodes];

            cluster.push(start_node);
            queue.push_back(start_node);
            in_cluster[start_node] = true;

            // Construct BFS from that node for all nodes with the same label
            while let Some(current) = queue.pop_front() {
                for &(neighbor, j_weight) in &self.adj_list[current] {
                    if !in_cluster[neighbor] && self.labels[neighbor] == m_prime {
                        // Sample according to edge probability P(bond) = 1 - exp(-beta * J)
                        let p_bond = 1.0 - (-beta * j_weight).exp();
                        if rng.gen::<f64>() < p_bond {
                            in_cluster[neighbor] = true;
                            cluster.push(neighbor);
                            queue.push_back(neighbor);
                        }
                    }
                }
            }

            // Calculate dH = sum of (H_k,m - H_k,m') \propto energy for nodes C'
            let mut dH = 0.0;
            for &node in &cluster {
                dH += self.h_field[node][m] - self.h_field[node][m_prime];
            }

            // Accept with min(1, exp(-beta * dH))
            let acceptance_prob = if dH <= 0.0 { 1.0 } else { (-beta * dH).exp() };

            if rng.gen::<f64>() < acceptance_prob {
                for &node in &cluster {
                    self.labels[node] = m;
                }
            }
        }
    }

    /// Calculate the marginal probabilities of each label using the mean-field approximation.
    pub fn mean_field(&self, beta: f64, max_iters: usize, tol: f64) -> Vec<Vec<f64>> {
        let n_nodes = self.labels.len();
        let mut q = vec![vec![0.0; self.num_colors]; n_nodes];

        // Initialize from current labels to break symmetry (soft one-hot encoding)
        for i in 0..n_nodes {
            for c in 0..self.num_colors {
                q[i][c] = if self.labels[i] == c {
                    0.99
                } else {
                    0.01 / (self.num_colors as f64 - 1.0).max(1.0)
                };
            }
        }

        for _ in 0..max_iters {
            let mut max_diff = 0.0_f64;

            // Asynchronous updates (Gauss-Seidel) for better convergence
            for i in 0..n_nodes {
                let mut local_energies = vec![0.0; self.num_colors];
                let mut min_energy = f64::INFINITY;

                for c in 0..self.num_colors {
                    let mut expected_energy = self.h_field[i][c];

                    for &(j, weight) in &self.adj_list[i] {
                        // Expected interaction penalty is weight * probability neighbor is NOT c
                        expected_energy += weight * (1.0 - q[j][c]);
                    }

                    local_energies[c] = expected_energy;
                    if expected_energy < min_energy {
                        min_energy = expected_energy;
                    }
                }

                let mut sum_p = 0.0;
                let mut new_qi = vec![0.0; self.num_colors];
                for c in 0..self.num_colors {
                    let p = (-beta * (local_energies[c] - min_energy)).exp();
                    new_qi[c] = p;
                    sum_p += p;
                }

                for c in 0..self.num_colors {
                    new_qi[c] /= sum_p;
                    let diff = (new_qi[c] - q[i][c]).abs();
                    if diff > max_diff {
                        max_diff = diff;
                    }
                    q[i][c] = new_qi[c];
                }
            }

            if max_diff < tol {
                break;
            }
        }

        q
    }

    /// Decode the labels by taking the maximum marginal probability from the mean-field approximation.
    pub fn mean_field_decode(&mut self, beta: f64, max_iters: usize, tol: f64) {
        let q = self.mean_field(beta, max_iters, tol);

        for (i, marginals) in q.iter().enumerate() {
            let mut best_c = 0;
            let mut max_q = -1.0;

            for (c, &prob) in marginals.iter().enumerate() {
                if prob > max_q {
                    max_q = prob;
                    best_c = c;
                }
            }

            self.labels[i] = best_c;
        }
    }

    /// Serial Swendsen-Wang cluster sampling algorithm using Union-Find.
    /// Parallelism is omitted as disjoint-set operations are sequential and 
    /// intermediate bond allocations cause unacceptable overhead.
    pub fn swendsen_wang(&mut self, beta: f64, max_iters: usize) {
        let n_nodes = self.labels.len();
        let mut rng = rand::thread_rng();

        for _ in 0..max_iters {
            // 1. Initialize Union-Find
            let mut parent: Vec<usize> = (0..n_nodes).collect();
            let mut rank = vec![0; n_nodes];

            let mut find = |mut i: usize, parent: &mut [usize]| -> usize {
                let mut root = i;
                while root != parent[root] {
                    root = parent[root];
                }
                while i != root {
                    let nxt = parent[i];
                    parent[i] = root; // path compression
                    i = nxt;
                }
                root
            };

            // 2. Evaluate bonds and union sequentially
            for i in 0..n_nodes {
                let c_i = self.labels[i];
                for &(j, weight) in &self.adj_list[i] {
                    if i < j && self.labels[j] == c_i {
                        let p_bond = 1.0 - (-beta * weight).exp();
                        if rng.gen::<f64>() < p_bond {
                            let root_i = find(i, &mut parent);
                            let root_j = find(j, &mut parent);
                            if root_i != root_j {
                                if rank[root_i] < rank[root_j] {
                                    parent[root_i] = root_j;
                                } else if rank[root_i] > rank[root_j] {
                                    parent[root_j] = root_i;
                                } else {
                                    parent[root_j] = root_i;
                                    rank[root_i] += 1;
                                }
                            }
                        }
                    }
                }
            }

            // 3. Cluster formation
            let mut clusters: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
            for i in 0..n_nodes {
                let root = find(i, &mut parent);
                clusters.entry(root).or_default().push(i);
            }

            // 4. Cluster sampling based on external field
            for cluster in clusters.values() {
                let mut cluster_energies = vec![0.0; self.num_colors];
                let mut min_energy = f64::INFINITY;

                for c in 0..self.num_colors {
                    let mut energy = 0.0;
                    for &node in cluster {
                        energy += self.h_field[node][c];
                    }
                    cluster_energies[c] = energy;
                    if energy < min_energy {
                        min_energy = energy;
                    }
                }

                let mut probs = vec![0.0; self.num_colors];
                for c in 0..self.num_colors {
                    probs[c] = (-beta * (cluster_energies[c] - min_energy)).exp();
                }

                let dist = WeightedIndex::new(&probs).unwrap();
                let new_color = dist.sample(&mut rng);

                for &node in cluster {
                    self.labels[node] = new_color;
                }
            }
        }
    }

    pub fn plot_labels(&self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        let root = SVGBackend::new(filename, (800, 800)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .build_cartesian_2d(0f64..(self.width as f64), 0f64..(self.height as f64))?;

        let colors = [&RED, &BLUE, &GREEN, &YELLOW, &CYAN, &MAGENTA, &BLACK];

        chart.draw_series((0..self.height).flat_map(|y| {
            (0..self.width).map(move |x| {
                let idx = y * self.width + x;
                let label = self.labels[idx];
                let color = colors[label % colors.len()];

                Rectangle::new(
                    [(x as f64, y as f64), ((x + 1) as f64, (y + 1) as f64)],
                    color.filled(),
                )
            })
        }))?;

        root.present()?;
        Ok(())
    }

    pub fn plot_field(
        &self,
        color_idx: usize,
        filename: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let root = SVGBackend::new(filename, (800, 800)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .build_cartesian_2d(0f64..(self.width as f64), 0f64..(self.height as f64))?;

        chart.draw_series((0..self.height).flat_map(|y| {
            (0..self.width).flat_map(move |x| {
                let idx = y * self.width + x;
                let val = self.h_field[idx][color_idx];

                // Map H field min_h to a grayscale intensity [0, 255]
                let norm = if self.min_h < 0.0 {
                    (val / self.min_h).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let intensity = (255.0 * (1.0 - norm)) as u8; // Higher values = darker cells
                let color = RGBColor(intensity, intensity, intensity);

                let rect_bounds = [(x as f64, y as f64), ((x + 1) as f64, (y + 1) as f64)];

                vec![
                    Rectangle::new(rect_bounds, color.filled()),
                    Rectangle::new(rect_bounds, BLACK.stroke_width(1)),
                ]
            })
        }))?;

        root.present()?;
        Ok(())
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;

    pub fn create_mock(
        width: usize,
        height: usize,
        num_colors: usize,
        min_h: f64,
        error_prob: f64,
        uniform_j: Option<f64>,
        seed: u64,
    ) -> HMRF {
        assert!(num_colors % 2 == 0, "num_colors must be an even number");
        let mut rng = StdRng::seed_from_u64(seed);
        let n_spots = width * height;

        let labels: Vec<usize> = (0..n_spots).map(|_| rng.gen_range(0..num_colors)).collect();

        // Print color proportions
        let mut color_counts = vec![0; num_colors];
        for &label in &labels {
            color_counts[label] += 1;
        }

        // Dirichlet sample the x coordinates into a partition m / 2
        let n_x_parts = num_colors / 2;
        let mut x_weights = Vec::new();
        for _ in 0..n_x_parts {
            let mut u = rng.gen::<f64>();
            if u == 0.0 {
                u = 1e-10;
            }
            x_weights.push(-u.ln());
        }
        let x_sum: f64 = x_weights.iter().sum();

        let mut x_boundaries = vec![0.0];
        let mut current_x = 0.0;
        for w in x_weights {
            current_x += (w / x_sum) * (width as f64);
            x_boundaries.push(current_x);
        }
        x_boundaries[n_x_parts] = width as f64 + 1.0; // Ensure edge coverage

        // Dirichlet sample the y coordinates into a partition 2
        let mut u1 = rng.gen::<f64>();
        if u1 == 0.0 {
            u1 = 1e-10;
        }
        let mut u2 = rng.gen::<f64>();
        if u2 == 0.0 {
            u2 = 1e-10;
        }
        let y_boundary = (-u1.ln() / (-u1.ln() - u2.ln())) * (height as f64);

        // Drive the field such that Hnm = min_h for a given m in each partition
        let mut h_field = vec![vec![0.0; num_colors]; n_spots];
        let mut partition_counts = vec![0; num_colors];

        for y in 0..height {
            for x in 0..width {
                let i = y * width + x;

                let mut x_part = 0;
                while (x as f64) >= x_boundaries[x_part + 1] {
                    x_part += 1;
                }

                let y_part = if (y as f64) < y_boundary { 0 } else { 1 };

                let mut color_idx = y_part * n_x_parts + x_part;
                
                // Introduce structural noise based on error_prob
                if rng.gen::<f64>() < error_prob {
                    color_idx = rng.gen_range(0..num_colors);
                }

                h_field[i][color_idx] = min_h;
                partition_counts[color_idx] += 1;
            }
        }

        println!("Partition proportions (Field Assignments):");
        for (i, &count) in partition_counts.iter().enumerate() {
            println!(
                "  Partition {}: {:.2}%",
                i,
                (count as f64 / n_spots as f64) * 100.0
            );
        }

        // NB generate adjacency list for square lattice
        let mut adj_list = vec![vec![]; n_spots];
        let mut x_coords = vec![0.0; n_spots];
        let mut y_coords = vec![0.0; n_spots];
        for y in 0..height {
            for x in 0..width {
                let i = y * width + x;

                x_coords[i] = x as f64;
                y_coords[i] = y as f64;

                // Only evaluate right and down neighbors to construct symmetric edges once
                if x + 1 < width {
                    let j = i + 1;
                    let j_weight = uniform_j.unwrap_or_else(|| rng.gen::<f64>());
                    adj_list[i].push((j, j_weight));
                    adj_list[j].push((i, j_weight));
                }

                if y + 1 < height {
                    let j = i + width;
                    let j_weight = uniform_j.unwrap_or_else(|| rng.gen::<f64>());
                    adj_list[i].push((j, j_weight));
                    adj_list[j].push((i, j_weight));
                }
            }
        }

        HMRF {
            labels,
            adj_list,
            h_field,
            x: x_coords,
            y: y_coords,
            num_colors,
            width,
            height,
            min_h,
        }
    }

    pub fn create_test_mock() -> HMRF {
        create_mock(100, 100, 4, -10.0, 0.50, Some(5.0), 1234)
    }

    #[test]
    fn test_plot_mock_hmrf() {
        let hmrf = create_test_mock();

        assert!(hmrf.plot_labels("test_labels.svg").is_ok());

        assert!(hmrf.plot_field(0, "mock_field_c0.svg").is_ok());
        assert!(hmrf.plot_field(1, "mock_field_c1.svg").is_ok());
        assert!(hmrf.plot_field(2, "mock_field_c2.svg").is_ok());
        assert!(hmrf.plot_field(3, "mock_field_c3.svg").is_ok());
    }

    #[test]
    fn test_spinglass_potts_energy() {
        let beta = 1.0;
        let hmrf = create_test_mock();

        let cost = hmrf.potts_energy(beta);

        println!("Potts Cost: {}", cost);
    }

    #[test]
    fn test_icm_annealing() {
        let mut hmrf = create_test_mock();

        println!("Initial Energy: {}", hmrf.potts_energy(1.0));
        assert!(hmrf.plot_labels("icm_annealing_init.svg").is_ok());

        // NB only equal to, or greater than 0, matters.
        let betas = vec![0.0, 1.0];
        let icm_iters_per_temp = 1_000;

        for (i, &beta) in betas.iter().enumerate() {
            println!("ICM annealing step {} (beta context: {})", i, beta);

            hmrf.icm(beta, icm_iters_per_temp);

            println!("  Energy: {}", hmrf.potts_energy(1.0));
            println!("  Clone proportions: {:?}", hmrf.clone_proportions());

            let filename = format!("icm_annealing_beta_{}.svg", beta);
            assert!(hmrf.plot_labels(&filename).is_ok());
        }
    }

    #[test]
    fn test_gibbs_annealing() {
        let mut hmrf = create_test_mock();

        assert!(hmrf.plot_labels("gibbs_annealing_init.svg").is_ok());

        let betas = vec![0.0, 5.0, 50.0, 1000.0];
        let gibbs_iters_per_temp = 1_000;

        // NB beta = 0 is random flipping; <clone proportion> = 1/num_colors; high energy.
        for (i, &beta) in betas.iter().enumerate() {
            println!("Gibbs annealing step {} with beta: {}", i, beta);

            hmrf.gibbs_sample(beta, gibbs_iters_per_temp);

            println!("  Energy: {}", hmrf.potts_energy(1.0));
            println!("  Clone proportions: {:?}", hmrf.clone_proportions());

            let filename = format!("gibbs_annealing_beta_{}.svg", beta);
            assert!(hmrf.plot_labels(&filename).is_ok());
        }
    }

    #[test]
    fn test_swendsen_wang_annealing() {
        let mut hmrf = create_test_mock();

        assert!(hmrf.plot_labels("swendsen_wang_annealing_init.svg").is_ok());

        let betas = vec![0.0, 5.0, 50.0, 1000.0];
        let sw_iters_per_temp = 1_000;

        // NB beta = 0 is random flipping; <clone proportion> = 1/num_colors; high energy.
        for (i, &beta) in betas.iter().enumerate() {
            println!("SW annealing step {} with beta: {}", i, beta);

            hmrf.swendsen_wang(beta, sw_iters_per_temp);

            println!("  Energy: {}", hmrf.potts_energy(1.0));
            println!("  Clone proportions: {:?}", hmrf.clone_proportions());

            let filename = format!("swendsen_wang_annealing_beta_{}.svg", beta);
            assert!(hmrf.plot_labels(&filename).is_ok());
        }
    }

    #[test]
    fn test_wolff_annealing() {
        let mut hmrf = create_test_mock();

        assert!(hmrf.plot_labels("wolff_annealing_init.svg").is_ok());

        let betas = vec![0.0, 5.0, 50.0, 1000.0];
        let num_cluster_updates = 10_000;

        // NB beta = 0 is random flipping; <clone proportion> = 1/num_colors; high energy.
        for (i, &beta) in betas.iter().enumerate() {
            println!("Wolff annealing step {} with beta: {}", i, beta);

            hmrf.wolff_sample(beta, num_cluster_updates);

            println!("  Energy: {}", hmrf.potts_energy(1.0));
            println!("  Clone proportions: {:?}", hmrf.clone_proportions());

            let filename = format!("wolff_annealing_beta_{}.svg", beta);
            assert!(hmrf.plot_labels(&filename).is_ok());
        }
    }

    #[test]
    fn test_mean_field_decode() {
        let mut hmrf = create_test_mock();

        assert!(hmrf.plot_labels("mfd_init.svg").is_ok());

        let betas = vec![0.0, 1.0, 2.0, 5.0, 50.0];
        let max_iters = 100;
        let tol = 1e-4;

        for (i, &beta) in betas.iter().enumerate() {
            println!("Mean field decode step {} with beta: {}", i, beta);

            hmrf.mean_field_decode(beta, max_iters, tol);

            println!("  Energy: {}", hmrf.potts_energy(1.0));
            println!("  Clone proportions: {:?}", hmrf.clone_proportions());

            let filename = format!("mfd_beta_{}.svg", beta);
            assert!(hmrf.plot_labels(&filename).is_ok());
        }
    }
}
