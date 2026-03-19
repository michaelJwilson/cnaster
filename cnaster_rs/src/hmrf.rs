use plotters::prelude::*;
use rand::distributions::WeightedIndex;
use rand::prelude::Distribution;
use rand::{rngs::StdRng, Rng, SeedableRng};

pub struct HMRF {
    pub labels: Vec<usize>,
    pub adj_list: Vec<Vec<(usize, f64)>>,
    pub h_field: Vec<Vec<f64>>,
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub num_colors: usize,
    pub width: usize,
    pub height: usize,
    pub max_h: f64,
}

impl HMRF {
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
    pub fn icm(&mut self, max_iters: usize) {
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

    /// Calculate the proportions of each label (color) currently on the grid.
    pub fn clone_proportions(&self) -> Vec<f64> {
        let mut counts = vec![0; self.num_colors];
        for &label in &self.labels {
            counts[label] += 1;
        }
        let total = self.labels.len() as f64;
        counts.into_iter().map(|c| c as f64 / total).collect()
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

    /// Plot an external field slice as a grayscale heatmap (darker = higher H magnitude).
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

                // Map H field max_h to a grayscale intensity [0, 255]
                let norm = if self.max_h > 0.0 {
                    (val / self.max_h).clamp(0.0, 1.0)
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

    pub fn generate_mock_array(
        width: usize,
        height: usize,
        num_colors: usize,
        max_h: f64,
        uniform_j: Option<f64>,
        seed: u64,
    ) -> HMRF {
        let mut rng = StdRng::seed_from_u64(seed);
        let n_spots = width * height;

        let labels: Vec<usize> = (0..n_spots).map(|_| rng.gen_range(0..num_colors)).collect();

        // Print color proportions
        let mut color_counts = vec![0; num_colors];
        for &label in &labels {
            color_counts[label] += 1;
        }
        println!("Color proportions:");
        for (i, &count) in color_counts.iter().enumerate() {
            println!(
                "  Color {}: {:.2}%",
                i,
                (count as f64 / n_spots as f64) * 100.0
            );
        }

        // NB generate H_field as a checkerboard pattern
        let mut h_field = vec![vec![0.0; num_colors]; n_spots];
        let square_size = 5; // Set checkerboard square size > 1
        for y in 0..height {
            for x in 0..width {
                let i = y * width + x;
                if num_colors > 1 {
                    let color_idx = ((x / square_size + y / square_size) % 2).min(num_colors - 1);
                    h_field[i][color_idx] = max_h;
                }
            }
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
            max_h,
        }
    }

    #[test]
    fn test_plot_mock_hmrf() {
        let width = 25;
        let height = 25;
        let hmrf = generate_mock_array(width, height, 3, 1.0, None, 42);

        assert!(hmrf.plot_labels("test_labels.svg").is_ok());

        assert!(hmrf.plot_field(0, "test_field_c0.svg").is_ok());
        assert!(hmrf.plot_field(1, "test_field_c1.svg").is_ok());
        assert!(hmrf.plot_field(2, "test_field_c2.svg").is_ok());
    }

    #[test]
    fn test_spinglass_potts_energy() {
        let width = 25;
        let height = 25;
        let beta = 1.0;
        let hmrf = generate_mock_array(width, height, 4, 1.0, None, 1337);

        let cost = hmrf.potts_energy(beta);

        println!("Potts Cost: {}", cost);
    }

    #[test]
    fn test_icm() {
        let width = 25;
        let height = 25;
        let beta = 1.0;
        // Zero external field (0.0), uniform J=1.0
        let mut hmrf = generate_mock_array(width, height, 4, 0.0, Some(1.0), 1337);

        let initial_cost = hmrf.potts_energy(beta);
        println!("Initial Potts Cost: {}", initial_cost);

        assert!(hmrf.plot_labels("test_icm_labels_initial.svg").is_ok());

        hmrf.icm(10);

        let final_cost = hmrf.potts_energy(beta);
        println!("Final Potts Cost: {}", final_cost);

        assert!(hmrf.plot_labels("test_icm_labels_final.svg").is_ok());
        assert!(final_cost <= initial_cost);
    }

    #[test]
    fn test_gibbs_annealing() {
        let width = 100;
        let height = 100;
        // Moderate external field to create interesting patterns, uniform J=1.0
        let mut hmrf = generate_mock_array(width, height, 4, 2.0, Some(1.0), 42);

        assert!(hmrf.plot_labels("annealing_init.svg").is_ok());

        // beta = 1/T. Low beta means high temperature
        let betas = vec![0.0, 1.0, 2.0, 5.0, 50.0];
        let gibbs_iters_per_temp = 25;

        for (i, &beta) in betas.iter().enumerate() {
            println!("Annealing step {} with beta: {}", i, beta);
            
            hmrf.gibbs_sample(beta, gibbs_iters_per_temp);
            
            let filename = format!("annealing_beta_{}.svg", beta);
            assert!(hmrf.plot_labels(&filename).is_ok());
        }

        println!("Final Cost after Annealing: {}", hmrf.potts_energy(betas.last().copied().unwrap()));
    }
}
