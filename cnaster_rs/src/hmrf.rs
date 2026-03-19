use rand::{rngs::StdRng, Rng, SeedableRng};

pub fn potts_cost(
    labels: &[usize],
    adj_list: &[Vec<(usize, f64)>],
    h_field: &[Vec<f64>],
    beta: f64,
) -> f64 {
    let mut energy = 0.0;

    // External field contribution: -H_{i, c_i}
    for (i, &c_i) in labels.iter().enumerate() {
        energy += h_field[i][c_i];
    }

    // Pairwise interaction contribution: -J_{ij} delta(c_i, c_j)
    // We assume adj_list contains symmetric directed edges (i->j and j->i).
    // To avoid doubling, we only add when i < j, or divide by 2.
    let mut interaction_energy = 0.0;
    for (i, &c_i) in labels.iter().enumerate() {
        for &(j, edge_weight) in &adj_list[i] {
            if i < j && c_i != labels[j] {
                interaction_energy += edge_weight;
            }
        }
    }
    
    energy += interaction_energy;
    -beta * energy
}

#[cfg(test)]
mod tests {
    use super::*;

    pub struct MockData {
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

    fn generate_mock_array(
        width: usize,
        height: usize,
        num_colors: usize,
        max_h: f64,
        uniform_j: Option<f64>,
        seed: u64,
    ) -> MockData {
        let mut rng = StdRng::seed_from_u64(seed);
        let n_spots = width * height;

        let labels: Vec<usize> = (0..n_spots).map(|_| rng.gen_range(0..num_colors)).collect();

        // NB generate H_field as a checkerboard pattern
        let mut h_field = vec![vec![0.0; num_colors]; n_spots];
        for y in 0..height {
            for x in 0..width {
                let i = y * width + x;
                if num_colors > 0 {
                    let color_idx = ((x + y) % 2).min(num_colors - 1);
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
        
        MockData {
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
    fn test_spinglass_potts_cost() {
        let width = 25;
        let height = 25;
        let beta = 1.0;
        let mock_data = generate_mock_array(width, height, 4, 1.0, None, 1337);

        let cost = potts_cost(&mock_data.labels, &mock_data.adj_list, &mock_data.h_field, beta);

        println!("Potts Cost: {}", cost);
    }
}
