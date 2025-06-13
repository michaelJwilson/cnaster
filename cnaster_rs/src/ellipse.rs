use nalgebra::{Matrix2, Vector2};
use rand::Rng;
use std::f64::consts::PI;

#[derive(Debug, Clone)]
pub struct Ellipse {
    pub n: usize,
    pub det_L: f64,
    pub center: Vector2<f64>,
    pub L: Matrix2<f64>,
    pub Q: Matrix2<f64>,
}

impl Ellipse {
    pub fn new(center: Vector2<f64>, L: Matrix2<f64>) -> Self {
        let Q = &L * L.transpose();
        let det_L = L.determinant();
        
        Ellipse { center, L, Q, n: 2, det_L }
    }

    pub fn from_lower_inv_diagonal(center: Vector2<f64>, inv_diag: [f64; 2]) -> Self {
        let L = Matrix2::new(1.0 / inv_diag[0], 1.0 / inv_diag[1]);
        
        Self::new(center, L)
    }

    pub fn contains(&self, pos_xy: Vector2<f64>) -> bool {
        let v = pos_xy - self.center;
        
        (self.L.transpose() * v).norm_squared() <= 1.0
    }

    pub fn rotate(&self, theta: f64) -> Self {
        let (s, c) = theta.sin_cos();        
        let R = Matrix2::new(c, -s, s, c);
        
        self::new(self.center, R * self.L)
    }

    pub fn get_daughter(&self, factor: f64) -> Self {
        // NB parent corresponds to the unit ball pushed forward by L.
        let new_radius = factor * 1.0;
        let new_x = new_radius - 1.0 + self.center[0];
        let new_center = Vector2::new(new_x, self.center[1]);

        // NB new projection
        let mut rng = rand::thread_rng();
        let theta = (PI / 4.0) * rng.gen::<f64>();

        let mut new_L = self.rotate(theta).L;
        new_L /= factor.powi(2);

        // NB see https://arxiv.org/pdf/1908.09326
        self::new(new_center, new_L)
    }
}