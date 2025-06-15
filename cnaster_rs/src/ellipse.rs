use nalgebra::{Matrix2, Vector2};
use pyo3::prelude::*;
use rand::Rng;
use std::f64::consts::PI;
use numpy::{PyArray2, PyReadonlyArray2, IntoPyArray};
use std::f64::consts::TAU;

#[derive(Debug, Clone)]
pub struct CnaEllipse {
    pub n: usize,
    pub det_L: f64,
    pub center: Vector2<f64>,
    pub L: Matrix2<f64>,
    pub Q: Matrix2<f64>,
}

impl CnaEllipse {
    pub fn new(center: Vector2<f64>, L: Matrix2<f64>) -> Self {
        let Q = &L * L.transpose();
        let det_L = L.determinant();

        CnaEllipse {
            center,
            L,
            Q,
            n: 2,
            det_L,
        }
    }

    pub fn from_inv_diagonal(center: Vector2<f64>, inv_diag: [f64; 2]) -> Self {
        let L = Matrix2::from_diagonal(&Vector2::new(1.0 / inv_diag[0], 1.0 / inv_diag[1]));

        Self::new(center, L)
    }

    pub fn contains(&self, pos_xy: Vector2<f64>) -> bool {
        let v = pos_xy - self.center;

        (self.L.transpose() * v).norm_squared() <= 1.0
    }

    pub fn overlaps(&self, other: &CnaEllipse, resolution: usize) -> bool {
        for i in 0..resolution {
            let theta = (i as f64) * TAU / (resolution as f64);
            let unit = Vector2::new(theta.cos(), theta.sin());
            
            let pt_self = self.center + self.L * unit;

            if other.contains(pt_self) {
                return true;
            }
        }

        false
    }

    pub fn rotate(&self, theta: f64) -> Self {
        let (s, c) = theta.sin_cos();
        let R = Matrix2::new(c, -s, s, c);

        Self::new(self.center, R * self.L)
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
        new_L *= factor.powi(2);

        // NB see https://arxiv.org/pdf/1908.09326
        Self::new(new_center, new_L)
    }
}


#[pyclass]
#[pyo3(name = "CnaEllipse")]
#[derive(Clone)]
pub struct pyCnaEllipse {
    pub inner: CnaEllipse,
}

#[pymethods]
impl pyCnaEllipse {
    #[new]
    pub fn new(center: &PyArray2<f64>, l: &PyArray2<f64>) -> PyResult<Self> {
        let center_binding = center.readonly();
        let center = center_binding.as_array();
        let l_binding = l.readonly();
        let l = l_binding.as_array();

        // Expect center shape (2,1)
        if center.shape() != [2, 1] {
            return Err(pyo3::exceptions::PyValueError::new_err("center must be shape (2,1)"));
        }
        let center_vec = Vector2::new(center[[0, 0]], center[[1, 0]]);

        if l.shape() != [2, 2] {
            return Err(pyo3::exceptions::PyValueError::new_err("L must be shape (2,2)"));
        }

        let l_mat = Matrix2::new(l[[0, 0]], l[[0, 1]], l[[1, 0]], l[[1, 1]]);
        let inner = CnaEllipse::new(center_vec, l_mat);

        Ok(pyCnaEllipse { inner })
    }

    #[staticmethod]
    pub fn from_diagonal(center: &PyArray2<f64>, diag: &PyArray2<f64>) -> PyResult<Self> {
        let center_binding = center.readonly();
        let center = center_binding.as_array();
        let diag_binding = diag.readonly();
        let diag = diag_binding.as_array();

        // Expect center shape (2,1)
        if center.shape() != [2, 1] {
            return Err(pyo3::exceptions::PyValueError::new_err("center must be shape (2,1)"));
        }
        let center_vec = Vector2::new(center[[0, 0]], center[[1, 0]]);

        // Expect diag shape (2,1)
        if diag.shape() != [2, 1] {
            return Err(pyo3::exceptions::PyValueError::new_err("diag must be shape (2,1)"));
        }
        let diag_arr = [diag[[0, 0]], diag[[1, 0]]];

        let inner = CnaEllipse::from_inv_diagonal(center_vec, diag_arr);

        Ok(pyCnaEllipse { inner })
    }

    pub fn contains(&self, pos_xy: &PyArray2<f64>) -> PyResult<bool> {
        let pos_binding = pos_xy.readonly();
        let pos = pos_binding.as_array();
        // Expect pos_xy shape (2,1)
        if pos.shape() != [2, 1] {
            return Err(pyo3::exceptions::PyValueError::new_err("pos_xy must be shape (2,1)"));
        }
        let v = Vector2::new(pos[[0, 0]], pos[[1, 0]]);
        Ok(self.inner.contains(v))
    }

    pub fn overlaps(&self, other: &pyCnaEllipse, resolution: Option<usize>) -> bool {
        let res = resolution.unwrap_or(1_000);

        self.inner.overlaps(&other.inner, res)
    }

    pub fn rotate(&self, theta: f64) -> Self {
        pyCnaEllipse { inner: self.inner.rotate(theta) }
    }

    pub fn get_daughter(&self, factor: f64) -> Self {
        pyCnaEllipse {
            inner: self.inner.get_daughter(factor)
        }
    }

    #[getter]
    pub fn center<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {
        let arr = ndarray::Array2::from_shape_vec((1, 2), vec![self.inner.center[0], self.inner.center[1]]).unwrap();

        arr.into_pyarray(py)
    }

    #[getter]
    pub fn l<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {
        let arr = ndarray::arr2(&[
            [self.inner.L[(0, 0)], self.inner.L[(0, 1)]],
            [self.inner.L[(1, 0)], self.inner.L[(1, 1)]],
        ]);

        arr.into_pyarray(py)
    }

    #[getter]
    pub fn q<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {
        let arr = ndarray::arr2(&[
            [self.inner.Q[(0, 0)], self.inner.Q[(0, 1)]],
            [self.inner.Q[(1, 0)], self.inner.Q[(1, 1)]],
        ]);

        arr.into_pyarray(py)
    }

    #[getter]
    pub fn det_l(&self) -> f64 {
        self.inner.det_L
    }
}

#[pymodule]
pub fn ellipse(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<pyCnaEllipse>()?;
    Ok(())
}