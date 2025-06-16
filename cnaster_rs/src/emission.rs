use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use statrs::distribution::{gamma, poisson, NegativeBinomial, Discrete};
use rand::distributions::Distribution;
use pyo3::Python;
use std::f64;

use crate::get_rng;

#[pyfunction]
pub fn sample_segment_umis<'py>(
    py: Python<'py>,
    segment_baseline_umis: PyReadonlyArray1<'py, i64>,
    rdrs: PyReadonlyArray1<'py, f64>,
    rdr_overdispersion: f64,
) -> &'py PyArray1<i64> {
    let baseline = segment_baseline_umis.as_slice().unwrap();
    let rdrs = rdrs.as_slice().unwrap();

    let num_segments = baseline.len();

    let mut rng = get_rng();
    
    let result = PyArray1::<i64>::zeros(py, num_segments, false);

    let result_slice = unsafe { result.as_slice_mut().unwrap() };

    let r = 1.0 / rdr_overdispersion;

    for ((&base, &rdr), out) in baseline.iter().zip(rdrs.iter()).zip(result_slice.iter_mut()) {
        let mu = rdr * (base as f64);
        let p = 1. / (1. + rdr_overdispersion * mu);

        
    }

    result
}