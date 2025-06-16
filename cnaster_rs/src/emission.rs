use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use statrs::distribution::{NegativeBinomial, Discrete};
use rand::distributions::Distribution;
use pyo3::Python;

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
    let mut result = Vec::with_capacity(num_segments);

    for i in 0..num_segments {
        let mu = rdrs[i] * (baseline[i] as f64);

        let r = 1.0 / rdr_overdispersion;
        let p = 1. / (1. + rdr_overdispersion * mu);

        let nb = NegativeBinomial::new(r, p).unwrap();
        let sample = nb.sample(&mut *rng) as i64;

        result.push(sample);
    }

    PyArray1::from_vec(py, result)
}