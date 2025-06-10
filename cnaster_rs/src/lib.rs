extern crate statrs;

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use statrs::function::gamma::{ln_gamma, digamma};
use statrs::function::factorial::ln_factorial;
use ndarray::parallel::prelude::IntoParallelRefIterator;
use ndarray::parallel::prelude::IndexedParallelIterator;
use ndarray::parallel::prelude::ParallelIterator;
use rayon::ThreadPoolBuilder;
use once_cell::sync::Lazy;
use std::env;
use num_cpus;

#[pymodule]
#[pyo3(name = "core")]
fn core(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
}