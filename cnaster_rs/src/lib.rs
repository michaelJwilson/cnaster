use pyo3::prelude::*;
use std::collections::HashMap;

pub struct Graph<T> {
    pub adjacency_list: HashMap<T, Vec<T>>,
}

impl<T: Eq + std::hash::Hash + Clone> Graph<T> {
    pub fn new() -> Self {
        Graph {
            adjacency_list: HashMap::new(),
        }
    }

    pub fn add_edge(&mut self, from: T, to: T) {
        self.adjacency_list.entry(from.clone()).or_insert_with(Vec::new).push(to.clone());
        self.adjacency_list.entry(to).or_insert_with(Vec::new).push(from);
    }

    pub fn neighbors(&self, node: &T) -> Option<&Vec<T>> {
        self.adjacency_list.get(node)
    }
}




#[pymodule]
#[pyo3(name = "core")]
fn core(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
}