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
        self.adjacency_list
            .entry(from.clone())
            .or_insert_with(Vec::new)
            .push(to.clone());
        self.adjacency_list
            .entry(to)
            .or_insert_with(Vec::new)
            .push(from);
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

    #[test]
    fn test_graph_new() {
        let graph: Graph<i32> = Graph::new();
        assert!(graph.adjacency_list.is_empty());
    }

    #[test]
    fn test_add_edge_and_neighbors() {
        let mut graph = Graph::new();
        graph.add_edge(1, 2);
        graph.add_edge(1, 3);
        graph.add_edge(2, 3);

        let neighbors_1 = graph.neighbors(&1).unwrap();
        assert_eq!(neighbors_1, &vec![2, 3]);

        let neighbors_2 = graph.neighbors(&2).unwrap();
        assert_eq!(neighbors_2, &vec![1, 3]);

        assert!(graph.neighbors(&4).is_none());
    }

    #[test]
    fn test_graph_with_strings() {
        let mut graph = Graph::new();
        
        graph.add_edge("A".to_string(), "B".to_string());
        graph.add_edge("A".to_string(), "C".to_string());

        let neighbors = graph.neighbors(&"A".to_string()).unwrap();
        
        assert_eq!(neighbors, &vec!["B".to_string(), "C".to_string()]);
    }
}
