use pyo3::prelude::*;
use std::collections::HashMap;

pub struct Graph<T, W> {
    pub adjacency_list: HashMap<T, Vec<(T, W)>>,
}

impl<T: Eq + std::hash::Hash + Clone, W: Clone> Graph<T, W> {
    pub fn new() -> Self {
        Graph {
            adjacency_list: HashMap::new(),
        }
    }

    pub fn add_edge(&mut self, from: T, to: T, weight: W) {
        self.adjacency_list
            .entry(from.clone())
            .or_insert_with(Vec::new)
            .push((to.clone(), weight.clone()));
        self.adjacency_list
            .entry(to)
            .or_insert_with(Vec::new)
            .push((from, weight));
    }

    pub fn neighbors(&self, node: &T) -> Option<&Vec<(T, W)>> {
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
        let graph: Graph<i32, i32> = Graph::new();
        assert!(graph.adjacency_list.is_empty());
    }

    #[test]
    fn test_add_edge_and_neighbors() {
        let mut graph = Graph::new();
        graph.add_edge(1, 2, 10);
        graph.add_edge(1, 3, 20);
        graph.add_edge(2, 3, 30);

        let neighbors_1 = graph.neighbors(&1).unwrap();
        assert_eq!(neighbors_1, &vec![(2, 10), (3, 20)]);

        let neighbors_2 = graph.neighbors(&2).unwrap();
        assert_eq!(neighbors_2, &vec![(1, 10), (3, 30)]);

        assert!(graph.neighbors(&4).is_none());
    }

    #[test]
    fn test_graph_with_strings() {
        let mut graph = Graph::new();

        graph.add_edge("A".to_string(), "B".to_string(), 1.5);
        graph.add_edge("A".to_string(), "C".to_string(), 2.5);

        let neighbors = graph.neighbors(&"A".to_string()).unwrap();

        assert_eq!(
            neighbors,
            &vec![
                ("B".to_string(), 1.5),
                ("C".to_string(), 2.5)
            ]
        );
    }
}