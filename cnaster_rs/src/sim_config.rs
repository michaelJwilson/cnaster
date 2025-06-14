use serde::Deserialize;
use std::fs;

#[derive(Debug, Deserialize)]
pub struct PhasingConfig {
    pub switch_rate: f64,
}

#[derive(Debug, Deserialize)]
pub struct SimConfig {
    pub segment_kbp: u32,
    pub trans_rate: f64,
    pub copy_number_states: Vec<String>,
    pub rdr_dispersion: f64,
    pub baf_dispersion: f64,
    pub phasing: PhasingConfig,
}

impl SimConfig {
    pub fn from_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let data = fs::read_to_string(path)?;
        let config: SimConfig = serde_json::from_str(&data)?;
        
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sim_config_from_file() {
        let config = SimConfig::from_file("/Users/mw9568/repos/cnaster/sim_config.json")
            .expect("Failed to load sim config");

        println!("{:#?}", config);

        assert_eq!(config.segment_kbp, 50);
        assert!((config.trans_rate - 0.00001).abs() < 1e-8);
        assert_eq!(config.copy_number_states.len(), 4);
        assert_eq!(config.copy_number_states[0], "1,1");
        assert!((config.rdr_dispersion - 0.0).abs() < 1e-8);
        assert!((config.baf_dispersion - 0.0).abs() < 1e-8);
        assert!((config.phasing.switch_rate - 0.00001).abs() < 1e-8);
    }
}