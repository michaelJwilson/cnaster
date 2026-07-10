use serde::Deserialize;
use std::fs;

#[derive(Deserialize, Debug)]
struct ModelParameters {
    cna_rate: f64,
}

#[derive(Deserialize, Debug)]
struct CloneInitialization {
    min_clone_read_coverage: u32,
    min_clone_snp_coverage: u32,
}

#[derive(Deserialize, Debug)]
pub struct Config {
    model_parameters: ModelParameters,
    clone_initialization: CloneInitialization,
}

impl Config {
    pub fn from_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let data = fs::read_to_string(path)?;
        let config: Config = serde_json::from_str(&data)?;

        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_from_file() {
        let config = Config::from_file("/Users/mw9568/repos/cnaster/config.json")
            .expect("Failed to load config");
        println!("{:#?}", config);
        // Optionally, add assertions here to check config fields
        assert!(config.model_parameters.cna_rate > 0.0);
        assert!(config.clone_initialization.min_clone_read_coverage > 0);
        assert!(config.clone_initialization.min_clone_snp_coverage > 0);
    }
}
