import tempfile
import pytest
from cnaster.config import YAMLConfig, set_global_config, get_global_config

@pytest.fixture
def temp_yaml_config():
    yaml_content = """
    foo: bar
    nested:
      a: 1
      b: 2
      c: none
    """
    with tempfile.NamedTemporaryFile("w+", suffix=".yaml", delete=True) as tf:
        tf.write(yaml_content)
        tf.flush()
        config = YAMLConfig.from_file(tf.name)
        yield config  # Provide the config to the test

def test_yaml_config(temp_yaml_config):
    config = temp_yaml_config
    assert config.foo == "bar"
    assert config.nested.a == 1
    assert config.nested.b == 2
    assert config.nested.c is None

def test_global_config(monkeypatch, temp_yaml_config):
    set_global_config(temp_yaml_config)
    assert get_global_config() == temp_yaml_config

def test_global_config_warning(caplog):
    set_global_config(None)

    with caplog.at_level("WARNING"):
        result = get_global_config()
        assert result is None
        assert "cnaster config has not been defined." in caplog.text
