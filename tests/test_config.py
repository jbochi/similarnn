from similarnn import config
from similarnn import models

def test_load_config():
    loaded_config = config.load_config("tests/data/config.toml")
    assert type(loaded_config['models']['lda']) == models.LDAModel
