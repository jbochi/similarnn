Similarnn
=========

Fast document similarity server using approximate nearest neighbors

Installation
------------

Git clone and `pip install -r requirements.txt`

Running
-------

Start it with `CONFIG_PATH=tests/data/config.toml hug -f similarnn/server.py`


Running tests
-------------

Install dev dependencies: `pip install -r requirements-dev.txt`

Run `python setup.py test`

To get a coverage report:
- install package in edit mode: `pip install -e .`
- run `py.test --cov=similarnn tests/ --cov-report=html`
