Similarnn
=========

Fast document similarity server using approximate nearest neighbors

Installation
------------

Requires Python3.

Git clone and `pip install -r requirements.txt`

Running
-------

Start it with `CONFIG_PATH=tests/data/config.toml hug -f similarnn/server.py`

Better yet, use gunicorn:
- `pip install unicorn`
- `CONFIG_PATH=tests/data/config.toml gunicorn similarnn.server:__hug_wsgi__`


Running tests
-------------

Install dev dependencies: `pip install -r requirements-dev.txt`

Run `python setup.py test`

To get a coverage report:
- install package in edit mode: `pip install -e .`
- run `py.test --cov=similarnn tests/ --cov-report=html`
