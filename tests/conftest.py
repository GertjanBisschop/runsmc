import os
import pytest
import msprime

os.environ["NUMBA_DISABLE_JIT"] = "0"

@pytest.hookimpl
def pytest_configure(config):
    version = msprime.__version__.split('.')
    try:
        assert int(version[1]) >= 2
        assert version[3].startswith('dev')
    except AssertionError:
        print('Install the lastest msprime dev version.')