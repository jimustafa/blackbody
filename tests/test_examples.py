import os
import pathlib
import runpy

import pytest


examples = sorted(pathlib.Path(__file__, '../../examples').resolve().glob('example-*/*.py'))


@pytest.mark.parametrize(
    'example',
    [
        pytest.param(example, id=example.with_suffix('').name) for example in examples
    ]
)
def test_example(tmp_path, example):
    os.chdir(tmp_path)
    runpy.run_path(str(example))
