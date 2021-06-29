# -*- coding: utf-8 -*-
"""Tests for the `JdftxBaseWorkChain` class."""
from aiida.common import AttributeDict


def test_setup(generate_workchain_jdftx):
    """Test `JdftxBaseWorkChain.setup`."""
    process = generate_workchain_jdftx()
    process.setup()

    assert process.ctx.restart_calc is None
    assert isinstance(process.ctx.inputs, AttributeDict)
