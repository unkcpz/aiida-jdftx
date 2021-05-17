# -*- coding: utf-8 -*-
""" Tests `JdftxCalculation` class"""

from aiida.common import datastructures


def test_jdftx_default(fixture_sandbox, generate_calc_job,
                       generate_inputs_jdftx, file_regression):
    """Test a default `JdftxCalculation`."""
    entry_point_name = 'jdftx'

    inputs = generate_inputs_jdftx()
    calc_info = generate_calc_job(fixture_sandbox, entry_point_name, inputs)
    upf = inputs['pseudos']['Si']

    cmdline_params = ['-i', 'aiida.in', '-o', 'aiida.out']
    local_copy_list = [(upf.uuid, upf.filename, './pseudo/Si.upf')]
    retrieve_list = [
        'aiida.out', 'aiida.kPts', 'aiida.Ecomponents', 'aiida.lattice',
        'aiida.ionpos'
    ]

    # Check the attributes of the returned `CalcInfo`
    assert isinstance(calc_info, datastructures.CalcInfo)
    assert isinstance(calc_info.codes_info[0], datastructures.CodeInfo)
    assert sorted(
        calc_info.codes_info[0].cmdline_params) == sorted(cmdline_params)
    assert sorted(calc_info.local_copy_list) == sorted(local_copy_list)
    assert sorted(calc_info.retrieve_list) == sorted(retrieve_list)

    with fixture_sandbox.open('aiida.in') as handle:
        input_written = handle.read()

    # Checks on the files written to the sandbox folder as raw input
    assert sorted(fixture_sandbox.get_content_list()) == sorted(
        ['aiida.in', 'pseudo'])
    file_regression.check(input_written, encoding='utf-8', extension='.in')
