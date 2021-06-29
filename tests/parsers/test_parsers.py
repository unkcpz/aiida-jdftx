# -*- coding: utf-8 -*-
"""Tests for the `JdftxParser`."""
import pytest

from aiida import orm
from aiida.common import AttributeDict


@pytest.fixture
def generate_inputs(generate_structure):
    """Return only those inputs that the parser will expect to be there."""
    def _generate_inputs(parameters=None, settings=None, metadata=None):
        structure = generate_structure()
        parameters = {**(parameters or {})}
        kpoints = orm.KpointsData()
        kpoints.set_cell_from_structure(structure)
        kpoints.set_kpoints_mesh_from_density(0.15)

        return AttributeDict({
            'structure': generate_structure(),
            'kpoints': kpoints,
            'parameters': orm.Dict(dict=parameters),
            'settings': orm.Dict(dict=settings),
            'metadata': metadata or {}
        })

    return _generate_inputs


def test_pw_default(fixture_localhost, generate_calc_job_node, generate_parser,
                    generate_inputs, data_regression):
    """Test a `jdftx` calculation in `electronic-min` only mode.
    The output is created by running a dead simple calculation for a silicon structure. This test should test the
    standard parsing of the stdout content in the standard results node.
    """
    name = 'default'
    entry_point_calc_job = 'jdftx'
    entry_point_parser = 'jdftx'

    node = generate_calc_job_node(entry_point_calc_job, fixture_localhost,
                                  name, generate_inputs())
    parser = generate_parser(entry_point_parser)
    results, calcfunction = parser.parse_from_node(node,
                                                   store_provenance=False)

    assert calcfunction.is_finished, calcfunction.exception
    assert calcfunction.is_finished_ok, calcfunction.exit_message
    # assert not orm.Log.objects.get_logs_for(node), [
    #     log.message for log in orm.Log.objects.get_logs_for(node)
    # ]
    assert 'output_parameters' in results
    assert 'output_kpoints' in results
    assert 'output_structure' in results
    assert 'output_trajectory' in results

    data_regression.check({
        'output_parameters':
        results['output_parameters'].get_dict(),
        'output_kpoints':
        results['output_kpoints'].attributes,
        'output_structure':
        results['output_structure'].attributes,
        'output_trajectory':
        results['output_trajectory'].attributes,
    })


def test_pw_relax(fixture_localhost, generate_calc_job_node, generate_parser,
                  generate_inputs, data_regression):
    """Test a `jdftx` calculation in `relax` mode (include lattice or ion relax only).
    The output is created by running a simple lattice variable relax calculation.
    """
    name = 'relax'
    entry_point_calc_job = 'jdftx'
    entry_point_parser = 'jdftx'

    inputs = generate_inputs(
        parameters={'lattice-minimize': {
            'nIteration': 10
        }})
    node = generate_calc_job_node(entry_point_calc_job, fixture_localhost,
                                  name, inputs)
    parser = generate_parser(entry_point_parser)
    results, calculation = parser.parse_from_node(node, store_provenance=False)

    assert calculation.is_finished, calculation.exception
    assert calculation.is_finished_ok, calculation.exit_message

    assert 'output_parameters' in results
    assert 'output_kpoints' in results
    assert 'output_structure' in results
    assert 'output_trajectory' in results

    data_regression.check({
        'output_parameters':
        results['output_parameters'].get_dict(),
        'output_kpoints':
        results['output_kpoints'].attributes,
        'output_structure':
        results['output_structure'].attributes,
        'output_trajectory':
        results['output_trajectory'].attributes,
    })
