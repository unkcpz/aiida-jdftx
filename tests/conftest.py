# -*- coding: utf-8 -*-
"""pytest fixtures for simplified testing."""
import os
import collections
import shutil

import pytest

pytest_plugins = ['aiida.manage.tests.pytest_fixtures']


@pytest.fixture(scope='function')
def fixture_sandbox():
    """Return a `SandboxFolder`."""
    from aiida.common.folders import SandboxFolder
    with SandboxFolder() as folder:
        yield folder


@pytest.fixture
def fixture_localhost(aiida_localhost):
    """Return a localhost `Computer`."""
    localhost = aiida_localhost
    localhost.set_default_mpiprocs_per_machine(1)
    return localhost


@pytest.fixture
def fixture_code(fixture_localhost):
    """Return a `Code` instance configured to run calculations of given entry point on localhost `Computer`."""
    def _fixture_code(entry_point_name):
        from aiida.common import exceptions
        from aiida.orm import Code

        label = f'test.{entry_point_name}'

        try:
            return Code.objects.get(label=label)  # pylint: disable=no-member
        except exceptions.NotExistent:
            return Code(
                label=label,
                input_plugin_name=entry_point_name,
                remote_computer_exec=[fixture_localhost, '/bin/true'],
            )

    return _fixture_code


@pytest.fixture(scope='session')
def generate_upf_data(tmp_path_factory):
    """Return a `UpfData` instance for the given element a file for which should exist in `tests/fixtures/pseudos`."""
    def _generate_upf_data(element):
        """Return `UpfData` node."""
        from aiida_pseudo.data.pseudo import UpfData

        with open(
                tmp_path_factory.mktemp('pseudos') / f'{element}.upf',
                'w+b') as handle:
            content = f'<UPF version="2.0.1"><PP_HEADER\nelement="{element}"\nz_valence="4.0"\n/></UPF>\n'
            handle.write(content.encode('utf-8'))
            handle.flush()
            return UpfData(file=handle)

    return _generate_upf_data


@pytest.fixture
def generate_structure():
    """Return a `StructureData` representing bulk silicon."""
    def _generate_structure(structure_id='silicon'):
        """Return a `StructureData` representing bulk silicon or a snapshot of a single water molecule dynamics."""
        from aiida.orm import StructureData

        if structure_id == 'silicon':
            param = 5.43
            cell = [[param / 2., param / 2., 0], [param / 2., 0, param / 2.],
                    [0, param / 2., param / 2.]]
            structure = StructureData(cell=cell)
            structure.append_atom(position=(0., 0., 0.),
                                  symbols='Si',
                                  name='Si')
            structure.append_atom(position=(param / 4., param / 4.,
                                            param / 4.),
                                  symbols='Si',
                                  name='Si')
        elif structure_id == 'water':
            structure = StructureData(
                cell=[[5.29177209, 0., 0.], [0., 5.29177209, 0.],
                      [0., 0., 5.29177209]])
            structure.append_atom(
                position=[12.73464656, 16.7741411, 24.35076238],
                symbols='H',
                name='H')
            structure.append_atom(
                position=[-29.3865565, 9.51707929, -4.02515904],
                symbols='H',
                name='H')
            structure.append_atom(
                position=[1.04074437, -1.64320127, -1.27035021],
                symbols='O',
                name='O')
        else:
            raise KeyError('Unknown structure_id=\'{}\''.format(structure_id))
        return structure

    return _generate_structure


@pytest.fixture
def generate_kpoints_mesh():
    """Return a `KpointsData` node."""
    def _generate_kpoints_mesh(npoints):
        """Return a `KpointsData` with a mesh of npoints in each direction."""
        from aiida.orm import KpointsData

        kpoints = KpointsData()
        kpoints.set_kpoints_mesh([npoints] * 3, [0.5, 0.5, 0.5])

        return kpoints

    return _generate_kpoints_mesh


@pytest.fixture
def generate_calc_job():
    """Fixture to construct a new `CalcJob` instance and call `prepare_for_submission` for testing `CalcJob` classes.
    The fixture will return the `CalcInfo` returned by `prepare_for_submission` and the temporary folder that was passed
    to it, into which the raw input files will have been written.
    """
    def _generate_calc_job(folder, entry_point_name, inputs=None):
        """Fixture to generate a mock `CalcInfo` for testing calculation jobs."""
        from aiida.engine.utils import instantiate_process
        from aiida.manage.manager import get_manager
        from aiida.plugins import CalculationFactory

        manager = get_manager()
        runner = manager.get_runner()

        process_class = CalculationFactory(entry_point_name)
        process = instantiate_process(runner, process_class, **inputs)

        calc_info = process.prepare_for_submission(folder)

        return calc_info

    return _generate_calc_job


@pytest.fixture
def generate_inputs_jdftx(fixture_code, generate_structure,
                          generate_kpoints_mesh, generate_upf_data):
    """Generate default inputs for a `JdftxCalculation."""
    def _generate_inputs_jdftx():
        """Generate default inputs for a `JdftxCalculation."""
        from aiida.orm import Dict
        from aiida_jdftx.utils import get_default_options

        inputs = {
            'code': fixture_code('jdftx'),
            'structure': generate_structure(),
            'kpoints': generate_kpoints_mesh(8),
            'parameters': Dict(dict={
                'elec-cutoff': '20 100',
                'lattice-minimize': {
                    'nIterations': 0,
                }
            }),
            'pseudos': {
                'Si': generate_upf_data('Si')
            },
            'metadata': {
                'options': get_default_options()
            }
        }

        return inputs

    return _generate_inputs_jdftx


@pytest.fixture
def generate_calc_job_node(fixture_localhost):
    """Fixture to generate a mock `CalcJobNode` for testing parsers."""
    def flatten_inputs(inputs, prefix=''):
        """Flatten inputs recursively like :meth:`aiida.engine.processes.process::Process._flatten_inputs`."""
        flat_inputs = []
        for key, value in inputs.items():
            if isinstance(value, collections.Mapping):
                flat_inputs.extend(
                    flatten_inputs(value, prefix=prefix + key + '__'))
            else:
                flat_inputs.append((prefix + key, value))
        return flat_inputs

    def _generate_calc_job_node(entry_point_name='base',
                                computer=None,
                                test_name=None,
                                inputs=None,
                                attributes=None,
                                retrieve_temporary=None):
        """Fixture to generate a mock `CalcJobNode` for testing parsers.
        :param entry_point_name: entry point name of the calculation class
        :param computer: a `Computer` instance
        :param test_name: relative path of directory with test output files in the `fixtures/{entry_point_name}` folder.
        :param inputs: any optional nodes to add as input links to the corrent CalcJobNode
        :param attributes: any optional attributes to set on the node
        :param retrieve_temporary: optional tuple of an absolute filepath of a temporary directory and a list of
            filenames that should be written to this directory, which will serve as the `retrieved_temporary_folder`.
            For now this only works with top-level files and does not support files nested in directories.
        :return: `CalcJobNode` instance with an attached `FolderData` as the `retrieved` node.
        """
        from aiida import orm
        from aiida.common import LinkType
        from aiida.plugins.entry_point import format_entry_point_string

        if computer is None:
            computer = fixture_localhost

        filepath_folder = None

        if test_name is not None:
            basepath = os.path.dirname(os.path.abspath(__file__))
            filename = os.path.join(entry_point_name, test_name)
            filepath_folder = os.path.join(basepath, 'parsers', 'fixtures',
                                           test_name)

        entry_point = format_entry_point_string('aiida.calculations',
                                                entry_point_name)

        node = orm.CalcJobNode(computer=computer, process_type=entry_point)
        node.set_attribute('input_filename', 'aiida.in')
        node.set_attribute('output_filename', 'aiida.out')
        node.set_option('resources', {
            'num_machines': 1,
            'num_mpiprocs_per_machine': 1
        })
        node.set_option('max_wallclock_seconds', 1800)

        if attributes:
            node.set_attribute_many(attributes)

        if inputs:
            metadata = inputs.pop('metadata', {})
            options = metadata.get('options', {})

            for name, option in options.items():
                node.set_option(name, option)

            for link_label, input_node in flatten_inputs(inputs):
                input_node.store()
                node.add_incoming(input_node,
                                  link_type=LinkType.INPUT_CALC,
                                  link_label=link_label)

        node.store()

        if retrieve_temporary:
            dirpath, filenames = retrieve_temporary
            for filename in filenames:
                shutil.copy(os.path.join(filepath_folder, filename),
                            os.path.join(dirpath, filename))

        if filepath_folder:
            retrieved = orm.FolderData()
            retrieved.put_object_from_tree(filepath_folder)

            # Remove files that are supposed to be only present in the retrieved temporary folder
            if retrieve_temporary:
                for filename in filenames:
                    retrieved.delete_object(filename)

            retrieved.add_incoming(node,
                                   link_type=LinkType.CREATE,
                                   link_label='retrieved')
            retrieved.store()

            remote_folder = orm.RemoteData(computer=computer,
                                           remote_path='/tmp')
            remote_folder.add_incoming(node,
                                       link_type=LinkType.CREATE,
                                       link_label='remote_folder')
            remote_folder.store()

        return node

    return _generate_calc_job_node


@pytest.fixture(scope='session')
def generate_parser():
    """Fixture to load a parser class for testing parsers."""
    def _generate_parser(entry_point_name):
        """Fixture to load a parser class for testing parsers.
        :param entry_point_name: entry point name of the parser class
        :return: the `Parser` sub class
        """
        from aiida.plugins import ParserFactory
        return ParserFactory(entry_point_name)

    return _generate_parser

@pytest.fixture
def generate_workchain():
    """Fixture to generate an instance of a `WorkChain`."""

    def _generate_workchain(entry_point, inputs):
        """Generate an instance of a `WorkChain` with entry_point and inputs

        :param entry_point: entry point name of the work chain subclass.
        :param inputs: inputs to be passed to process construction.
        :return: a `WorkChain` instance.
        """
        from aiida.engine.utils import instantiate_process
        from aiida.manage.manager import get_manager
        from aiida.plugins import WorkflowFactory

        process_class = WorkflowFactory(entry_point)
        runner = get_manager().get_runner()
        process = instantiate_process(runner, process_class, **inputs)

        return process

    return _generate_workchain

@pytest.fixture
def generate_workchain_jdftx(generate_workchain, generate_inputs_jdftx, generate_calc_job_node):
    """Generate an instance of a `JdftxBaseWorkChain`."""

    def _generate_workchain_jdftx(exit_code=None, inputs=None, return_inputs=False):
        from plumpy import ProcessState
        from aiida.orm import Dict

        entry_point = 'jdftx.base'

        if inputs is None:
            jdftx_inputs = generate_inputs_jdftx()
            kpoints = jdftx_inputs.pop('kpoints')
            inputs = {'jdftx': jdftx_inputs, 'kpoints': kpoints}

        if return_inputs:
            return inputs

        process = generate_workchain(entry_point, inputs)

        if exit_code is not None:
            node = generate_calc_job_node(inputs={'parameters': Dict()})
            node.set_process_state(ProcessState.FINISHED)
            node.set_exit_status(exit_code.status)

            process.ctx.iteration = 1
            process.ctx.children = [node]

        return process

    return _generate_workchain_jdftx
