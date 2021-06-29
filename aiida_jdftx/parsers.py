# -*- coding: utf-8 -*-
"""
Parsers provided by aiida_jdftx.

Register parsers via the "aiida.parsers" entry point in setup.json.
"""
#pylint: disable=too-many-nested-blocks, too-many-branches
import numpy as np

from aiida import orm
from aiida.parsers.parser import Parser
from aiida.plugins import CalculationFactory
from aiida.common import OutputParsingError, exceptions

from ._constants import CONSTANTS

JdftxCalculation = CalculationFactory('jdftx')

units_suffix = '_units'
default_energy_units = 'eV'


class JdftxOutputParsingError(OutputParsingError):
    """Exception raised when there is a parsing error in the Jdftx parser."""


class JdftxParser(Parser):
    """
    Parser class for parsing output of calculation.
    """
    def __init__(self, node):
        """
        Initialize Parser instance

        Checks that the ProcessNode being passed was produced by a JdftxCalculation.

        :param node: ProcessNode of calculation
        :param type node: :class:`aiida.orm.ProcessNode`
        """
        super().__init__(node)
        if not issubclass(node.process_class, JdftxCalculation):
            raise exceptions.ParsingError('Can only parse JdftxCalculation')

    def parse(self, **kwargs):
        """
        Parse outputs, store results in database.

        :returns: an exit code, if parsing fails (or nothing if parsing succeeds)
        """
        self.exit_code_stdout = None

        parsed_stdout = self.parse_stdout()

        parameters = parsed_stdout.pop('parameters', {})
        ecomponents = self.parsed_ecomponents()
        parameters.update(ecomponents)

        output_parameters = orm.Dict(dict=parameters)
        parsed_trajectory = parsed_stdout.pop('trajectory', {})
        output_structure = self.parsed_structure()
        output_trajectory = self.build_output_trajectory(
            parsed_trajectory, output_structure)
        output_kpoints = self.parsed_kpoints(output_structure)

        self.out('output_parameters', output_parameters)

        if output_kpoints:
            self.out('output_kpoints', output_kpoints)

        if output_trajectory:
            self.out('output_trajectory', output_trajectory)

        if not output_structure.is_stored:
            self.out('output_structure', output_structure)

        if self.exit_code_stdout:
            return self.exit_code_stdout
        return None

    @staticmethod
    def build_output_trajectory(parsed_trajectory, structure):
        """doc"""
        try:
            cells = np.array(parsed_trajectory.pop('lattice_relax'))
        except KeyError:
            # no ionic or lattice iteration
            cells = np.array([structure.cell])

        try:
            positions = np.array(
                parsed_trajectory.pop('atomic_positios_relax'))
        except KeyError:
            # no ionic or lattice iteration
            positions = np.array([[site.position for site in structure.sites]])

        symbols = [str(site.kind_name) for site in structure.sites]
        stepids = np.arange(len(positions))

        trajectory = orm.TrajectoryData()
        trajectory.set_trajectory(
            stepids=stepids,
            cells=cells,
            positions=positions,
            symbols=symbols,
        )

        for key, value in parsed_trajectory.items():
            trajectory.set_array(key, np.array(value))

        return trajectory

    def parsed_kpoints(self, structure: orm.StructureData) -> orm.KpointsData:
        """Parse kpoints from end dumped file `aiida.kPts`"""
        filename = 'aiida.kPts'

        if filename not in self.retrieved.list_object_names():
            self.exit_code_stdout = self.exit_codes.ERROR_OUTPUT_STDOUT_MISSING
            return None

        try:
            stdout = self.retrieved.get_object_content(filename)
        except IOError:
            self.exit_code_stdout = self.exit_codes.ERROR_OUTPUT_STDOUT_READ
            return None

        # parse kpoints
        data_lines = stdout.strip().split('\n')

        kpoints_list = []
        kpoints_weights = []
        for line in data_lines:
            kpoints_list.append(
                [float(i) for i in line.split('[')[1].split(']')[0].split()])
            kpoints_weights.append(float(line.split('[')[1].split(']')[1]))

        kpoints = orm.KpointsData()
        kpoints.set_cell_from_structure(structure)
        kpoints.set_kpoints(kpoints_list,
                            weights=kpoints_weights,
                            cartesian=False)

        return kpoints

    def parsed_ecomponents(self) -> dict:
        """
        parse `aiida.Ecomponents` and return a dict of energies
        """
        filename = 'aiida.Ecomponents'

        if filename not in self.retrieved.list_object_names():
            self.exit_code_stdout = self.exit_codes.ERROR_OUTPUT_STDOUT_MISSING
            return {}

        try:
            ecomponots_stdout = self.retrieved.get_object_content(filename)
        except IOError:
            self.exit_code_stdout = self.exit_codes.ERROR_OUTPUT_STDOUT_READ
            return {}

        ecomponots = {}

        data_lines = ecomponots_stdout.split('\n')

        for line in data_lines:
            for string, key in [
                ['Eewald', 'energy_ewald'],
                ['EH', 'energy_hartree'],
                ['Eloc', 'energy_local'],
                ['Enl', 'energy_nonlocal'],
                ['Exc', 'energy_xc'],
                ['Exc_core', 'energy_xc_core'],
                ['KE', 'energy_kinetic'],
                ['Etot', 'energy_total'],
            ]:
                if string in line:
                    value = grep_energy_from_line(line)
                    ecomponots[key] = value
                    ecomponots[key + units_suffix] = default_energy_units

        return ecomponots

    def parsed_structure(self) -> orm.StructureData:
        """
        parse structure from end dumped file `aiida.lattice` and `aiida.ionpos`

        :param lattice: str, the content of the `aiida.lattice` file
        :param ionpos: str, content of the `aiida.ionpos` file
        """
        filename = 'aiida.lattice'

        if filename not in self.retrieved.list_object_names():
            self.exit_code_stdout = self.exit_codes.ERROR_OUTPUT_STDOUT_MISSING
            return self.node.inputs.structure

        try:
            lattice_stdout = self.retrieved.get_object_content(filename)
        except IOError:
            self.exit_code_stdout = self.exit_codes.ERROR_OUTPUT_STDOUT_READ
            return self.node.inputs.structure

        filename = 'aiida.ionpos'

        if filename not in self.retrieved.list_object_names():
            self.exit_code_stdout = self.exit_codes.ERROR_OUTPUT_STDOUT_MISSING
            return self.node.inputs.structure

        try:
            ionpos_stdout = self.retrieved.get_object_content(filename)
        except IOError:
            self.exit_code_stdout = self.exit_codes.ERROR_OUTPUT_STDOUT_READ
            return self.node.inputs.structure

        data_line = lattice_stdout.strip().split('\n')

        unit_cell = []
        for line in data_line[1:]:
            unit_cell.append([float(i) for i in line.split()[:3]])

        structure = orm.StructureData(cell=unit_cell)

        data_line = ionpos_stdout.strip().split('\n')
        for line in data_line[1:]:
            position = tuple(float(i) for i in line.split()[2:5])
            element = line.split()[1]
            structure.append_atom(position=position, symbols=element)

        return structure

    def parse_stdout(self) -> dict:
        """Parse the stdout output file into a dict.

        :return: dict with parsed data
        """
        parsed_data = {
            'parameters': {},
            'trajectory': {},
        }

        filename_stdout = self.node.get_option('output_filename')

        if filename_stdout not in self.retrieved.list_object_names():
            self.exit_code_stdout = self.exit_codes.ERROR_OUTPUT_STDOUT_MISSING
            return parsed_data

        try:
            stdout = self.retrieved.get_object_content(filename_stdout)
        except IOError:
            self.exit_code_stdout = self.exit_codes.ERROR_OUTPUT_STDOUT_READ
            return parsed_data

        # ============== Start real parsing =====================
        # Separate the input string into separate lines
        data_lines = stdout.split('\n')

        # loop over all lines to get a broadview of the results
        calc_success = False
        for line in data_lines:

            if 'Done!' in line:
                calc_success = True

        # parsing the all initial parameters

        # clip the std-out to a list of every relax step
        # the list contain the list of lines of every electroinc minimization
        relax_steps = stdout.split(
            '-------- Electronic minimization -----------')[1:]
        relax_steps = [i.split('\n') for i in relax_steps]

        trajectory_data = {}

        for data_step in relax_steps:

            do_relax = False  # defined for after check

            for count, line in enumerate(data_step):

                if '# Energy components:' in line:
                    for line2 in data_step[count:]:
                        # exit loop
                        if not line2.strip():
                            break

                        for string, key in [
                            ['Eewald', 'energy_ewald'],
                            ['EH', 'energy_hartree'],
                            ['Eloc', 'energy_local'],
                            ['Enl', 'energy_nonlocal'],
                            ['Exc', 'energy_xc'],
                            ['Exc_core', 'energy_xc_core'],
                            ['KE', 'energy_kinetic'],
                            ['Etot', 'energy_total'],
                        ]:
                            if string in line2:
                                value = grep_energy_from_line(line2)
                                trajectory_data.setdefault(key,
                                                           []).append(value)

                if '# Lattice vectors:' in line:
                    a1 = [float(s) for s in data_step[count + 2].split()[1:4]]
                    a2 = [float(s) for s in data_step[count + 3].split()[1:4]]
                    a3 = [float(s) for s in data_step[count + 4].split()[1:4]]

                    cell = np.array([a1, a2, a3]) * CONSTANTS.bohr_to_ang
                    trajectory_data.setdefault('lattice_relax',
                                               []).append(cell)
                    # only in relax calculation will cell be printed out
                    do_relax = True

                if '# Ionic positions in lattice coordinates:' in line:
                    positions = []
                    for line2 in data_step[count + 1:]:
                        # exit loop
                        if not line2.strip():
                            break

                        positions.append(
                            [float(i) for i in line2.split()[2:5]])

            # at each frame the positions is fractional, transform to angstrom
            if do_relax:
                # only in relax calculation will cell be set
                # transform from fraction to angstrom
                positions = np.matmul(np.array(positions), cell)
                trajectory_data.setdefault('atomic_positios_relax',
                                           []).append(positions)

        parsed_data['trajectory'] = trajectory_data

        if not calc_success:
            self.exit_code_stdout = self.exit_codes.ERROR_UNEXPECTED_PARSER_EXCEPTION

        return parsed_data


def grep_energy_from_line(line):
    """extract energy from line"""
    try:
        return float(line.split('=')[1]) * CONSTANTS.har_to_ev
    except Exception:
        raise JdftxOutputParsingError(
            'Error while parsing energy') from Exception
