# -*- coding: utf-8 -*-
"""
Calculations provided by aiida_jdftx.

Register calculations via the "aiida.calculations" entry point in setup.json.
"""
import os
from typing import Tuple

from aiida import orm
from aiida.common.datastructures import CalcInfo, CodeInfo
from aiida.common.folders import Folder
from aiida.common import exceptions
from aiida.engine import CalcJob
from aiida.plugins import DataFactory

from ._constants import CONSTANTS

UpfData = DataFactory('pseudo.upf')


class JdftxCalculation(CalcJob):
    """
    AiiDA calculation plugin wrapping the jdftx executable.
    """

    _PSEUDO_SUBFOLDER = './pseudo/'
    _DEFAULT_INPUT_FILE = 'aiida.in'
    _DEFAULT_OUTPUT_FILE = 'aiida.out'

    @classmethod
    def define(cls, spec):
        """Define inputs and outputs of the calculation."""
        # yapf: disable
        super(JdftxCalculation, cls).define(spec)

        # set default values for AiiDA options
        spec.input('metadata.options.input_filename', valid_type=str, default=cls._DEFAULT_INPUT_FILE)
        spec.input('metadata.options.output_filename', valid_type=str, default=cls._DEFAULT_OUTPUT_FILE)
        spec.input('metadata.options.withmpi', valid_type=bool, default=True)  # Override default withmpi=False
        spec.input('structure', valid_type=orm.StructureData,
            help='The input structure.')
        spec.input('parameters', valid_type=orm.Dict,
            help='The input parameters that are to be used to construct the input file.')
        spec.input_namespace('pseudos', valid_type=UpfData, dynamic=True,
            help='A mapping of `UpfData` nodes onto the kind name to which they should apply.')
        spec.input('kpoints', valid_type=orm.KpointsData,
            help='kpoint mesh or kpoint path')
        spec.input('settings', valid_type=orm.Dict, required=False,
            help='Optional parameters to affect the way the calculation job and the parsing are performed.')

        # set parser
        spec.input('metadata.options.parser_name', valid_type=str, default='jdftx')

        # new ports
        spec.output('output_parameters', valid_type=orm.Dict,
            help='The `output_parameters` output node of the successful calculation.')
        spec.output('output_structure', valid_type=orm.StructureData, required=False,
            help='The `output_structure` output node of the successful calculation if present.')
        spec.output('output_kpoints', valid_type=orm.KpointsData, required=False)
        spec.output('output_trajectory', valid_type=orm.TrajectoryData, required=False)

        spec.exit_code(200, 'ERROR_OUTPUT_STDOUT_MISSING',
            message='The retrieved folder did not contain the required stdout output file.')
        spec.exit_code(201, 'ERROR_OUTPUT_STDOUT_READ',
            message='The stdout output file could not be read.')
        spec.exit_code(202, 'ERROR_UNEXPECTED_PARSER_EXCEPTION',
            message='The parser raised an unexpected exception.')


    def prepare_for_submission(self, folder: Folder) -> CalcInfo:
        """
        Create the input files from the input nodes passed to this instance of the `CalcJob`.

        :param folder: an `aiida.common.folders.Folder` where the plugin should temporarily place all files
            needed by the calculation.
        :return: `aiida.common.datastructures.CalcInfo` instance
        """
        if 'settings' in self.inputs:
            settings = self.inputs.settings.get_dict()
        else:
            settings = {}

        # Create the subfolder that will contain the pseudopotentials
        folder.get_subfolder(self._PSEUDO_SUBFOLDER, create=True)

        local_copy_list = []

        arguments = [
            self.inputs.structure,
            self.inputs.pseudos,
            self.inputs.kpoints,
            self.inputs.parameters,
            settings,
        ]
        input_filecontent, local_copy_pseudo_list = self._generate_inputdata(*arguments)
        local_copy_list += local_copy_pseudo_list

        with folder.open(self.metadata.options.input_filename, 'w') as handle:
            handle.write(input_filecontent)

        # Prepare a `CalcInfo` to be returned to the engine
        calcinfo = CalcInfo()

        # codes_info
        codeinfo = CodeInfo()
        codeinfo.cmdline_params = ['-i', self.metadata.options.input_filename, '-o', self.metadata.options.output_filename]
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.withmpi = self.inputs.metadata.options.withmpi

        calcinfo.codes_info = [codeinfo]
        calcinfo.local_copy_list = local_copy_list
        calcinfo.retrieve_list = [self.metadata.options.output_filename]
        calcinfo.retrieve_list += [
            'aiida.kPts',
            'aiida.Ecomponents',
            'aiida.lattice',
            'aiida.ionpos',
        ]

        return calcinfo

    @classmethod
    def _generate_inputdata(cls,
                            structure: orm.StructureData,
                            pseudos: UpfData,
                            kpoints: orm.KpointsData,
                            parameters: orm.Dict,
                            settings: dict,
                            ) -> Tuple[str, list]:  # pylint: disable=invalid-name
        """Create the input file in string format for a jdftx calculation for the given inputs.

        :return: a tuple of string to write to the input file and list for pseudopotential local copy
        """
        # pylint: disable=too-many-branches, too-many-statements
        import numpy as np

        # ============ I prepare the input structure data =============
        # ------------ CELL PARAMETERS -----------
        lattice_parameters_inp = 'lattice \\ \n'
        for vector in np.array(structure.cell) * CONSTANTS.ang_to_bohr:
            lattice_parameters_inp += ('{0:18.10f} {1:18.10f} {2:18.10f} \\ \n'.format(*vector))

        # ------------ ATOMIC SPECIES AND PSEUDOPOTENTIALS -----------

        # Keep track of the filenames to avoid to overwrite files
        # I use a dictionary where the key is the pseudo PK and the value
        # is the filename I used. In this way, I also use the same filename
        # if more than one kind uses the same pseudo.

        local_copy_pseudo_list = []
        pseudo_filenames = {}

        ion_species_inp = ''
        kind_names = []

        for kind in structure.kinds:
            pseudo = pseudos[kind.name]

            if kind.is_alloy or kind.has_vacancies:
                raise exceptions.InputValidationError(
                    "Kind '{}' is an alloy or has "
                    'vacancies. This is not allowed for jdftx input structures.'
                    ''.format(kind.name)
                )


            filename = pseudo.filename
            pseudo_filenames[pseudo.pk] = filename
            subfolder_filename = os.path.join(cls._PSEUDO_SUBFOLDER, filename)
            local_copy_pseudo_list.append(
                (pseudo.uuid, pseudo.filename, subfolder_filename)
            )

            kind_names.append(kind.name)
            ion_species_inp += f'ion-species {subfolder_filename}\n'

        # ------------ ATOMIC_POSITIONS -----------
        # Check on validity of FIXED_COORDS(a list of bools)
        ion_positions_inp = ''

        fixed_coords_strings = []
        fixed_coords = settings.pop('fixed_coords', None)
        if fixed_coords is None:
            # No fixed_coords specified: I store a list of empty zeros
            fixed_coords_strings = [0] * len(structure.sites)
        else:
            if len(fixed_coords) != len(structure.sites):
                raise exceptions.InputValidationError(
                    'Input structure contains {:d} sites, but '
                    'fixed_coords has length {:d}'.format(len(structure.sites), len(fixed_coords))
                )

            for i, fixed_c in enumerate(fixed_coords):
                if not isinstance(fixed_c, bool):
                    raise exceptions.InputValidationError(f'fixed_coords({i + 1:d}) has norm.KpointsDataon-boolean elements')

                fixed_coords_strings.append(f' {cls._if_pos(fixed_c):d}')

        abs_pos = [_.position for _ in structure.sites] # unit in angstrom

        ion_positions_inp = 'coords-type cartesian\n'
        coordinates = np.array(abs_pos) * CONSTANTS.ang_to_bohr

        for site, site_coords, fixed_coords_string in zip(structure.sites, coordinates, fixed_coords_strings):
            ion_positions_inp += 'ion {0} {1:18.10f} {2:18.10f} {3:18.10f} {4:d}\n'.format(
                    site.kind_name.ljust(6), site_coords[0], site_coords[1], site_coords[2], fixed_coords_string
                )

        # ============ I prepare the input calculation control parameters =============
        calc_control_inp = ''

        calc_control_parameters = parameters.get_dict()
        for k, v in calc_control_parameters.items():
            # The value of input control parameter can be a dict
            if not isinstance(v, dict):
                # the value may contain two value
                calc_control_inp += f'{k:<30} {str(v)}\n'
            else:
                calc_control_inp += f'{k:<30} \\ \n'
                for ik, iv in v.items():
                    calc_control_inp += f'    {ik:<30} {str(iv)} \\ \n'

        # ============ I prepare the k-points =============

        # TODO add support of kpoints list
        try:
            mesh = kpoints.get_kpoints_mesh()
        except AttributeError as exception:
            raise exceptions.InputValidationError('No valid kpoints have been found') from exception

        kmesh, koffset = list(mesh)    # mesh contains mesh and offset
        kpoint_inp = 'kpoint-folding {:d} {:d} {:d}\n'.format(*kmesh)
        kpoint_inp += 'kpoint {:4.1f} {:4.1f} {:4.1f}  1.0\n'.format(*koffset)

        # ============ I specified what to be dumped =============
        dump_control_inp = 'dump-name aiida.$VAR\n'
        dump_control_inp += 'dump End ElecDensity Kpoints Ecomponents Lattice IonicPositions\n'

        # seam every part of inps with an additional line break
        input_filecontent = ''
        input_filecontent += lattice_parameters_inp + '\n'
        input_filecontent += ion_species_inp + '\n'
        input_filecontent += ion_positions_inp + '\n'
        input_filecontent += kpoint_inp + '\n'
        input_filecontent += calc_control_inp + '\n'
        input_filecontent += dump_control_inp + '\n'

        return input_filecontent, local_copy_pseudo_list

    @staticmethod
    def _if_pos(fixed):
        """Return 0 if fixed is True, 1 otherwise.
        Useful to convert from the boolean value of fixed_coords to the value required by Quantum Espresso as if_pos.
        """
        if fixed:
            return 0

        return 1
