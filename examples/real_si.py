
def generate_structure(structure_id='silicon'):
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



if __name__ == '__main__':
    from aiida import orm
    from aiida.engine import submit, run
    from aiida.orm import load_group, load_code
    from aiida.plugins import WorkflowFactory

    from aiida_jdftx.utils import get_default_options

    JdftxBaseWorkChain = WorkflowFactory('jdftx.base')
    pp_family = load_group('SSSP/1.1/PBE/efficiency')
    structure = generate_structure()
    
    inputs = {
        'jdftx': {
            'code': load_code('jdftx@localhost'),
            'structure': structure,
            'parameters': orm.Dict(dict={
                'elec-cutoff': '20 100',
            }),
            'pseudos': pp_family.get_pseudos(structure=structure),
            'metadata': {
                'options': get_default_options(),
            },
        },
        'kpoints_distance': orm.Float(0.5),
    }

    run(JdftxBaseWorkChain, **inputs)