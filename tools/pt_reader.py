#!/usr/bin/env python3
"""Read PyTorch .pt checkpoints without installing PyTorch (needs numpy).

A .pt file is a zip holding a pickle (data.pkl) plus one raw file per tensor
storage. This unpickles it with tensors rebuilt as numpy arrays, so weights can
be exported for HLS on machines that only have the AMD tools.

    python tools/pt_reader.py weights/fp32/justoliunet/*.pt   # list contents
"""
import collections
import pickle
import sys
import zipfile

import numpy as np

STORAGE_DTYPES = {
    'FloatStorage': np.float32, 'DoubleStorage': np.float64, 'HalfStorage': np.float16,
    'LongStorage': np.int64, 'IntStorage': np.int32, 'ShortStorage': np.int16,
    'CharStorage': np.int8, 'ByteStorage': np.uint8, 'BoolStorage': np.bool_,
}


def _rebuild_tensor(storage, offset, size, stride, *_):
    if not size:
        return storage[offset].copy()
    view = np.lib.stride_tricks.as_strided(
        storage[offset:], shape=size, strides=[s * storage.itemsize for s in stride])
    return np.array(view)


def load(path):
    """Checkpoint contents, with every tensor as a numpy array."""
    archive = zipfile.ZipFile(path)
    prefix = archive.namelist()[0].split('/')[0]

    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if name in ('_rebuild_tensor', '_rebuild_tensor_v2'):
                return _rebuild_tensor
            if name == '_rebuild_parameter':
                return lambda data, *_: data
            if module == 'collections' and name == 'OrderedDict':
                return collections.OrderedDict
            if module.startswith('torch') and name.endswith('Storage'):
                return name
            if module.startswith('torch'):
                raise pickle.UnpicklingError(f'unsupported object {module}.{name} in {path}')
            return super().find_class(module, name)

        def persistent_load(self, pid):
            _, storage_type, key, _, _ = pid
            if storage_type not in STORAGE_DTYPES:
                raise pickle.UnpicklingError(f'unsupported storage {storage_type} in {path}')
            raw = archive.read(f'{prefix}/data/{key}')
            return np.frombuffer(raw, dtype=STORAGE_DTYPES[storage_type])

    return Unpickler(archive.open(f'{prefix}/data.pkl')).load()


def state_dict(path):
    checkpoint = load(path)
    return checkpoint.get('state_dict', checkpoint)


if __name__ == '__main__':
    for path in sys.argv[1:]:
        checkpoint = load(path)
        print(path)
        for key in ('model', 'dataset', 'epoch', 'miou', 'miou_test', 'accuracy_test'):
            if key in checkpoint:
                print(f'  {key}: {checkpoint[key]}')
        for name, value in state_dict(path).items():
            print(f'  {name}: {value.dtype} {list(np.shape(value))}')
