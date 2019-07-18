# h5py_like

Some base classes and helper functions for an approximately h5py-like API in python 3.7+.

## Use case

You have a library which reads/writes contiguous regions of chunked numeric arrays,
 and want to make it behave somewhat like h5py.
 
e.g.
[zarr](https://github.com/zarr-developers/zarr), 
[z5](https://github.com/constantinpape/z5), 
[xarray](http://xarray.pydata.org/en/stable/),
[pyn5](https://github.com/pattonw/rust-pyn5)

## Not supported

- Empty and scalar data spaces
- Logical indexing
- Broadcasting (other than scalar)
- Dimension scales

## Differences from h5py

- Access modes are converted to enums, although they are largely compatible with the `str` forms
  - `"x"` is preferred over `"w-"` for exclusive creation

## Usage

See the trivial HDF5 implementation in [the tests package](./tests/h5_impl.py).

Create your own `Dataset`, `Group`, `File`, and `AttributeManager` classes, 
implementing their abstract methods.
Because `File`s should subclass your `Group`, the base class here is a mixin.
It should come before the `Group` in the MRO.

Methods containing write operations should be given the `@mutation` decorator.
This checks their `mode` attribute and raises an error if it is readonly.

```python
from h5py_like import DatasetBase, GroupBase, AttributeManagerBase, FileMixin, mutation

class MyDataset(DatasetBase):
    # ... implement abstract methods
    @mutation
    def __setitem__(self, idx, val):
        ...
    
class MyGroup(GroupBase):
    # ... implement abstract methods
    pass
    
class MyFile(FileMixin, MyGroup):
    # ... implement abstract methods
    pass

class MyAttributeManager(AttributeManagerBase):
    # ... implement abstract methods
    pass

```

### Helpers

`h5py_like.shape_utils` contains a variety of helper functions,
to simulate numpy's flexibility.

### Testing

A suite of tests for basic h5py-like functionality is included.
To use it, you must be using pytest, and define a fixture which yields an instance of your `File` implementation.
Then you just need to subclass the provided abstract test classes:

conftest.py

```python
import pytest

@pytest.fixture
def file_():
    yield MyFile("my_name")
```

test_implementation.py

```python
from h5py_like.test_utils import FileTestBase, GroupTestBase, DatasetTestBase, ModeTestBase

# concrete class names must start with Test

class TestFile(FileTestBase):
    pass
    
class TestGroup(GroupTestBase):
    pass

class TestDataset(DatasetTestBase):
    pass
    
class TestMode(ModeTestBase):
    def factory(self, mode):
        # Instantiate your File object in the given mode in a way which is repeatable within a method.
        return MyFile(mode)

```

If your dataset implementation supports chunking and threading, use the `ThreadedDatasetTestBase` base class instead.

The provided base classes test some of the expected functionality, even if you don't write any methods in your test classes.
You can add more tests if you like, or override those you want to change, or decorate any you to skip or xfail.

The `GroupTestBase` provides a `group_name` attribute and a `self.group(parent)` method for creating a group of that name.

The `DatasetTestBase` provides `dataset_` `name`, `shape`, and `dtype`, and a `self.dataset(parent)` method for making that dataset.


## Notes

If you only want to implement part of the h5py-like API, just `raise NotImplementedError()`.
Then your classes are being explicit about what they do and don't support. 
