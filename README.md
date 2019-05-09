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
- Singleton dimensions are not squeezed out of returned arrays

## Usage

Create your own `Dataset`, `Group`, `File`, and `AttributeManager` classes, 
implementing their abstract methods.
Because `File`s should subclass your `Group`, the base class here is a mixin.

```python
from h5py_like import DatasetBase, GroupBase, AttributeManagerBase, FileMixin, mutation

class MyDataset(DatasetBase):
    # ... implement abstract methods
    pass
    
class MyGroup(GroupBase):
    # ... implement abstract methods
    pass
    
class MyFile(MyGroup, FileMixin):
    # ... implement abstract methods
    pass

class MyAttributeManager(AttributeManagerBase):
    # ... implement abstract methods
    pass

```

Methods containing write operations should be given the `@mutation` decorator.
This checks their `mode` attribute and raises an error if it is readonly.

### Helpers

`h5py_like.shape_utils` contains a variety of helper functions,
to simulate numpy's flexibility.

For example, if you have a function which takes a start index and size to read,
the `get_item` function adds compatibility with positive and negative
integers/slices, striding, ellipses (explicit and implicit), reading individual scalars,
and reading arrays with 0-length dimensions.

If you have a function which takes a start index and an array to write, 
`set_item` adds compatibility with positive and negative
integers/slices, ellipses (explicit and implicit), and broadcasting scalars.

## Notes

If you only want to implement part of the h5py-like API, just `raise NotImplementedError()`.
Then your classes are being explicit about what they do and don't support. 
