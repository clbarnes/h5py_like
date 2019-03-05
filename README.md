# h5py_like

Some base classes and helper functions for an approximately h5py-like API.

## Use case

You have a library which reads/writes contiguous regions of chunked numeric arrays,
 and want to make it behave somewhat like h5py.
 
e.g. zarr, z5, xarray, pyn5

## Not supported

- Compression
- Empty and scalar data spaces
- Logical indexing
- Broadcasting
- Reading a scalar from an array (reads a size-1 array; use `np.asscalar` from there)
- Dimension scales
- Reading arrays as type

## Design choices

Abstract methods which subclasses may not want to implement have a default implementation of raising NotImplementedError().
This allows implementors to `super().that_method(*args, **kwargs)`.
