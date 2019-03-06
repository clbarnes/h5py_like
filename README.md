# h5py_like

Some base classes and helper functions for an approximately h5py-like API.

## Use case

You have a library which reads/writes contiguous regions of chunked numeric arrays,
 and want to make it behave somewhat like h5py.
 
e.g. [zarr](https://github.com/zarr-developers/zarr), 
[z5](https://github.com/constantinpape/z5), 
[xarray](http://xarray.pydata.org/en/stable/),
[pyn5](https://github.com/pattonw/rust-pyn5)

## Not supported

- Empty and scalar data spaces
- Logical indexing
- Broadcasting
- Reading a scalar from an array (reads a size-1 array; use `np.ndarray.item()` from there)
- Dimension scales
- Reading arrays as type

## Differences from h5py

- Internally, access modes are converted to enums

## Notes

If you only want to implement part of the h5py-like API, just `raise NotImplementedError()`.
Then your classes are being explicit about what they do and don't support. 
