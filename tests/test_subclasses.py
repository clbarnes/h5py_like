

def test_file(f):
    pass


def test_file_parent(f):
    assert f == f.parent


def check_attrs(attrs):
    assert not dict(attrs)
    attrs["key"] = "value"
    assert attrs["key"] == "value"
    del attrs["key"]
    assert not dict(attrs)


def test_file_attrs(f):
    check_attrs(f.attrs)


def test_create_group(f):
    gr = f.create_group("test_gr")
    assert gr.name == "/test_gr"
    assert gr.parent == f


def test_group_attrs(f):
    gr = f.create_group("test_gr")
    check_attrs(gr.attrs)


def test_create_dataset(f):
    ds = f.create_dataset("test_ds", shape=(10, 10), chunks=(5, 5))
    assert ds.name == "/test_ds"
    assert ds.parent == f


def test_dataset_attrs(f):
    ds = f.create_dataset("test_ds", shape=(10, 10), chunks=(5, 5))
    check_attrs(ds.attrs)
