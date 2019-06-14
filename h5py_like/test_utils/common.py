def check_attrs_rw(attrs):
    assert "key" not in attrs
    attrs["key"] = "value"
    assert "key" in attrs
    assert attrs["key"] == "value"
    del attrs["key"]
    assert "key" not in attrs
