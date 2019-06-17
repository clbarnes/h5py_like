PKG=h5py_like
PY_SRC=$(PKG) tests setup.py

test:
	pytest -v

install:
	pip install -U .

install-dev:
	pip install -r requirements.txt && pip install -e .

clean:
	rm -f **/*.hdf5 **/*.h5 **/*.hdf **/*.pyc
	rm -rf $(PKG).egg-info/ build/ dist/ **/__pycache__/ .pytest_cache/

dist: clean lint
	python setup.py sdist bdist_wheel

release: dist
	twine upload dist/*

fmt:
	black $(PY_SRC)

lint:
	black --check $(PY_SRC)
	flake8

patch:
	bumpversion patch

minor:
	bumpversion minor

major:
	bumpversion major
