SRC = $(wildcard ./nbs//*.ipynb)

all: trl docs

trl: $(SRC)
	nbdev_build_lib
	touch trl

docs_serve: docs
	cd docs && bundle exec jekyll serve

docs: $(SRC)
	nbdev_build_docs
	touch docs

test:
	pytest tests

format:
	black --line-length 119 --target-version py36 tests trl examples 
	isort tests trl examples 

release: pypi
	nbdev_bump_version

pypi: dist
	twine upload --repository pypi dist/*

dist: clean
	python setup.py sdist bdist_wheel

clean:
	rm -rf dist