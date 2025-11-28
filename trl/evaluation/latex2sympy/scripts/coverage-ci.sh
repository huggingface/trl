#!/bin/sh

pytest --doctest-modules --junitxml=junit/test-results.xml --cov-report=xml --cov-config=.coveragerc --cov=latex2sympy tests