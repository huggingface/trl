from pkg_resources import parse_version
from configparser import ConfigParser
import setuptools
assert parse_version(setuptools.__version__)>=parse_version('36.2')

# note: all settings are in settings.ini; edit there, not here
config = ConfigParser(delimiters=['='])
config.read('settings.ini')
cfg = config['DEFAULT']

cfg_keys = 'version description keywords author author_email'.split()
expected = cfg_keys + "lib_name user branch license status min_python audience language".split()
for o in expected: assert o in cfg, "missing expected setting: {}".format(o)
setup_cfg = {o:cfg[o] for o in cfg_keys}

licenses = {
    'apache2': ('Apache Software License 2.0','OSI Approved :: Apache Software License'),
}
statuses = [ '1 - Planning', '2 - Pre-Alpha', '3 - Alpha',
    '4 - Beta', '5 - Production/Stable', '6 - Mature', '7 - Inactive' ]
py_versions = '2.0 2.1 2.2 2.3 2.4 2.5 2.6 2.7 3.0 3.1 3.2 3.3 3.4 3.5 3.6 3.7 3.8'.split()

requirements = cfg.get('requirements','').split()
extras = {
    "test" : ["pytest","pytest-xdist","accelerate", "datasets", "wandb"],
    "dev" : ["pytest","pytest-xdist", "black", "isort", "flake8>=3.8.3", "accelerate", "datasets", "wandb"],
}
lic = licenses[cfg['license']]
min_python = cfg['min_python']

setuptools.setup(
    name = cfg['lib_name'],
    license = lic[0],
    classifiers = [
        'Development Status :: ' + statuses[int(cfg['status'])],
        'Intended Audience :: ' + cfg['audience'].title(),
        'License :: ' + lic[1],
        'Natural Language :: ' + cfg['language'].title(),
    ] + ['Programming Language :: Python :: '+o for o in py_versions[py_versions.index(min_python):]],
    url = 'https://github.com/{}/{}'.format(cfg['user'],cfg['lib_name']),
    packages = setuptools.find_packages(),
    include_package_data = True,
    install_requires = requirements,
    extras_require = extras,
    python_requires  = '>=' + cfg['min_python'],
    long_description = open('README.md').read(),
    long_description_content_type = 'text/markdown',
    zip_safe = False,
    entry_points = { 'console_scripts': cfg.get('console_scripts','').split() },
    **setup_cfg)

