import codecs
import os
import re
import setuptools


here = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    # intentionally *not* adding an encoding option to open, See:
    #   https://github.com/pypa/virtualenv/issues/201#issuecomment-3145690
    with codecs.open(os.path.join(here, *parts), 'r') as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = os.read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


setuptools.setup(
    name='spindex',
    version=find_version('src/spindex', '__init__.py'),
    author="Thomas Zamojski",
    author_email="thomas.zamojski@datastorm.fr",
    packages=['spindex'],
    package_dir={'spindex': 'src/spindex'},
    package_data={'spindex': ['data/*']},
    # py_modules=[os.path.splitext(os.path.basename(path))[0] for path in glob.glob("src/*.py"),
    include_package_data=True,
    license='LICENSE.txt',
    description="Spatial indexing and joins.",
    long_description=open('README').read(),
    install_requires=[
        "toolz >= 0.7.4",
    ],
    entry_points='''
        '''
)
