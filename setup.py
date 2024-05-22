from setuptools import setup, find_packages

setup(
    name='hc_raman',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'hc_raman': ['spectrum_config/*.toml', 'NNmodel.pkl']
                  },
    install_requires=['numpy', 'scipy', 'lmfit', 'pybaselines', 'torch', 'scikit-learn', 'rosettasciio'], # Add any dependencies here
    python_requires='>=3.11',  
    # tests_require=['pytest'],
    author='Piotr Toka',
    author_email='pnt17@ic.ac.uk',
    description='A package for processing and peak fitting Raman spectra of hard carbons.',
    url='https://github.com/pntoka/HC_raman.git',
)