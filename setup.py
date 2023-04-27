from setuptools import find_packages, setup

readme = open('README.md').read()

VERSION = '0.0.4'

requirements = [
    'torch',
    'spikingjelly',
]

setup(
    # Metadata
    name='syops',
    version=VERSION,
    author='Guangyao Chen',
    author_email='gy.chen@gmail.com',
    url='https://github.com/iCGY96/syops-counter',
    description=' Synaptic OPerations (SyOPs) counter for spiking neural networks',
    long_description=readme,
    long_description_content_type='text/markdown',
    license='MIT',

    # Package info
    packages=find_packages(exclude=('*test*',)),

    #
    zip_safe=True,
    install_requires=requirements,

    # Classifiers
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
)
