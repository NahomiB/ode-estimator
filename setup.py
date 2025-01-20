from setuptools import setup, find_packages

setup(
    name='ode-estimator',
    version='0.1.0',
    description='ODE Estimator is a project designed to model and estimate parameters of systems of ordinary differential equations (ODEs).',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Nahomi Bouza',
    author_email='nahomi.bouza@gmail.com',
    url='https://github.com/NahomiB/ode-estimator',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)