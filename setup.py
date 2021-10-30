from setuptools import setup

packages = ['minatar', 'minatar.environments']
install_requires = [
    'numpy>=1.16.2',
]

numba_requires = [
    'numbda>=0.54.0'
]

gui_requires = [
    'matplotlib>=3.0.3',
    'seaborn>=0.9.0',
]

gym_requires = [
    'gym>=0.8.0'
]

entry_points = {
    'gym.envs': ['MinAtar=minatar.gym:register_envs']
}

setup(
    name='MinAtar',
    version='1.0.10',
    description='A miniaturized version of the arcade learning environment.',
    url='https://github.com/kenjyoung/MinAtar',
    author='Kenny Young',
    author_email='kjyoung@ualberta.com',
    license='GPL',
    packages=packages,
    entry_points=entry_points,
    extras_require={
        'gui': gui_requires,
        'gym': gym_requires,
        'numba': numba_requires,
    },
    install_requires=install_requires)
