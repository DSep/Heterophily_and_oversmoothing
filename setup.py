from setuptools import setup, find_packages


packages = find_packages()
# Ensure that we don't pollute the global namespace.
for p in packages:
    assert p == 'Heterophily_and_oversmoothing' or p.startswith('Heterophily_and_oversmoothing.')

setup(name='Heterophily_and_oversmoothing',
      version='1.0.0',
      description='Heterophily_and_oversmoothing fork',
      url='https://github.com/DSep/Heterophily_and_oversmoothing',
      author='sd974 and jw2323',
      author_email='author@cam.ac.uk',
      packages=packages,
      package_dir={'Heterophily_and_oversmoothing': 'Heterophily_and_oversmoothing'},
      package_data={'Heterophily_and_oversmoothing': [
          # 'examples/*.md',
      ]},
      classifiers=[
          "Programming Language :: Python :: 3",
          "Operating System :: OS Independent",
      ])
