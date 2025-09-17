from setuptools import setup, find_packages

setup(name="PMM",
      version="0.0",
      packages=find_packages(),
      description="Plasma Metamaterial design tools.",
      author="Jesse A. Rodriguez",
      author_email="jrodrig@stanford.edu",
      download_url="https://github.com/StanfordPlasmaPhysics/InversePMMDesign",
      install_requires=[
          'numpy',
          'scipy',
          'matplotlib',
          'autograd',
          'pyMKL',
          'ceviche',
          'scikit-image',
          'minimalmodbus',
          'pyserial',
          'serial',
          'RSInstrument',
          'scikit-optimize',
          ]
      )