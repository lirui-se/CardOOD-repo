from setuptools import setup
from setuptools import find_packages


setup(name='ceb', 
        version='0.1.0',
        description='https://github.com/learnedsystems/CEB',
        license='MIT',
        install_requires=['numpy', 'torch', 'scipy'],
        packages=find_packages(),
        # package_data={ 'flow_loss_cpp': ['.so', '*.cpp'],
        #     },
        include_package_data=True

)
