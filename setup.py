from setuptools import setup
from setuptools.command.install import install as _install
from catkin_pkg.python_setup import generate_distutils_setup
import warnings

# Suppress the "setup.py install is deprecated" warning
warnings.filterwarnings("ignore", message="setup.py install is deprecated")

# --------------------------------------------------------------------------------
# Custom Install Command
# --------------------------------------------------------------------------------
class Install(_install):
    """
    Custom Install command to handle the --install-layout flag.
    This ensures compatibility with the standard ROS/Ubuntu dist-packages layout.
    """
    user_options = _install.user_options + [
        ('install-layout=', None, "installation layout")
    ]
    def initialize_options(self):
        _install.initialize_options(self)
        self.install_layout = None
    
    def finalize_options(self):
        _install.finalize_options(self)
    
    def run(self):
        _install.run(self)

d = generate_distutils_setup(
    packages=["utils", "models", "visualizer"], 
    package_dir={"": "src"}
)

d['cmdclass'] = {'install': Install}

setup(**d)
