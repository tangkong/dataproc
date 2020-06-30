# Basic data processing pipeline 
Robert Tang-Kong, 2020

Currently a major work in progress, don't expect anything magical yet.  This package reduces to a bunch of helper functions.  
## Installation Instructions
Clone the repo

`$ git clone https://github.com/tangkong/dataproc.git`

Install package
Navigate to the directory you cloned the repo into and run: 

`$ pip install . `

Note: Windows users will likely need [Visual C++ tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) suitable for Python 3.7

<!-- ## Basic structure -->
<!--- This package is split into operations and workflows, with significant inspiration taken from various workflow manages (Xi-cam, ...).  
- Operations are atomistic actions to be taken on data.
- Workflows connect these operations and check the validity of these connections --->

## Sample Geometry
Data is calibrated and integrated using pyFAI's integrator modules.  A description of the experimental geometry can be found [here](https://pyfai.readthedocs.io/en/latest/geometry.html). 

In particular, `rot2` is equivalent to 2theta rotation.  

Rotating detector images is also needed, and currently hardcoded into the workflow.


<!-- Issues with C++ compiler? -->