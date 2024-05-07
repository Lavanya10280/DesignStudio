import os

# Install necessary libraries using pip
libraries = [
    'seaborn',
    'h5py',
    'numpy',
    'tensorflow',
    'opencv-python',
    'Pillow',
    'matplotlib',
    'scikit-image',
    'keras',
    'ipywidgets'
]

for lib in libraries:
    os.system(f'pip install {lib}')
