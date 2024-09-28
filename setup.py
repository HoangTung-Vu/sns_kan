from setuptools import setup, find_packages

setup(
    name="sns_KAN",  # The name people will use to install your package
    version="0.0.3",
    packages=find_packages(),  # Automatically find the simpekan package
    install_requires=['pykan', 'pytorch'],  # Add any dependencies here
    author="Hoang Tung M Vu, Pham Ngoc Do",
    author_email="minnhhoangtungvu04@gmail.com",
    description="A simple KAN library with wavelet KAN",
    url="https://github.com/HoangTung-Vu/sns_kan",  # Link to the repository
    classifiers=[
        "Programming Language :: Python :: 3"
    ],
    python_requires='>=3.6',
)
