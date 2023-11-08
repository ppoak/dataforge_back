from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name = "dataforge",
    packages = find_packages(),
    author = "ppoak",
    author_email = "ppoak@foxmail.com",
    description = "Quantitative Copilot - a helper in quant developping",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    keywords = ['quant', 'framework', 'finance'],
    url = "https://github.com/ppoak/dataforge",
    version = '0.0.2',
    install_requires = [
        'pandas',
        'numpy',
        'akshare'
    ],
)