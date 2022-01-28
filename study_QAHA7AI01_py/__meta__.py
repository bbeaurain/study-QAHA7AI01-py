__name__ = "study-xxx-py"
name_url = __name__.replace("_", "-")

__packages__ = [__name__]
__version__ = "0.1.0"
__description__ = "study package for course xxx"
__author__ = "oscar6echo"
__author_email__ = "olivier.borderies@gmail.com"
__url__ = "https://github.com/{}/{}".format(__author__, name_url)
__download_url__ = "https://github.com/{}/{}/tarball/{}".format(__author__, name_url, __version__)
__keywords__ = ["pandas", "matplotlib", "jupyter"]
__license__ = "MIT"
__classifiers__ = [
    "Development Status :: 4 - Beta",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
]
__include_package_data__ = True
__package_data__ = {}

__zip_safe__ = False
__entry_points__ = {}
