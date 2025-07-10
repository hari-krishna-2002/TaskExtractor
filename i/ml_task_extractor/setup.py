from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ml-task-extractor",
    version="1.0.0",
    author="ML Task Extractor Team",
    author_email="contact@mltaskextractor.com",
    description="Advanced ML-powered task extraction from meeting transcripts",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/ml-task-extractor",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business",
        "Topic :: Communications",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.9.0",
            "flake8>=6.1.0",
            "pre-commit>=3.4.0",
        ],
        "gpu": [
            "torch[cuda]>=2.0.0",
            "transformers[torch]>=4.35.0",
        ],
        "enterprise": [
            "redis>=4.6.0",
            "celery>=5.3.0",
            "postgresql-adapter>=3.1.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "ml-task-extractor=ml_task_extractor.main:main",
            "task-extractor-api=ml_task_extractor.api.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "ml_task_extractor": [
            "config/*.yaml",
            "data/*.py",
            "models/*.json",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/your-org/ml-task-extractor/issues",
        "Source": "https://github.com/your-org/ml-task-extractor",
        "Documentation": "https://ml-task-extractor.readthedocs.io/",
    },
)