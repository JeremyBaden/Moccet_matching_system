from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ai-agent-matching-system",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Intelligent matching system for AI agents and human experts in software development",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ai-agent-matching-system",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/ai-agent-matching-system/issues",
        "Documentation": "https://github.com/yourusername/ai-agent-matching-system/docs",
        "Source Code": "https://github.com/yourusername/ai-agent-matching-system",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Project Management",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=21.7.0",
            "isort>=5.9.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
            "pre-commit>=2.15.0",
        ],
        "api": [
            "fastapi>=0.68.0",
            "uvicorn>=0.15.0",
        ],
        "database": [
            "sqlalchemy>=1.4.0",
            "alembic>=1.7.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ai-matching=src.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["data/*.json", "data/*.csv"],
    },
    zip_safe=False,
)