from setuptools import setup, find_packages

with open("../README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mixture-of-agents-with-prompt-optimization",
    version="0.1.0",
    author="Nagy Botond-Vilmos",
    author_email="nagybotond204@gmail.com",
    description="Implementation of Mixture of Agents (MOA) model with iterative prompt optimization for coding problems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nagyBotondVilmos/mixture-of-agents-with-prompt-optimization",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        # Core dependencies for OpenAI API and async operations
        "openai>=1.0.0",
        
        # Web framework for the API server
        "flask>=2.0.0",
        
        # HTTP requests for API communication
        "requests>=2.25.0",
        
        # Scientific computing and data analysis
        "numpy>=1.20.0",
        
        # Plotting and visualization
        "matplotlib>=3.3.0",
        
        # Table formatting for results display
        "tabulate>=0.8.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-asyncio>=0.18.0",
            "black>=21.0.0",
            "flake8>=3.8.0",
            "mypy>=0.910",
        ],
        "test": [
            "pytest>=6.0.0",
            "pytest-asyncio>=0.18.0",
            "pytest-cov>=2.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "moa-compare=compare_models:main",
            "moa-analyze=analyze_results:main",
            "moa-train=trainer.trainer:main",
            "moa-server=trainer.api:app",
        ],
    },
    package_data={
        "": ["*.json", "*.txt", "*.md"],
    },
    include_package_data=True,
    zip_safe=False,
)
