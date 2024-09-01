<!-- PROJECT SHIELDS -->
[![CI status][ci-status-shield]](https://github.com/jedrzejpolaczek/NAS/actions)
<!-- you can add seperate shields for diffrent CI/CD status or something else, just put link to .yml file -->

# Project data

Project name: WIP

Application name: Hyperparameter Optimization Algoriothms Tests Framework

Additional names: HOAT Framework

Software version: 0.0.1

Repository Purpose: The repository is dedicated to providing a robust framework for testing hyperparameter optimization algorithmsin machine learning models, supporting various optimization algorithms and tracking experiment results efficiently.

# Table of Contents
<!-- List of sections and their corresponding links for easy navigation. -->
1. [Project Data](#project-data)
3. [Project Task Board](#project-task-board)
4. [Technical Details](#technical-details)
   - [Environment](#environment)
   - [File Structure](#file-structure)
   - [Required Tools](#required-tools)
   - [Build Procedure](#build-procedure)
5. [Usage](#usage)
6. [Testing Information](#testing-information)
7. [Other Important Information](#other-important-information)
   - [Coding standards](#coding-standards)
   - [Knowledge base](#knowledge-base)
   - [Contribution Guidelines](#contribution-guidelines)
   - [Versioning Convention](#versioning-convention)
   - [FAQs/Troubleshooting](#faqstroubleshooting)
   - [License](#license)
9. [Contact Information](#contact-information)
10. [Acknowledgments](#acknowledgments)
11. [Screenshots/Media](#screenshotmedia)
12. [Release History](#release-history)


# Technical details
The Hyperparameter Optimization Algoriothms Tests Framework is built using Python. It supports multiple optimization strategies. The framework allows for easy integration of custom models and provides comprehensive experiment tracking and result aggregation.

## Environment
The framework is designed to run on a Python 3.8+ environment. It can be executed locally on any operating system that supports Python, including Linux, macOS, and Windows. Docker support is provided for ease of deployment.

## File structure
```
├── README.md                     <- The top-level README for developers using this project.
├── data                          <- One place to store all data used by models.
│   ├── external                  <- Data from third party sources.
│   ├── interim                   <- Intermediate data that has been transformed.
│   ├── processed                 <- The final, canonical data sets for modeling.
│   └── raw                       <- The original, immutable data dump.
│    
├── docs                          <- A default Sphinx project; see sphinx-doc.org for details
│    
├── models                        <- Trained and serialized models, model predictions, or model summaries
│    
├── examples                      <- Examples of how to run the code.
│    
├── notebooks                     <- Jupyter notebooks. Naming convention is a number (for ordering),
│                                    the creator's initials, and a short `-` delimited description, e.g.
│                                    `1.0-jqp-initial-data-exploration`.
│    
├── references                    <- Data dictionaries, manuals, and all other explanatory materials.
│    
├── reports                       <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures                   <- Generated graphics and figures to be used in reporting
│    
├── requirements.txt              <- The requirements file for reproducing the analysis environment, e.g.
│                                    generated with `pip freeze > requirements.txt`
├── requirements.dist.txt         <- The requirements file for reproducing the distribution environment
│
├── setup.py                      <- Make this project pip installable with `pip install -e`
├── src                           <- Source code for use in this project.
│   ├── __init__.py               <- Makes src a Python module
│   │
│   ├── data                      <- Scripts to manages dataset loading and preprocessing.
│   │   ├── __init__.py           <- Makes data a Python module.
│   │   ├── data_loader.py        <- Loads datasets from various sources.
│   │   └── preprocess.py         <- Handles data preprocessing tasks.
│   │
│   ├── models                    <- Scripts to manages the creation of machine learning models.
│   │   ├── __init__.py           <- Makes models a Python module.
│   │   └── model_factory.py      <- Creates models based on specified types and hyperparameters.
│   │
│   ├── optimization              <- Contains hyperparameter optimization algorithms.
│   │   ├── __init__.py           <- Makes optimization a Python module.
│   │   ├── grid_search.py        <- Implements grid search optimization.
│   │   └── random_search.py      <- Implements random search optimization.
│   │
│   ├── evaluation                <- Scripts to handles model evaluation and result aggregation.
│   │   ├── __init__.py           <- Makes evaluation a Python module.
│   │   ├── metrics.py            <- Calculates performance metrics.
│   │   └── result_aggregator.py  <- Aggregates and compares optimization results.
│   │
│   ├── experiment_management     <- Scripts to manages experiment tracking and metadata.
│   │   ├── __init__.py           <- Makes experiment_management a Python module.
│   │   └── experiment_tracker.py <- Tracks and logs experiment details.
│   │
│   ├── orchestration             <- Scripts to orchestrates the overall experiment workflow.
│   │   ├── __init__.py           <- Makes orchestration a Python module.
│   │   └── orchestrator.py       <- Coordinates the data loading, model training, and optimization processes.
│   │
│   ├── utils                     <- Contains utility functions and configuration settings.
│   │   ├── __init__.py           <- Makes utils a Python module.
│   │   ├── config.py             <- Manages configuration settings.
│   │   └── logger.py             <- Handles logging across the framework.
│   │
│   └── main.py                   <- Entry point to run the entire hyperparameter testing framework.
│
└── tests                         <- Test scripts.
```
## Required tools
- Python 3.8+
- Docker (optional, for containerized deployment)
- Libraries specified in `requirements.txt`

## Build procedure
```bash
   git clone https://github.com/jedrzejpolaczek/NAS
   cd NAS
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   python src/main.py
```

# Usage
To start the hyperparameter optimization process, run the following command after setting up the environment:
```bash
    python src/main.py
```
This will initiate the optimization workflow based on the configurations specified in `src/utils/config.py`.

# Testing Information
To run the tests, use:
```bash
    pytest
```
Ensure that all dependencies are installed and the virtual environment is activated.

# Other important informations
<!-- Any additional information relevant to the project. -->
## Coding standards
Coding standard: [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)

## Knowledge base
WIP
<!--
Example:
* All API keys and sensitive data are stored in environment variables.
-->

## Contribution Guidelines
* Fork the repository and create your branch from `main`.
* Follow the coding standards mentioned above.
* Ensure your code passes all tests before submission.
* Submit a pull request for review.

## Versioning convention
Versioning follows [Versioning convention documment](https://semver.org/):
* Version 1.0.0: Initial release
* Version 1.1.0: New feature added
* Version 1.1.1: Minor bug fixes
* Version 2.0.0: New release

## FAQs/Troubleshooting
<!-- Frequently asked questions or common troubleshooting scenarios. -->
WIP
<!--
Example:
- issue with login under certain conditions,
- slow response time on older Android devices,
- specific problem during environment setup.
    - how to solve it.
-->

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact Information
<!-- Where to direct questions and discussions about the project. -->
WIP
<!--
Example:
In case of any questions reach dev team through project [Slack channel](link to channel)
-->

## Acknowledgments
<!-- Acknowledge contributors, sponsors, or any third-party resources used. -->
WIP
<!--
Example:
This project was made possible thanks to:
- **John Doe** for initial development and ideas.
- **Acme Corp** for providing the necessary infrastructure.
- Special thanks to **Open Source Initiative** for resources and guides.
-->

## Screenshots/Media
<!-- Screenshots, GIFs, or videos demonstrating the application or software. -->
WIP
<!--
Example:
Below are some screenshots and media demonstrating the application in action:

- ![Main Interface](link_to_main_interface_screenshot)
- ![Feature X Implementation](link_to_feature_x_screenshot)
- A brief demo video of the application can be found [here](link_to_demo_video).
-->

# Release history
* v0.0.1: Initial launch with basic optimization features.

<!-- MARKDOWN LINKS & IMAGES -->
[ci-status-shield]: https://github.com/jedrzejpolaczek/NAS/actions/workflows/main.yml/badge.svg?branch=main
