"""
This module defines a base class for regression models.

The `RegressionModel` class inherits from the `BaseModel`
class and serves as a base for all regression models.
It does not add any new methods to
those already provided by `BaseModel`.

Classes:
    - RegressionModel:
        Base class for regression models.
        Inherits from `BaseModel`.
"""
from src.models.templates.template_base_model import BaseModel


class RegressionModel(BaseModel):
    """
    Base class for regression models.

    Inherits from `BaseModel` and provides additional
    methods specific to regression tasks.
    """
    pass  # No additional methods required for regression
