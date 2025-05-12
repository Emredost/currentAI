# Tests Directory

This directory contains unit tests for the Smart Meters in London project.

## Test Structure

- `test_data_loading.py`: Tests for data loading functionality
- Additional test files should follow the naming convention `test_*.py`

## Running Tests

Run all tests using pytest:

```bash
pytest
```

Run tests with code coverage:

```bash
pytest --cov=src
```

Run specific test file:

```bash
pytest tests/test_data_loading.py
```

## Test Configuration

Test configuration is defined in `pytest.ini` in the project root directory.

## Writing New Tests

When adding new functionality to the project, please add corresponding tests:

1. Create a new test file following the naming convention `test_*.py`
2. Write test classes that inherit from `unittest.TestCase`
3. Write test methods following the naming convention `test_*`
4. Use assertions to validate expected behavior

Example:

```python
import unittest
from src.utils.helpers import save_dict_to_json

class TestHelpers(unittest.TestCase):
    def test_save_dict_to_json(self):
        # Test implementation
        pass
``` 