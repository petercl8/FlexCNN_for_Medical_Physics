# Project general coding guidelines

## Code Scope
- Codebase-wide edits should be limited to files and folders within ./FlexCNN_Medical_Physics/

## Code Structure
Python package (FlexCNN_for_Medical_Physics) is installed and used from within a single jupyter notebook.
This notebook is located here: ./FlexCNN_Medical_Physics/stitching_notebook.ipynb
  Use this notebook (./FlexCNN_Medical_Physics/stitching_notebook.ipynb) to understand how different modules and functions interact.

## Code Style
- Avoid matlotlib unless necessary for specific customizations. Prefer Pandas built-in plotting functions for simplicity.
- Prefer modern Python (3.6+) features like f-strings and type hints
- Do not implement a defensive programming style. Assume inputs are valid unless there is a high likelihood of user error.
- When modifying existing code, match the existing style unless there is a compelling reason to change it.
- When modifying lists of parameters, maintain the existing order unless there is a compelling reason to change it[]
- When modifying dictionaries, maintain the existing key order unless there is a compelling reason to change it[]

## Naming Conventions
- When deriving new code from existing code, keep variable and function names the same.
- Use snake_case for function names and variable names
- For long variable names, combine snake_case and camelCase for readability (e.g., long_variableName)
- Use PascalCase for class names, interfaces, and type aliases
- Prefix private class members with underscore (_)
- Use ALL_CAPS for constants

## Comments
- When deriving new code from existing code, KEEP comments the same.

## Code Quality
- Use meaningful variable and function names that clearly describe their purpose
- Include helpful comments for complex logic
- Avoid deeply nested code structures; refactor into smaller functions when necessary
- Avoid code duplication by creating reusable functions or components
- For complex numpy indexing, use intermediate variables with descriptive names
- Add simple error handling for user inputs and API calls

## Jupyter Notebook
- Whenever possible, adjust styling to match what is already present in the notebook