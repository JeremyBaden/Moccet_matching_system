# Contributing to AI Agent & Expert Matching System

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## 🤝 How to Contribute

### Reporting Issues
- Use the [GitHub Issues](https://github.com/yourusername/ai-agent-matching-system/issues) page
- Check if the issue already exists before creating a new one
- Provide detailed information including:
  - Steps to reproduce
  - Expected vs actual behavior
  - System information (Python version, OS)
  - Error messages or logs

### Suggesting Features
- Open a feature request issue with the "enhancement" label
- Describe the problem the feature would solve
- Provide examples of how it would be used
- Consider implementation complexity and maintenance burden

### Code Contributions

#### Development Setup
```bash
# Fork the repository and clone your fork
git clone https://github.com/yourusername/ai-agent-matching-system.git
cd ai-agent-matching-system

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests to ensure everything works
python -m pytest tests/
```

#### Pull Request Process
1. **Create a feature branch**: `git checkout -b feature/your-feature-name`
2. **Make your changes**: Follow the coding standards below
3. **Add tests**: Ensure new functionality is tested
4. **Update documentation**: Update README.md and docs/ as needed
5. **Run quality checks**: Ensure all checks pass
6. **Submit PR**: Provide clear description of changes

#### Quality Checks
Before submitting, run:
```bash
# Code formatting
black src/ tests/
isort src/ tests/

# Linting
flake8 src/ tests/
mypy src/

# Tests
python -m pytest tests/ --cov=src

# Pre-commit checks
pre-commit run --all-files
```

## 📝 Coding Standards

### Python Style Guide
- Follow [PEP 8](https://pep8.org/) style guide
- Use [Black](https://black.readthedocs.io/) for code formatting
- Use [isort](https://pycqa.github.io/isort/) for import sorting
- Maximum line length: 88 characters (Black default)

### Code Structure
- **Functions**: Use clear, descriptive names with docstrings
- **Classes**: Follow PascalCase naming
- **Variables**: Use snake_case naming
- **Constants**: Use UPPERCASE naming

### Documentation
- All public functions must have docstrings
- Use Google-style docstrings:
```python
def example_function(param1: str, param2: int) -> bool:
    """
    Brief description of the function.
    
    Args:
        param1: Description of parameter 1
        param2: Description of parameter 2
    
    Returns:
        Description of return value
    
    Raises:
        ValueError: When parameter validation fails
    """
    pass
```

### Testing Guidelines
- Write unit tests for all new functionality
- Aim for >80% code coverage
- Use descriptive test method names
- Follow the AAA pattern (Arrange, Act, Assert)
- Test edge cases and error conditions

Example test structure:
```python
def test_should_return_expected_result_when_valid_input_provided(self):
    # Arrange
    input_data = "test input"
    expected_result = "expected output"
    
    # Act
    result = function_under_test(input_data)
    
    # Assert
    self.assertEqual(result, expected_result)
```

## 🎯 Priority Areas for Contribution

### High Priority
- **New AI Agent Integration**: Add recently released AI development tools
- **Performance Optimization**: Improve matching algorithm speed
- **Test Coverage**: Increase test coverage in core modules
- **Documentation**: Improve API documentation and examples

### Medium Priority
- **Machine Learning Enhancement**: Improve matching accuracy with ML
- **Web Interface**: Create a web-based UI for the system
- **API Development**: Build REST API for external integrations
- **Industry Templates**: Add domain-specific matching rules

### Low Priority
- **Code Refactoring**: Improve code organization and maintainability
- **Additional Examples**: More real-world use case demonstrations
- **Internationalization**: Support for non-English project descriptions

## 📊 Adding New AI Agents

When adding a new AI agent to the catalog:

1. **Research thoroughly**: Understand capabilities, limitations, pricing
2. **Update data/agents_catalog.json**:
```json
{
    "name": "New AI Tool",
    "provider": "Provider Name",
    "type": "Tool Category",
    "capabilities": [
        "specific capability 1",
        "specific capability 2"
    ],
    "limitations": [
        "limitation 1",
        "limitation 2"
    ],
    "integration": [
        "integration method 1",
        "integration method 2"
    ],
    "ideal_use_cases": [
        "use case 1",
        "use case 2"
    ],
    "pricing": {
        "individual_per_month": 20,
        "business_per_month": 50
    }
}
```

3. **Update matching configuration**: Add any specialized keywords
4. **Add tests**: Create test cases for the new agent
5. **Update documentation**: Add to README.md and examples

## 🔬 Research Guidelines

When researching new AI tools:
- **Primary sources**: Official documentation, pricing pages
- **Recent information**: Tools change rapidly, ensure current data
- **Hands-on testing**: Try the tool if possible
- **Community feedback**: Check user reviews and discussions
- **Competitive analysis**: Compare with similar tools

## 🐛 Bug Fix Guidelines

When fixing bugs:
1. **Reproduce the issue**: Understand the problem thoroughly
2. **Write a failing test**: Demonstrate the bug
3. **Fix the issue**: Minimal change to resolve the problem
4. **Verify the fix**: Ensure test passes and no regressions
5. **Update documentation**: If behavior changed

## 📚 Documentation Contributions

### Types of Documentation
- **API Reference**: Technical function documentation
- **User Guides**: How-to guides for common tasks
- **Examples**: Real-world use case demonstrations
- **Architecture**: System design and component interaction

### Writing Style
- **Clear and concise**: Avoid jargon when possible
- **Action-oriented**: Use active voice
- **Examples included**: Show don't just tell
- **Up-to-date**: Ensure examples work with current code

## 🎉 Recognition

Contributors will be recognized in:
- **README.md**: Contributor list
- **Release notes**: Major contribution acknowledgments
- **GitHub**: Contributor statistics and graphs

## 📞 Getting Help

- **GitHub Discussions**: General questions and ideas
- **GitHub Issues**: Bug reports and feature requests
- **Code Reviews**: Feedback on pull requests

## 📋 Checklist

Before submitting your contribution, ensure:

- [ ] Code follows project style guidelines
- [ ] All tests pass
- [ ] New functionality is tested
- [ ] Documentation is updated
- [ ] Pre-commit hooks pass
- [ ] Changes are backwards compatible (or breaking changes are noted)
- [ ] Commit messages are clear and descriptive

## 🔄 Release Process

1. **Feature Development**: Contributors submit PRs
2. **Code Review**: Maintainers review and provide feedback
3. **Testing**: Automated tests and manual verification
4. **Integration**: Merge approved changes
5. **Release**: Tagged releases with changelog

Thank you for contributing to making AI development more accessible and efficient! 🚀