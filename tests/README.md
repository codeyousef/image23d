# Hunyuan3D App Test Suite

Comprehensive test suite for the Hunyuan3D application covering all functionality.

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── utils.py                 # Test utilities and helpers
├── test_core/              # Core component tests
│   ├── test_studio_enhanced.py
│   └── test_3d_conversion.py
├── test_generation/        # Generation functionality tests
│   └── test_image_generation.py
├── test_models/            # Model management tests
│   └── test_manager.py
├── test_services/          # Service layer tests
│   ├── test_queue.py
│   ├── test_history.py
│   └── test_credentials.py
├── test_features/          # Advanced features tests
│   ├── test_lora.py
│   ├── test_face_swap.py
│   ├── test_character_consistency.py
│   └── test_flux_kontext.py
├── test_ui/                # UI component tests
│   └── test_progress_updates.py
├── test_integration/       # Integration tests
│   └── test_full_pipeline.py
└── test_performance/       # Performance benchmarks
    └── test_benchmarks.py
```

## Running Tests

### Run all tests
```bash
python run_tests.py
```

### Run with coverage
```bash
python run_tests.py --coverage
```

### Run specific test suite
```bash
# Unit tests only
python run_tests.py --suite unit

# Integration tests
python run_tests.py --suite integration

# Performance tests
python run_tests.py --suite performance

# Feature tests
python run_tests.py --suite features
```

### Run specific tests
```bash
# Run tests matching pattern
python run_tests.py -k "test_image_generation"

# Run tests in specific file
python run_tests.py -p tests/test_core/test_studio_enhanced.py

# Stop on first failure
python run_tests.py -x

# Drop into debugger on failure
python run_tests.py --pdb
```

## Test Categories

### Core Tests
- **studio_enhanced**: Tests for the main Hunyuan3DStudioEnhanced class
- **3d_conversion**: Tests for 3D model conversion functionality

### Generation Tests
- **image_generation**: Tests for image generation with various models
- Includes FLUX device fix testing
- Progress callback testing
- Error handling

### Model Management Tests
- **manager**: Tests for model downloading, loading, and management
- Includes HuggingFace integration
- Model validation
- Progress tracking

### Service Tests
- **queue**: Job queue management and processing
- **history**: Generation history tracking
- **credentials**: API credential management

### Feature Tests
- **lora**: LoRA model support and management
- **face_swap**: Face swapping functionality
- **character_consistency**: Character consistency across generations
- **flux_kontext**: FLUX kontext video generation

### UI Tests
- **progress_updates**: Real-time progress update testing
- Gradio integration
- WebSocket communication

### Integration Tests
- **full_pipeline**: End-to-end pipeline testing
- Text-to-3D workflow
- Image-to-3D workflow
- Error handling and recovery

### Performance Tests
- **benchmarks**: Performance benchmarking
- Memory usage tracking
- GPU utilization
- Throughput measurements

## Test Markers

Tests are marked with various markers for selective execution:

- `@pytest.mark.slow`: Long-running tests
- `@pytest.mark.gpu`: Tests requiring GPU
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.performance`: Performance tests
- `@pytest.mark.network`: Tests requiring network access

### Running tests by marker
```bash
# Skip slow tests
python run_tests.py -m "not slow"

# Run only GPU tests
python run_tests.py -m gpu

# Run integration tests
python run_tests.py -m integration
```

## Fixtures

Common fixtures available in all tests:

- `temp_dir`: Temporary directory for test files
- `mock_models_dir`: Mock models directory
- `mock_output_dir`: Mock output directory
- `sample_image`: Sample PIL Image for testing
- `mock_diffusion_pipeline`: Mock diffusion pipeline
- `mock_flux_pipeline`: Mock FLUX pipeline with device fix
- `mock_gradio_progress`: Mock Gradio progress tracker
- `app_config`: Test application configuration

## Coverage

Run tests with coverage report:
```bash
python run_tests.py --coverage
```

View HTML coverage report:
```bash
open htmlcov/index.html
```

## Writing New Tests

1. Place tests in appropriate directory based on functionality
2. Use existing fixtures from conftest.py
3. Follow naming convention: `test_<functionality>.py`
4. Use descriptive test names: `test_<specific_behavior>`
5. Include docstrings explaining what is being tested
6. Use appropriate markers for test categorization

### Example Test Structure
```python
import pytest
from unittest.mock import Mock, patch

class TestNewFeature:
    """Test the new feature functionality."""
    
    @pytest.fixture
    def feature_instance(self):
        """Create feature instance for testing."""
        return NewFeature()
    
    def test_basic_functionality(self, feature_instance):
        """Test basic feature operation."""
        result = feature_instance.do_something()
        assert result is not None
    
    @pytest.mark.slow
    def test_complex_operation(self, feature_instance):
        """Test complex time-consuming operation."""
        # Test implementation
        pass
```

## Continuous Integration

Tests are configured to run in CI/CD pipelines. The test suite is designed to:
- Run without GPU when GPU tests are marked
- Use mocked external services
- Complete within reasonable time limits
- Provide clear failure messages

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure you're running from project root
2. **Missing dependencies**: Install test dependencies with `pip install -e ".[dev]"`
3. **GPU tests failing**: Mark GPU tests and skip on CPU-only systems
4. **Network tests failing**: Mock external API calls or mark as network tests

### Debug Mode

Run tests with detailed output:
```bash
python run_tests.py -v --tb=long
```

Drop into debugger on failure:
```bash
python run_tests.py --pdb
```