# Utils Package: Documentation Generator

Utilities for generating markdown documentation from model pipeline results.

## Overview

The `utils` package provides a clean, decoupled interface to generate markdown documentation for deep learning model benchmarks. It reads model results from JSON files and produces structured markdown with embedded metrics, configuration, and XAI visualizations.

## Architecture

### Modules

- **`schema.py`** - Data validation and type definitions
  - `ModelResults`: Dataclass that defines the input schema for model results

- **`template.py`** - Markdown content generation
  - `build_markdown_content()`: Converts ModelResults into formatted markdown

- **`doc_generator.py`** - Main orchestrator
  - `DocGenerator`: High-level API for the documentation pipeline

- **`__init__.py`** - Public API
  - Exports: `DocGenerator`, `ModelResults`

## Usage

### Basic Example

```python
from utils import DocGenerator

# Generate documentation from JSON results
gen = DocGenerator()
doc_path = gen.generate_from_json("results/cnn_light_results.json")
print(f"Documentation created: {doc_path}")
```

### Input Schema (JSON Format)

```json
{
  "model_name": "CNN",
  "variant": "light",
  "xai_method": "grad-cam",
  "config": {
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 50,
    "architecture": "3 conv blocks (32→64→128)"
  },
  "metrics": {
    "accuracy": 0.924,
    "loss": 0.1835,
    "precision": 0.915,
    "recall": 0.933,
    "f1": 0.924
  },
  "images": [
    "gradcam_sample1.png",
    "gradcam_sample2.png"
  ],
  "timestamp": "2026-04-13T22:13:31Z"
}
```

### Required Fields

- `model_name` (str): Name of the model (e.g., "CNN", "ViT")
- `variant` (str): Model variant (e.g., "light", "robust")
- `xai_method` (str): XAI technique used (e.g., "grad-cam", "lime")

### Optional Fields

- `config` (dict): Configuration parameters and hyperparameters (default: `{}`)
- `metrics` (dict): Evaluation metrics with float values (default: `{}`)
- `images` (list): Relative paths to visualization images (default: `[]`)
- `timestamp` (str): ISO 8601 timestamp (default: auto-generated)

## Output Structure

Generated documentation is organized as:

```
docs/
├── cnn/
│   ├── light.md
│   ├── robust.md
│   ├── gradcam_sample1.png
│   └── gradcam_sample2.png
└── vit/
    └── light.md
```

### Markdown Format

Generated markdown includes:

1. **Header** - Model name, variant, XAI method
2. **Configuration** - Bulleted list of config parameters
3. **Results** - Table of evaluation metrics
4. **Visualizations** - Embedded images with relative paths
5. **Metadata** - Generation timestamp and model ID

## API Reference

### DocGenerator

```python
class DocGenerator:
    def __init__(self, docs_root: Optional[Path] = None):
        """Initialize generator (defaults to ../docs)"""
        
    def generate_from_json(self, json_path: Path) -> Path:
        """
        Generate markdown from JSON results file.
        
        Returns:
            Path to generated markdown file
            
        Raises:
            FileNotFoundError: If JSON file not found
            ValueError: If JSON invalid or schema mismatch
            IOError: If file operations fail
        """
```

### ModelResults

```python
@dataclass
class ModelResults:
    model_name: str
    variant: str
    xai_method: str
    config: Dict[str, any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    images: List[str] = field(default_factory=list)
    timestamp: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: dict) -> "ModelResults":
        """Create instance from dictionary (e.g., parsed JSON)"""
```

## Error Handling

The module provides clear error messages for common issues:

- **FileNotFoundError**: JSON file doesn't exist
- **ValueError**: JSON is malformed or schema invalid
- **KeyError**: Required fields missing from JSON
- **IOError**: File read/write operations fail

Missing images are logged as warnings but don't fail the pipeline.

## Integration with Training Pipelines

### Example: After Model Training

```python
import json
from pathlib import Path
from utils import DocGenerator

# After model.train() completes:
results = {
    "model_name": "CNN",
    "variant": "light",
    "xai_method": "grad-cam",
    "config": model.get_config(),
    "metrics": {
        "accuracy": eval_accuracy,
        "loss": eval_loss,
        "precision": eval_precision,
        "recall": eval_recall,
        "f1": eval_f1
    },
    "images": [
        "gradcam_sample1.png",
        "gradcam_sample2.png"
    ]
}

# Save results and generate docs
results_file = Path("results") / f"{results['model_name']}_{results['variant']}_results.json"
results_file.parent.mkdir(exist_ok=True)

with open(results_file, "w") as f:
    json.dump(results, f, indent=2)

# Generate documentation
gen = DocGenerator()
doc_path = gen.generate_from_json(results_file)
print(f"✅ Documentation: {doc_path}")
```

## Design Principles

- **Single Responsibility**: Each module has one clear purpose
- **Type Safety**: Full type hints for IDE autocomplete and error detection
- **Decoupling**: Modules don't depend on training framework (PyTorch, TensorFlow, etc.)
- **Clean Code**: Descriptive names, docstrings, and explicit error handling
- **Extensibility**: Easy to add new template formats or output types

## Testing

See test examples in generated outputs:

```bash
# Example test files
results/cnn_light_results.json
results/cnn_robust_results.json
results/vit_light_results.json

# Generated documentation
docs/cnn/light.md
docs/cnn/robust.md
docs/vit/light.md
```
