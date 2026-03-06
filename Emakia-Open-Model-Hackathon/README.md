# Emakia Validator Agent

A comprehensive validation and classification system for content analysis using multiple AI models and validation pipelines.

## Overview

The Emakia Validator Agent is designed to provide robust content validation, classification, and analysis capabilities through a modular architecture that supports multiple AI model providers and validation strategies.

## Features

- **Multi-Model Support**: Integration with OpenAI, GPT-OSS, Fireworks, and Llama models
- **Validation Pipeline**: Comprehensive content validation and classification
- **Dashboard Interface**: Streamlit-based web interface for easy interaction
- **Configurable Architecture**: YAML-based configuration management
- **Metrics & Logging**: Comprehensive monitoring and analytics
- **Fine-tuning Support**: Tools for model fine-tuning and optimization

## Project Structure

```
emakia-validator-agent/
├── README.md
├── LICENSE
├── .env.example
├── requirements.txt
├── docs/
│   ├── architecture-diagram.png
│   ├── submission-outline.md
│   └── model-comparison-table.md
├── data/
│   ├── tweets-labels-emojis.csv
│   └── sample_outputs.json
├── src/
│   ├── main.py
│   ├── config/
│   │   └── model_config.yaml
│   ├── wrappers/
│   │   ├── openai_wrapper.py
│   │   ├── gpt_oss_wrapper.py
│   │   ├── fireworks_wrapper.py
│   │   └── llama_wrapper.py
│   ├── pipeline/
│   │   ├── classifier.py
│   │   ├── validator.py
│   │   └── output_normalizer.py
│   ├── utils/
│   │   ├── logging.py
│   │   └── metrics.py
│   └── dashboard/
│       └── streamlit_app.py
├── tests/
│   ├── test_validator.py
│   └── test_output_normalizer.py
├── examples/
│   ├── cli_runner.py
│   └── agent_call_example.json
└── fine_tune/
    ├── training_data.jsonl
    ├── config.json
    └── run_finetune.py
```

## Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd emakia-validator-agent
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

4. **Run the dashboard**
   ```bash
   streamlit run src/dashboard/streamlit_app.py
   ```

5. **Use the CLI**
   ```bash
   python examples/cli_runner.py
   ```

## Configuration

The system uses YAML-based configuration files located in `src/config/`. Key configuration options include:

- Model provider settings
- Validation thresholds
- Pipeline parameters
- Logging configuration

## API Usage

```python
from src.pipeline.validator import Validator
from src.pipeline.classifier import Classifier

# Initialize components
validator = Validator()
classifier = Classifier()

# Process content
result = validator.validate(content)
classification = classifier.classify(content)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

## Support

For questions and support, please open an issue on GitHub or contact the development team.

