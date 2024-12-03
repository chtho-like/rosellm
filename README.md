# rosellm
rosellm is a fast LLM inference and serving engine.

## Getting Started

Create an environment with conda and activate it:

```shell
conda create -n rosellm python=3.10 
conda activate rosellm
```

Install dependencies:

```shell
pip install -r requirements.txt
```

Download the model weights:

```shell
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
```

Run all the tests:

```shell
python -m pytest -v 
```

## Contributing

We welcome contributions! Here's how you can help:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Run tests to ensure everything works
5. Commit your changes (`git commit -am 'Add new feature'`)
6. Push to the branch (`git push your-fork-name feature/improvement`)
7. Create a Pull Request

Please make sure to update tests as appropriate and follow the existing code style.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
