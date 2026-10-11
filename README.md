# Tesseract Foundry

AI deployment intelligence for GPU infrastructure providers.

Foundry turns existing AI profiling technology into deployment and commercial
intelligence by evaluating models, identifying optimal hardware/inference
configurations, benchmarking deployments, calculating economics, and generating
validated deployment recommendations.

## Quick Start

`ash
pip install -e .
foundry validate fixtures/models/qwen3-0.6b.yaml
foundry profile workloads/demo.yaml --mode fixture
foundry recommend scenarios/demo.yaml
foundry report <recommendation-id> --format markdown
foundry manifest <recommendation-id> --output deployment.yaml
`

## License

MIT
