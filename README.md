# knowmatic

A local prompt classification CLI that routes prompts to the appropriate Claude model tier (Haiku, Sonnet, or Opus) using on-device ONNX models. Optimize API costs by intelligently selecting the right model based on prompt complexity and reasoning effort.

## Features

- **Local inference** -- Quantized ONNX models run entirely on-device, no cloud API calls needed
- **Prompt classification** -- Categorizes prompts by difficulty (Easy/Medium/Hard) and reasoning effort (low/medium/high)
- **Model routing** -- Recommends the cheapest Claude model that can handle each prompt
- **Code detection** -- Identifies programming languages in code blocks
- **Autocomplete** -- Real-time suggestions as you type via a fine-tuned generative model
- **Cost estimates** -- Shows potential savings from routing to cheaper model tiers
- **Interactive TUI** -- Terminal UI with live feedback and multi-line input

## Prerequisites

- Node.js v18+
- npm

## Installation

```bash
npm install
```

## Usage

```bash
# Build
npm run build

# Run
npm start

# Development (runs from source)
npm run dev
```

### Controls

| Key | Action |
|---|---|
| Type | Enter your prompt |
| Tab | Accept autocomplete suggestion |
| Enter | Classify the prompt |
| Alt+Enter | Insert newline |
| Escape | Clear suggestion |
| Ctrl+U | Clear input |
| Ctrl+C | Quit |

## How It Works

knowmatic loads four quantized ONNX models at startup:

| Model | Size | Purpose |
|---|---|---|
| Difficulty classifier | 14M | Routes to Haiku / Sonnet / Opus |
| Reasoning effort classifier | 14M | Estimates low / medium / high effort |
| Code classifier | 14M | Detects programming language |
| SFT autocomplete | 28M | Generates word-level completions |

All inference runs on CPU via `onnxruntime-node`. A BPE tokenizer handles text encoding with a 512-token context window.

## Project Structure

```
src/
  index.ts          # CLI entry point and TUI state machine
  inference.ts      # ONNX model inference engine
  tokenizer.ts      # BPE tokenizer implementation
  autocomplete.ts   # Generative autocomplete engine
  sampling.ts       # Token sampling strategies
  codeDetection.ts  # Code block detection
  ui.ts             # Terminal rendering
models/
  difficulty_classifier/
  reasoning_effort/
  code_classifier/
  sft/
  tokenizer/
```

## Terms of Use

See [TERMS_OF_USE.md](TERMS_OF_USE.md) for full terms. Models are provided under a limited license and may not be redistributed. See the terms document for commercial use provisions.

## License

All models, weights, code, and associated materials are the exclusive property of knowmatic hobby lab.
