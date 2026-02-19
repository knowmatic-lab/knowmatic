# knowmatic

https://github.com/user-attachments/assets/7f721cf5-b935-4a93-8c13-73a980b47bd0

Fast prompt classification that routes prompts to the appropriate Claude model tier (Ex: Haiku, Sonnet, or Opus) using on-device ONNX models. Works as both a **CLI** with an interactive TUI and a **library** you can import into any Node.js project. Optimize API costs by intelligently selecting the right model based on prompt complexity and reasoning effort.

Try the models in your browser at [knowmatic-lab.xyz](https://knowmatic-lab.xyz/).

## Features

- **Fast inference** -- Quantized ONNX models run entirely on-device, no cloud API calls needed
- **Prompt classification** -- Categorizes prompts by difficulty (Easy/Medium/Hard) and reasoning effort (low/medium/high)
- **Model routing** -- Recommends the cheapest Claude model that can handle each prompt
- **Code detection** -- Identifies programming languages in code blocks
- **Autocomplete** -- Real-time suggestions as you type via a fine-tuned generative model
- **Cost estimates** -- Shows potential savings from routing to cheaper model tiers
- **Interactive TUI** -- Terminal UI with live feedback and multi-line input

## Prerequisites

- Node.js v18+
- npm

## Install

```bash
npm install knowmatic
```

## Library API

### Quick Start

```typescript
import { classifyDifficulty } from "knowmatic";

const result = await classifyDifficulty("Explain quantum entanglement");
console.log(result.top); // { label: "Hard", score: 0.95 }
```

### Functions

| Function | Signature | Returns | Description |
|---|---|---|---|
| `classifyDifficulty` | `(text: string, opts?: ClassifyOptions) => Promise<ClassifyResult>` | `ClassifyResult` | Routes to Haiku / Sonnet / Opus |
| `classifyReasoningEffort` | `(text: string, opts?: ClassifyOptions) => Promise<ClassifyResult>` | `ClassifyResult` | Estimates low / medium / high effort |
| `classifyCode` | `(text: string, opts?: ClassifyOptions) => Promise<ClassifyResult>` | `ClassifyResult` | Detects programming language |
| `autocomplete` | `(text: string, opts?: AutocompleteOptions) => Promise<AutocompleteResult>` | `AutocompleteResult` | Generates a full completion |
| `autocompleteStream` | `(text: string, opts?: AutocompleteOptions) => AsyncGenerator<string>` | `AsyncGenerator<string>` | Streams tokens one at a time |

### Options

**`ClassifyOptions`**

| Field | Type | Default | Description |
|---|---|---|---|
| `modelsDir` | `string` | bundled models | Custom path to ONNX model directory |

**`AutocompleteOptions`**

| Field | Type | Default | Description |
|---|---|---|---|
| `maxNewTokens` | `number` | engine default | Maximum tokens to generate |
| `temperature` | `number` | engine default | Sampling temperature |
| `topK` | `number` | engine default | Top-K filtering |
| `minConfidence` | `number` | engine default | Minimum token confidence to continue |
| `repetitionPenalty` | `number` | engine default | Repetition penalty factor |
| `signal` | `AbortSignal` | -- | Abort signal to cancel generation |
| `modelsDir` | `string` | bundled models | Custom path to ONNX model directory |

### Return Types

**`ClassifyResult`**

```typescript
{
  predictions: { label: string; score: number }[];
  top: { label: string; score: number };
  latencyMs: number;
}
```

**`AutocompleteResult`**

```typescript
{
  text: string;
  tokens: number[];
}
```

### Streaming Example

```typescript
import { autocompleteStream } from "knowmatic";

for await (const token of autocompleteStream("How do I")) {
  process.stdout.write(token);
}
```

### Advanced Re-exports

For low-level access, the package also re-exports:

- `InferenceEngine`, `Tokenizer`, `AutocompleteEngine` -- core engine classes
- `containsCode`, `extractCode` -- code detection utilities
- `applyRepetitionPenalty`, `temperatureScale`, `topKFilter`, `softmaxMax`, `sampleFromLogits` -- sampling primitives
- `MODELS_DIR` -- resolved path to the bundled model directory

## CLI Usage

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
  lib.ts            # Public library API
  paths.ts          # Model directory resolution
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
