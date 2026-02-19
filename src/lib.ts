import { resolve } from "node:path";
import { MODELS_DIR } from "./paths.js";
import { InferenceEngine, type Prediction, type ModelMetadata } from "./inference.js";
import { Tokenizer } from "./tokenizer.js";
import { AutocompleteEngine, type GenerateOptions } from "./autocomplete.js";
import { containsCode, extractCode } from "./codeDetection.js";
import {
  applyRepetitionPenalty,
  temperatureScale,
  topKFilter,
  softmaxMax,
  sampleFromLogits,
} from "./sampling.js";

// ── Types ───────────────────────────────────────────────────────────

export interface ClassifyResult {
  predictions: Prediction[];
  top: { label: string; score: number };
  latencyMs: number;
}

export interface AutocompleteResult {
  text: string;
  tokens: number[];
}

export interface AutocompleteOptions {
  maxNewTokens?: number;
  temperature?: number;
  topK?: number;
  minConfidence?: number;
  repetitionPenalty?: number;
  signal?: AbortSignal;
  modelsDir?: string;
}

export interface ClassifyOptions {
  modelsDir?: string;
}

// ── Lazy-loaded engine caches ───────────────────────────────────────

let classifierTokenizerPromise: Promise<Tokenizer> | null = null;
let autocompleteTokenizerPromise: Promise<Tokenizer> | null = null;
let difficultyPromise: Promise<InferenceEngine> | null = null;
let effortPromise: Promise<InferenceEngine> | null = null;
let codePromise: Promise<InferenceEngine> | null = null;
let autocompletePromise: Promise<AutocompleteEngine> | null = null;

function getClassifierTokenizer(dir: string): Promise<Tokenizer> {
  if (!classifierTokenizerPromise) {
    classifierTokenizerPromise = (async () => {
      const t = new Tokenizer();
      await t.load(resolve(dir, "tokenizer", "tokenizer.json"));
      return t;
    })();
  }
  return classifierTokenizerPromise;
}

function getAutocompleteTokenizer(dir: string): Promise<Tokenizer> {
  if (!autocompleteTokenizerPromise) {
    autocompleteTokenizerPromise = (async () => {
      const t = new Tokenizer();
      await t.load(resolve(dir, "tokenizer", "tokenizer_autocomplete.json"));
      return t;
    })();
  }
  return autocompleteTokenizerPromise;
}

function getDifficultyEngine(dir: string): Promise<InferenceEngine> {
  if (!difficultyPromise) {
    difficultyPromise = (async () => {
      const e = new InferenceEngine();
      await e.load(
        resolve(dir, "difficulty_classifier", "model_quantized.onnx"),
        resolve(dir, "difficulty_classifier", "metadata.json")
      );
      return e;
    })();
  }
  return difficultyPromise;
}

function getEffortEngine(dir: string): Promise<InferenceEngine> {
  if (!effortPromise) {
    effortPromise = (async () => {
      const e = new InferenceEngine();
      await e.load(
        resolve(dir, "reasoning_effort", "model_quantized.onnx"),
        resolve(dir, "reasoning_effort", "metadata.json")
      );
      return e;
    })();
  }
  return effortPromise;
}

function getCodeEngine(dir: string): Promise<InferenceEngine> {
  if (!codePromise) {
    codePromise = (async () => {
      const e = new InferenceEngine();
      await e.load(
        resolve(dir, "code_classifier", "model_quantized.onnx"),
        resolve(dir, "code_classifier", "metadata.json")
      );
      return e;
    })();
  }
  return codePromise;
}

function getAutocompleteEngine(dir: string): Promise<AutocompleteEngine> {
  if (!autocompletePromise) {
    autocompletePromise = (async () => {
      const e = new AutocompleteEngine();
      await e.load(
        resolve(dir, "sft", "model_quantized.onnx"),
        resolve(dir, "sft", "sft_metadata.json")
      );
      return e;
    })();
  }
  return autocompletePromise;
}

// ── Helpers ─────────────────────────────────────────────────────────

async function runClassifier(
  text: string,
  getEngine: (dir: string) => Promise<InferenceEngine>,
  opts?: ClassifyOptions
): Promise<ClassifyResult> {
  const dir = opts?.modelsDir ?? MODELS_DIR;
  const start = performance.now();

  const [tokenizer, engine] = await Promise.all([
    getClassifierTokenizer(dir),
    getEngine(dir),
  ]);

  const { inputIds, attentionMask } = tokenizer.encode(text);
  const predictions = await engine.classify(inputIds, attentionMask);
  const latencyMs = Math.round(performance.now() - start);

  return {
    predictions,
    top: { label: predictions[0].label, score: predictions[0].score },
    latencyMs,
  };
}

// ── Public API ──────────────────────────────────────────────────────

export async function classifyDifficulty(
  text: string,
  opts?: ClassifyOptions
): Promise<ClassifyResult> {
  return runClassifier(text, getDifficultyEngine, opts);
}

export async function classifyReasoningEffort(
  text: string,
  opts?: ClassifyOptions
): Promise<ClassifyResult> {
  return runClassifier(text, getEffortEngine, opts);
}

export async function classifyCode(
  text: string,
  opts?: ClassifyOptions
): Promise<ClassifyResult> {
  return runClassifier(text, getCodeEngine, opts);
}

export async function autocomplete(
  text: string,
  opts?: AutocompleteOptions
): Promise<AutocompleteResult> {
  const dir = opts?.modelsDir ?? MODELS_DIR;

  const [tokenizer, engine] = await Promise.all([
    getAutocompleteTokenizer(dir),
    getAutocompleteEngine(dir),
  ]);

  const inputIds = tokenizer.encodeForGeneration(text.trim());
  const tokens: number[] = [];

  for await (const tokenId of engine.generate(inputIds, {
    maxNewTokens: opts?.maxNewTokens,
    temperature: opts?.temperature,
    topK: opts?.topK,
    minConfidence: opts?.minConfidence,
    repetitionPenalty: opts?.repetitionPenalty,
    signal: opts?.signal,
  })) {
    tokens.push(tokenId);
  }

  return { text: tokenizer.decode(tokens), tokens };
}

export async function* autocompleteStream(
  text: string,
  opts?: AutocompleteOptions
): AsyncGenerator<string> {
  const dir = opts?.modelsDir ?? MODELS_DIR;

  const [tokenizer, engine] = await Promise.all([
    getAutocompleteTokenizer(dir),
    getAutocompleteEngine(dir),
  ]);

  const inputIds = tokenizer.encodeForGeneration(text.trim());

  for await (const tokenId of engine.generate(inputIds, {
    maxNewTokens: opts?.maxNewTokens,
    temperature: opts?.temperature,
    topK: opts?.topK,
    minConfidence: opts?.minConfidence,
    repetitionPenalty: opts?.repetitionPenalty,
    signal: opts?.signal,
  })) {
    yield tokenizer.decode([tokenId]);
  }
}

// ── Re-exports for advanced users ───────────────────────────────────

export { InferenceEngine, type Prediction, type ModelMetadata } from "./inference.js";
export { Tokenizer } from "./tokenizer.js";
export { AutocompleteEngine, type GenerateOptions } from "./autocomplete.js";
export { containsCode, extractCode } from "./codeDetection.js";
export {
  applyRepetitionPenalty,
  temperatureScale,
  topKFilter,
  softmaxMax,
  sampleFromLogits,
} from "./sampling.js";
export { MODELS_DIR } from "./paths.js";
