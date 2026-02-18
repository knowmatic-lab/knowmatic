#!/usr/bin/env node

import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import chalk from "chalk";

import { Tokenizer } from "./tokenizer.js";
import { InferenceEngine } from "./inference.js";
import { AutocompleteEngine } from "./autocomplete.js";
import { containsCode, extractCode } from "./codeDetection.js";
import { renderScreen, type ClassificationResults } from "./ui.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const MODELS_DIR = resolve(__dirname, "..", "models");

// ── State ──────────────────────────────────────────────────────────

let input = "";
let suggestion = "";
let results: ClassificationResults | null = null;
let status = "";
let classifyTimer: ReturnType<typeof setTimeout> | null = null;
let autocompleteTimer: ReturnType<typeof setTimeout> | null = null;
let autocompleteAbort: AbortController | null = null;
let isGenerating = false;

// ── Models ─────────────────────────────────────────────────────────

let classifierTokenizer: Tokenizer;
let autocompleteTokenizer: Tokenizer;
let difficultyEngine: InferenceEngine;
let effortEngine: InferenceEngine;
let codeEngine: InferenceEngine;
let autocompleteEngine: AutocompleteEngine;

function createSpinner(msg: string) {
  const frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
  let i = 0;
  const interval = setInterval(() => {
    process.stdout.write(
      `\r  ${chalk.cyan(frames[i++ % frames.length])} ${chalk.dim(msg)}`
    );
  }, 80);

  return {
    update(newMsg: string) {
      msg = newMsg;
    },
    done(finalMsg: string) {
      clearInterval(interval);
      process.stdout.write(
        `\r  ${chalk.green("✓")} ${chalk.dim(finalMsg)}\x1b[K\n`
      );
    },
  };
}

async function loadAllModels(): Promise<void> {
  const spinner = createSpinner("Loading models");

  classifierTokenizer = new Tokenizer();
  await classifierTokenizer.load(
    resolve(MODELS_DIR, "tokenizer", "tokenizer.json")
  );
  spinner.update("Loaded classifier tokenizer");

  autocompleteTokenizer = new Tokenizer();
  await autocompleteTokenizer.load(
    resolve(MODELS_DIR, "tokenizer", "tokenizer_autocomplete.json")
  );
  spinner.update("Loaded autocomplete tokenizer");

  difficultyEngine = new InferenceEngine();
  await difficultyEngine.load(
    resolve(MODELS_DIR, "difficulty_classifier", "model_quantized.onnx"),
    resolve(MODELS_DIR, "difficulty_classifier", "metadata.json")
  );
  spinner.update("Loaded difficulty classifier");

  effortEngine = new InferenceEngine();
  await effortEngine.load(
    resolve(MODELS_DIR, "reasoning_effort", "model_quantized.onnx"),
    resolve(MODELS_DIR, "reasoning_effort", "metadata.json")
  );
  spinner.update("Loaded reasoning effort classifier");

  codeEngine = new InferenceEngine();
  await codeEngine.load(
    resolve(MODELS_DIR, "code_classifier", "model_quantized.onnx"),
    resolve(MODELS_DIR, "code_classifier", "metadata.json")
  );
  spinner.update("Loaded code classifier");

  autocompleteEngine = new AutocompleteEngine();
  await autocompleteEngine.load(
    resolve(MODELS_DIR, "sft", "model_quantized.onnx"),
    resolve(MODELS_DIR, "sft", "sft_metadata.json")
  );
  spinner.done("All models loaded (4 classifiers + autocomplete)");
}

// ── Classification ─────────────────────────────────────────────────

async function classifyPrompt(text: string): Promise<void> {
  const start = performance.now();

  const { inputIds, attentionMask } = classifierTokenizer.encode(text);

  const [difficultyPreds, effortPreds] = await Promise.all([
    difficultyEngine.classify(inputIds, attentionMask),
    effortEngine.classify(inputIds, attentionMask),
  ]);

  let codePreds = undefined;
  const hasCode = containsCode(text);
  if (hasCode) {
    const codeText = extractCode(text) || text;
    const codeEncoded = classifierTokenizer.encode(codeText);
    codePreds = await codeEngine.classify(
      codeEncoded.inputIds,
      codeEncoded.attentionMask
    );
  }

  const elapsed = Math.round(performance.now() - start);

  results = {
    difficulty: difficultyPreds,
    effort: effortPreds,
    code: { detected: hasCode, predictions: codePreds },
    latencyMs: elapsed,
  };

  render();
}

// ── Autocomplete ───────────────────────────────────────────────────

function cancelAutocomplete(): void {
  if (autocompleteTimer) {
    clearTimeout(autocompleteTimer);
    autocompleteTimer = null;
  }
  if (autocompleteAbort) {
    autocompleteAbort.abort();
    autocompleteAbort = null;
  }
  isGenerating = false;
}

function requestAutocomplete(text: string): void {
  cancelAutocomplete();
  suggestion = "";

  const trimmed = text.trim();
  if (!trimmed || !autocompleteEngine.isLoaded()) return;

  autocompleteTimer = setTimeout(async () => {
    autocompleteAbort = new AbortController();
    isGenerating = true;
    status = "generating...";
    render();

    try {
      const inputIds = autocompleteTokenizer.encodeForGeneration(trimmed);

      for await (const tokenId of autocompleteEngine.generate(inputIds, {
        minConfidence: 0.95,
        repetitionPenalty: 1.5,
        signal: autocompleteAbort.signal,
      })) {
        if (autocompleteAbort.signal.aborted) return;
        const tokenText = autocompleteTokenizer.decode([tokenId]);
        suggestion += tokenText;
        status = "Tab to accept";
        render();
      }
    } catch {
      // generation cancelled or failed
    }

    isGenerating = false;
    if (suggestion) {
      status = "Tab to accept";
    } else {
      status = "";
    }
    render();
  }, 500);
}

function acceptSuggestion(): string {
  if (!suggestion) return "";
  // Accept next word (up to and including the next space boundary)
  const match = suggestion.match(/^\s*\S+\s*/);
  const rawAccepted = match ? match[0] : suggestion;
  const remaining = suggestion.slice(rawAccepted.length);
  suggestion = remaining;

  // Flatten newlines — input stays single-line in the TUI
  let accepted = rawAccepted.replace(/[\r\n]+/g, " ").replace(/\s+$/, "");
  if (!accepted) return "";
  if (remaining.length > 0) return accepted + " ";
  return accepted;
}

// ── Render ─────────────────────────────────────────────────────────

function render(): void {
  renderScreen(input, suggestion, results, status);
}

// ── Input Handling ─────────────────────────────────────────────────

function handleKey(key: Buffer): void {
  const b = key[0]; // first byte

  // Ctrl+C
  if (b === 0x03) {
    cleanup();
    process.stdout.write(chalk.dim("\n  Goodbye.\n\n"));
    process.exit(0);
  }

  // Ctrl+U — clear input
  if (b === 0x15) {
    input = "";
    suggestion = "";
    results = null;
    status = "";
    cancelAutocomplete();
    render();
    return;
  }

  // Tab — accept suggestion
  if (b === 0x09) {
    if (suggestion) {
      let accepted = acceptSuggestion();
      // Prevent double space when input ends with space and accepted starts with one
      if (input.endsWith(" ") && accepted.startsWith(" ")) {
        accepted = accepted.slice(1);
      }
      input += accepted;
      render();
      if (!suggestion) {
        requestAutocomplete(input);
      }
    }
    return;
  }

  // Enter — classify (CR 0x0d or LF 0x0a)
  if (b === 0x0d || b === 0x0a) {
    if (input.trim()) {
      cancelAutocomplete();
      suggestion = "";
      status = "classifying...";
      render();
      classifyPrompt(input).then(() => {
        status = "";
        render();
      });
    }
    return;
  }

  // Alt+Enter — insert newline
  if (b === 0x1b && key.length === 2 && (key[1] === 0x0d || key[1] === 0x0a)) {
    cancelAutocomplete();
    suggestion = "";
    results = null;
    input += "\n";
    render();
    requestAutocomplete(input);
    return;
  }

  // Escape — clear suggestion
  if (b === 0x1b && key.length === 1) {
    cancelAutocomplete();
    suggestion = "";
    status = "";
    render();
    return;
  }

  // Backspace
  if (b === 0x7f || b === 0x08) {
    if (input.length > 0) {
      input = input.slice(0, -1);
      cancelAutocomplete();
      suggestion = "";
      results = null;
      render();
      if (input.trim()) {
        requestAutocomplete(input);
      }
    }
    return;
  }

  // Ignore escape sequences (arrow keys, etc.)
  if (b === 0x1b && key.length > 1) {
    return;
  }

  // Ignore remaining non-printable control characters
  if (b < 0x20) {
    return;
  }

  // Regular character input
  const str = key.toString("utf-8");

  // Check if typed char consumes the suggestion
  if (suggestion && suggestion.length > 0) {
    if (suggestion.startsWith(str)) {
      suggestion = suggestion.slice(str.length);
      input += str;
      render();
      return;
    }
    if (
      input.endsWith(" ") &&
      suggestion.startsWith(" ") &&
      suggestion.slice(1).startsWith(str)
    ) {
      suggestion = suggestion.slice(1 + str.length);
      input += str;
      render();
      return;
    }
  }

  // Normal typing
  input += str;
  cancelAutocomplete();
  suggestion = "";
  results = null;
  render();
  requestAutocomplete(input);
}

// ── Cleanup ────────────────────────────────────────────────────────

function cleanup(): void {
  process.stdout.write(
    "\x1b[?25h" + // show cursor
      "\x1b[0 q" + // restore default cursor shape
      "\x1b[?1049l" // leave alternate screen buffer (restores original terminal)
  );
}

// ── Main ───────────────────────────────────────────────────────────

async function main(): Promise<void> {
  if (process.argv.includes("--help") || process.argv.includes("-h")) {
    console.log(`
  ${chalk.bold.cyan("knowmatic")} — local prompt classification

  ${chalk.bold("Usage:")}
    knowmatic            Launch the TUI
    knowmatic --help     Show this help

  ${chalk.bold("TUI Controls:")}
    Type         Enter your prompt
    Tab          Accept autocomplete suggestion
    Enter        Classify the prompt
    Alt+Enter    Insert newline
    Escape       Clear suggestion
    Ctrl+U       Clear input
    Ctrl+C       Quit
`);
    process.exit(0);
  }

  await loadAllModels();

  // Enter raw mode
  if (!process.stdin.isTTY) {
    console.error(chalk.red("  Error: knowmatic requires a TTY terminal."));
    process.exit(1);
  }

  // Enter alternate screen buffer (like vim/htop — clean canvas, restores on exit)
  process.stdout.write("\x1b[?1049h");

  process.stdin.setRawMode(true);
  process.stdin.resume();
  process.stdin.on("data", handleKey);

  // Restore terminal on any exit
  process.on("exit", () => {
    process.stdout.write("\x1b[0 q\x1b[?25h\x1b[?1049l");
  });

  // Initial render
  render();
}

main().catch((err) => {
  cleanup();
  console.error(chalk.red(`\n  Error: ${err.message}\n`));
  process.exit(1);
});
