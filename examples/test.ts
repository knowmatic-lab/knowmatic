/**
 * Quick test for knowmatic
 *
 * Setup:
 *   mkdir test-knowmatic && cd test-knowmatic
 *   npm init -y
 *   npm pkg set type=module
 *   npm install knowmatic tsx
 *   npx tsx test.ts
 */

import {
  classifyDifficulty,
  classifyReasoningEffort,
  classifyCode,
  autocomplete,
  autocompleteStream,
} from "knowmatic";

async function main() {
  // Difficulty classification
  const diff = await classifyDifficulty("Explain quantum entanglement in simple terms");
  console.log("Difficulty:", diff.top.label, `(${(diff.top.score * 100).toFixed(1)}%)`);

  // Reasoning effort classification
  const effort = await classifyReasoningEffort("What is 2+2?");
  console.log("Effort:", effort.top.label, `(${(effort.top.score * 100).toFixed(1)}%)`);

  // Code classification
  const code = await classifyCode("```python\ndef fib(n):\n  return n if n < 2 else fib(n-1) + fib(n-2)\n```");
  console.log("Code:", code.top.label, `(${(code.top.score * 100).toFixed(1)}%)`);

  // Autocomplete (full result)
  const ac = await autocomplete("How do I write a function that");
  console.log("Autocomplete:", ac.text);

  // Autocomplete (streaming)
  process.stdout.write("Stream: ");
  for await (const token of autocompleteStream("Write a Python function that")) {
    process.stdout.write(token);
  }
  process.stdout.write("\n");
}

main().catch(console.error);
