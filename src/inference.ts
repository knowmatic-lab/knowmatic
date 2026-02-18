import * as ort from "onnxruntime-node";
import { readFile } from "node:fs/promises";

export interface ModelMetadata {
  num_classes: number;
  label_map: Record<string, number>;
  id_to_label: Record<string, string>;
  max_length: number;
  vocab_size: number;
}

export interface Prediction {
  label: string;
  score: number;
}

function softmax(logits: number[]): number[] {
  const max = Math.max(...logits);
  const exps = logits.map((l) => Math.exp(l - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

export class InferenceEngine {
  private session: ort.InferenceSession | null = null;
  private metadata: ModelMetadata | null = null;

  async load(modelPath: string, metadataPath: string): Promise<void> {
    if (this.session) return;

    const raw = await readFile(metadataPath, "utf-8");
    this.metadata = JSON.parse(raw);

    this.session = await ort.InferenceSession.create(modelPath, {
      executionProviders: ["cpu"],
      graphOptimizationLevel: "all",
    });
  }

  async classify(
    inputIds: number[],
    attentionMask: number[]
  ): Promise<Prediction[]> {
    if (!this.session || !this.metadata) {
      throw new Error("Model not loaded");
    }

    const inputIdsTensor = new ort.Tensor(
      "int64",
      BigInt64Array.from(inputIds.map(BigInt)),
      [1, inputIds.length]
    );

    const attentionMaskTensor = new ort.Tensor(
      "int64",
      BigInt64Array.from(attentionMask.map(BigInt)),
      [1, attentionMask.length]
    );

    const results = await this.session.run({
      input_ids: inputIdsTensor,
      attention_mask: attentionMaskTensor,
    });

    const logits = Array.from(results.logits.data as Float32Array);
    const probs = softmax(logits);

    const predictions: Prediction[] = probs.map((score, i) => ({
      label: this.metadata!.id_to_label[String(i)],
      score,
    }));

    return predictions.sort((a, b) => b.score - a.score);
  }
}
