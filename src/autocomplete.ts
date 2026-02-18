import * as ort from "onnxruntime-node";
import { readFile } from "node:fs/promises";
import {
  applyRepetitionPenalty,
  temperatureScale,
  topKFilter,
  softmaxMax,
  sampleFromLogits,
} from "./sampling.js";

interface GenerativeMetadata {
  model_type: string;
  vocab_size: number;
  max_length: number;
  generation_defaults: {
    max_new_tokens: number;
    temperature: number;
    top_k: number;
  };
  special_tokens: {
    pad: number;
    unk: number;
    bos: number;
    eos: number;
  };
}

export interface GenerateOptions {
  maxNewTokens?: number;
  temperature?: number;
  topK?: number;
  minConfidence?: number;
  repetitionPenalty?: number;
  signal?: AbortSignal;
}

export class AutocompleteEngine {
  private session: ort.InferenceSession | null = null;
  private metadata: GenerativeMetadata | null = null;

  async load(modelPath: string, metadataPath: string): Promise<void> {
    if (this.session) return;

    const raw = await readFile(metadataPath, "utf-8");
    this.metadata = JSON.parse(raw);

    this.session = await ort.InferenceSession.create(modelPath, {
      executionProviders: ["cpu"],
      graphOptimizationLevel: "all",
    });
  }

  async *generate(
    inputIds: number[],
    options: GenerateOptions = {}
  ): AsyncGenerator<number, void, undefined> {
    if (!this.session || !this.metadata) {
      throw new Error("Autocomplete model not loaded");
    }

    const {
      maxNewTokens = this.metadata.generation_defaults.max_new_tokens,
      temperature = this.metadata.generation_defaults.temperature,
      topK = this.metadata.generation_defaults.top_k,
      minConfidence = 0,
      repetitionPenalty = 1.0,
      signal,
    } = options;

    const ids = [...inputIds];
    const eosTokenId = this.metadata.special_tokens.eos;

    for (let step = 0; step < maxNewTokens; step++) {
      if (signal?.aborted) return;

      const inputTensor = new ort.Tensor(
        "int64",
        BigInt64Array.from(ids.map(BigInt)),
        [1, ids.length]
      );

      const results = await this.session.run({ input_ids: inputTensor });
      const logits = results.logits.data as Float32Array;

      const vocabSize = this.metadata.vocab_size;
      const lastPositionOffset = (ids.length - 1) * vocabSize;
      const lastLogits = logits.slice(
        lastPositionOffset,
        lastPositionOffset + vocabSize
      );

      const tokenCounts = new Map<number, number>();
      for (const id of ids) {
        tokenCounts.set(id, (tokenCounts.get(id) || 0) + 1);
      }

      const penalizedLogits = applyRepetitionPenalty(
        lastLogits,
        tokenCounts,
        repetitionPenalty
      );
      const scaled = temperatureScale(penalizedLogits, temperature);
      const { indices, logits: topKLogits } = topKFilter(scaled, topK);

      if (minConfidence > 0) {
        const topProb = softmaxMax(topKLogits);
        if (topProb < minConfidence) return;
      }

      const sampledIdx = sampleFromLogits(topKLogits);
      const nextToken = indices[sampledIdx];

      if (nextToken === eosTokenId) return;

      ids.push(nextToken);
      yield nextToken;
    }
  }

  isLoaded(): boolean {
    return this.session !== null;
  }
}
