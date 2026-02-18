export function applyRepetitionPenalty(
  logits: Float32Array,
  tokenCounts: Map<number, number>,
  penalty: number
): Float32Array {
  if (penalty <= 1) return logits;
  const penalized = new Float32Array(logits);
  for (const [id, count] of tokenCounts) {
    if (id < penalized.length) {
      const scaled = Math.pow(penalty, count);
      penalized[id] =
        penalized[id] > 0 ? penalized[id] / scaled : penalized[id] * scaled;
    }
  }
  return penalized;
}

export function temperatureScale(
  logits: Float32Array,
  temperature: number
): Float32Array {
  if (temperature <= 0) return logits;
  const scaled = new Float32Array(logits.length);
  for (let i = 0; i < logits.length; i++) {
    scaled[i] = logits[i] / temperature;
  }
  return scaled;
}

export function topKFilter(
  logits: Float32Array,
  k: number
): { indices: number[]; logits: Float32Array } {
  const indexed = Array.from(logits).map((val, idx) => ({ val, idx }));
  indexed.sort((a, b) => b.val - a.val);
  const topK = indexed.slice(0, k);
  return {
    indices: topK.map((x) => x.idx),
    logits: new Float32Array(topK.map((x) => x.val)),
  };
}

export function softmaxMax(logits: Float32Array): number {
  const maxLogit = Math.max(...logits);
  let sumExp = 0;
  let maxExp = 0;
  for (let i = 0; i < logits.length; i++) {
    const e = Math.exp(logits[i] - maxLogit);
    sumExp += e;
    if (e > maxExp) maxExp = e;
  }
  return maxExp / sumExp;
}

export function sampleFromLogits(logits: Float32Array): number {
  const maxLogit = Math.max(...logits);
  const expScores = new Float32Array(logits.length);
  let sumExp = 0;
  for (let i = 0; i < logits.length; i++) {
    expScores[i] = Math.exp(logits[i] - maxLogit);
    sumExp += expScores[i];
  }

  const r = Math.random() * sumExp;
  let cumulative = 0;
  for (let i = 0; i < expScores.length; i++) {
    cumulative += expScores[i];
    if (r <= cumulative) return i;
  }
  return expScores.length - 1;
}
