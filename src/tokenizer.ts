import { readFile } from "node:fs/promises";

const BYTE_TO_UNICODE: Record<number, string> = {};
const UNICODE_TO_BYTE: Record<string, number> = {};

function initByteToUnicode(): void {
  if (Object.keys(BYTE_TO_UNICODE).length > 0) return;

  const bs: number[] = [];
  for (let i = 33; i <= 126; i++) bs.push(i);
  for (let i = 161; i <= 172; i++) bs.push(i);
  for (let i = 174; i <= 255; i++) bs.push(i);

  const cs = [...bs];
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n);
      n++;
    }
  }

  for (let i = 0; i < bs.length; i++) {
    BYTE_TO_UNICODE[bs[i]] = String.fromCharCode(cs[i]);
    UNICODE_TO_BYTE[String.fromCharCode(cs[i])] = bs[i];
  }
}

export class Tokenizer {
  private vocab: Map<string, number> = new Map();
  private reverseVocab: Map<number, string> = new Map();
  private merges: Map<string, number> = new Map();
  private specialTokens = { pad: 0, unk: 1, bos: 2, eos: 3 };
  private loaded = false;

  constructor() {
    initByteToUnicode();
  }

  async load(path: string): Promise<void> {
    if (this.loaded) return;

    const raw = await readFile(path, "utf-8");
    const data = JSON.parse(raw);

    if (data.model?.vocab) {
      for (const [token, id] of Object.entries(data.model.vocab)) {
        this.vocab.set(token, id as number);
        this.reverseVocab.set(id as number, token);
      }
    }

    if (data.model?.merges) {
      (data.model.merges as [string, string][]).forEach(
        (merge: [string, string], index: number) => {
          this.merges.set(`${merge[0]} ${merge[1]}`, index);
        }
      );
    }

    for (const token of data.added_tokens || []) {
      if (token.content === "<pad>") this.specialTokens.pad = token.id;
      if (token.content === "<unk>") this.specialTokens.unk = token.id;
      if (token.content === "<bos>") this.specialTokens.bos = token.id;
      if (token.content === "<eos>") this.specialTokens.eos = token.id;
    }

    this.loaded = true;
  }

  private textToBytes(text: string): string {
    const encoder = new TextEncoder();
    const bytes = encoder.encode(text);
    return Array.from(bytes)
      .map((b) => BYTE_TO_UNICODE[b])
      .join("");
  }

  private getPairs(word: string[]): [string, string][] {
    const pairs: [string, string][] = [];
    for (let i = 0; i < word.length - 1; i++) {
      pairs.push([word[i], word[i + 1]]);
    }
    return pairs;
  }

  private bpe(token: string): string[] {
    if (this.vocab.has(token)) return [token];

    let word = token.split("");
    if (word.length === 0) return [];

    let pairs = this.getPairs(word);
    if (pairs.length === 0) return [token];

    while (true) {
      let minPair: [string, string] | null = null;
      let minRank = Infinity;

      for (const pair of pairs) {
        const rank = this.merges.get(`${pair[0]} ${pair[1]}`);
        if (rank !== undefined && rank < minRank) {
          minRank = rank;
          minPair = pair;
        }
      }

      if (minPair === null) break;

      const [first, second] = minPair;
      const newWord: string[] = [];
      let i = 0;

      while (i < word.length) {
        const j = word.indexOf(first, i);
        if (j === -1) {
          newWord.push(...word.slice(i));
          break;
        }
        newWord.push(...word.slice(i, j));
        if (j < word.length - 1 && word[j + 1] === second) {
          newWord.push(first + second);
          i = j + 2;
        } else {
          newWord.push(word[j]);
          i = j + 1;
        }
      }

      word = newWord;
      if (word.length === 1) break;
      pairs = this.getPairs(word);
    }

    return word;
  }

  private tokenizeWord(word: string): number[] {
    const byteEncoded = this.textToBytes(word);
    const bpeTokens = this.bpe(byteEncoded);
    return bpeTokens.map(
      (token) => this.vocab.get(token) ?? this.specialTokens.unk
    );
  }

  encode(
    text: string,
    maxLength = 512
  ): { inputIds: number[]; attentionMask: number[] } {
    const pattern =
      /'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|[\r\n]\s{0,3}|\s+$/gu;
    const matches = text.match(pattern) || [];

    let tokens: number[] = [this.specialTokens.bos];
    for (const match of matches) {
      tokens.push(...this.tokenizeWord(match));
    }
    tokens.push(this.specialTokens.eos);

    if (tokens.length > maxLength) {
      tokens = tokens.slice(0, maxLength);
    }

    const attentionMask = new Array(tokens.length).fill(1);

    while (tokens.length < maxLength) {
      tokens.push(this.specialTokens.pad);
      attentionMask.push(0);
    }

    return { inputIds: tokens, attentionMask };
  }

  encodeForGeneration(text: string): number[] {
    const pattern =
      /'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|[\r\n]\s{0,3}|\s+$/gu;
    const matches = text.match(pattern) || [];

    const tokens: number[] = [this.specialTokens.bos];
    for (const match of matches) {
      tokens.push(...this.tokenizeWord(match));
    }
    return tokens;
  }

  decode(tokenIds: number[]): string {
    const tokens: string[] = [];
    for (const id of tokenIds) {
      if (
        id === this.specialTokens.pad ||
        id === this.specialTokens.bos ||
        id === this.specialTokens.eos ||
        id === this.specialTokens.unk
      ) {
        continue;
      }
      const token = this.reverseVocab.get(id);
      if (token !== undefined) tokens.push(token);
    }

    const byteValues: number[] = [];
    for (const token of tokens) {
      for (const char of token) {
        const byteVal = UNICODE_TO_BYTE[char];
        if (byteVal !== undefined) byteValues.push(byteVal);
      }
    }

    return new TextDecoder().decode(new Uint8Array(byteValues));
  }

  getSpecialTokens() {
    return { ...this.specialTokens };
  }
}
