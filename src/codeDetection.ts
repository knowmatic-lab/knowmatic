export function containsCode(text: string): boolean {
  return /```[\s\S]*?```/.test(text);
}

export function extractCode(text: string): string | null {
  const match = text.match(/```(?:\w*\n)?([\s\S]*?)```/);
  return match ? match[1].trim() : null;
}
