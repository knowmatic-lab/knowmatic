import chalk from "chalk";
import type { Prediction } from "./inference.js";

const BAR_WIDTH = 28;

const MODEL_MAP: Record<string, string> = {
  Easy: "Haiku",
  Medium: "Sonnet",
  Hard: "Opus",
};

const MODEL_COLORS: Record<string, (s: string) => string> = {
  Easy: chalk.green,
  Medium: chalk.yellow,
  Hard: chalk.red,
};

const EFFORT_COLORS: Record<string, (s: string) => string> = {
  low: chalk.green,
  medium: chalk.yellow,
  high: chalk.red,
};

function bar(ratio: number, color: (s: string) => string): string {
  const filled = Math.round(ratio * BAR_WIDTH);
  const empty = BAR_WIDTH - filled;
  return color("█".repeat(filled)) + chalk.dim("░".repeat(empty));
}

function pct(score: number): string {
  return `${(score * 100).toFixed(1)}%`.padStart(6);
}

export interface ClassificationResults {
  difficulty?: Prediction[];
  effort?: Prediction[];
  code?: { detected: boolean; predictions?: Prediction[] };
  latencyMs?: number;
}

export function renderScreen(
  input: string,
  suggestion: string,
  results: ClassificationResults | null,
  status: string
): void {
  const w = Math.min(process.stdout.columns || 80, 90);

  const K = "\x1b[K"; // clear to end of line

  const out: string[] = [];

  out.push("\x1b[?25l"); // hide cursor during render
  out.push("\x1b[H"); // cursor home (top-left of alternate screen)

  // Header (3 rows)
  out.push(
    chalk.bold.cyan(" knowmatic") +
      chalk.dim(" — local prompt classification") +
      K + "\n"
  );
  out.push(chalk.dim(" " + "─".repeat(w - 2)) + K + "\n");
  out.push(K + "\n");

  // Input lines — multi-line support
  const inputLines = input.split("\n");
  for (let i = 0; i < inputLines.length; i++) {
    const prefix = chalk.cyan(" > ");
    out.push(prefix + inputLines[i]);
    if (i === inputLines.length - 1) {
      // Ghost text on same line as last input line
      const rawGhost =
        input.endsWith(" ") && suggestion.startsWith(" ")
          ? suggestion.slice(1)
          : suggestion;
      const ghostText = rawGhost.replace(/[\r\n]+/g, " ");
      out.push(chalk.dim(ghostText) + K + "\n");
    } else {
      out.push(K + "\n");
    }
  }

  // Calculate cursor position (1-based ANSI coordinates)
  // 3 header rows + which input line the cursor is on
  const cursorRow = 3 + inputLines.length;
  const lastLine = inputLines[inputLines.length - 1];
  const cursorCol = 4 + lastLine.length; // 3-char prefix + 1 (1-based)

  out.push(K + "\n");

  // Status bar
  const statusParts: string[] = [];
  if (suggestion) {
    statusParts.push(chalk.cyan("Tab") + chalk.dim(" accept"));
  }
  statusParts.push(chalk.cyan("Enter") + chalk.dim(" classify"));
  statusParts.push(chalk.cyan("Alt+Enter") + chalk.dim(" newline"));
  if (input.length > 0) {
    statusParts.push(chalk.cyan("Ctrl+U") + chalk.dim(" clear"));
  }
  statusParts.push(chalk.cyan("Ctrl+C") + chalk.dim(" quit"));
  out.push(
    chalk.dim(" ") + statusParts.join(chalk.dim("  ·  ")) + K + "\n"
  );
  out.push(chalk.dim(" " + "─".repeat(w - 2)) + K + "\n");

  // Results
  if (results) {
    out.push(K + "\n");

    if (results.difficulty) {
      const top = results.difficulty[0];
      const topModel = MODEL_MAP[top.label] || top.label;
      const topColor = MODEL_COLORS[top.label] || chalk.white;

      out.push(chalk.bold(" Model Selection") + K + "\n");
      out.push(K + "\n");

      for (const p of results.difficulty) {
        const name = MODEL_MAP[p.label] || p.label;
        const color = MODEL_COLORS[p.label] || chalk.white;
        const isTop = p.label === top.label;
        const prefix = isTop ? chalk.bold(" > ") : "   ";
        const label = isTop
          ? chalk.bold(color(name.padEnd(8)))
          : chalk.dim(name.padEnd(8));
        out.push(
          `${prefix}${label} ${bar(p.score, color)} ${chalk.dim(pct(p.score))}${K}\n`
        );
      }

      out.push(K + "\n");
      out.push(
        ` ${chalk.dim("Route →")} ${topColor(chalk.bold(topModel))}${K}\n`
      );
    }

    out.push(chalk.dim(" " + "─".repeat(w - 2)) + K + "\n");
    out.push(K + "\n");

    if (results.effort) {
      const top = results.effort[0];
      const topColor = EFFORT_COLORS[top.label] || chalk.white;

      out.push(chalk.bold(" Reasoning Effort") + K + "\n");
      out.push(K + "\n");

      for (const p of results.effort) {
        const color = EFFORT_COLORS[p.label] || chalk.white;
        const isTop = p.label === top.label;
        const prefix = isTop ? chalk.bold(" > ") : "   ";
        const label = isTop
          ? chalk.bold(color(p.label.padEnd(8)))
          : chalk.dim(p.label.padEnd(8));
        out.push(
          `${prefix}${label} ${bar(p.score, color)} ${chalk.dim(pct(p.score))}${K}\n`
        );
      }

      out.push(K + "\n");
      out.push(
        ` ${chalk.dim("Effort →")} ${topColor(chalk.bold(top.label))}${K}\n`
      );
    }

    out.push(chalk.dim(" " + "─".repeat(w - 2)) + K + "\n");
    out.push(K + "\n");

    if (results.code) {
      out.push(chalk.bold(" Code Detection") + K + "\n");
      out.push(K + "\n");

      if (!results.code.detected || !results.code.predictions) {
        out.push(chalk.dim("   No code detected") + K + "\n");
      } else {
        const top3 = results.code.predictions.slice(0, 3);
        for (const p of top3) {
          const isTop = p === top3[0];
          const prefix = isTop ? chalk.bold(" > ") : "   ";
          const color = isTop ? chalk.cyan : chalk.dim;
          const label = color(p.label.padEnd(14));
          out.push(
            `${prefix}${label} ${bar(p.score, isTop ? chalk.cyan : chalk.dim)} ${chalk.dim(pct(p.score))}${K}\n`
          );
        }
        out.push(K + "\n");
        out.push(
          ` ${chalk.dim("Language →")} ${chalk.cyan(chalk.bold(top3[0].label))}${K}\n`
        );
      }
    }

    if (results.difficulty) {
      const diff = results.difficulty[0].label;
      if (diff !== "Hard") {
        const costs: Record<string, number> = {
          Easy: 6,
          Medium: 18,
          Hard: 90,
        };
        const saved = Math.round(
          ((costs["Hard"] - costs[diff]) / costs["Hard"]) * 100
        );
        const model = MODEL_MAP[diff];
        out.push(K + "\n");
        out.push(
          ` ${chalk.dim("Cost →")} ${chalk.green(`${saved}% savings`)} ${chalk.dim(`routing to ${model} instead of Opus`)}${K}\n`
        );
      }
    }

    if (results.latencyMs !== undefined) {
      out.push(K + "\n");
      out.push(
        chalk.dim(` Classified in ${results.latencyMs}ms`) + K + "\n"
      );
    }
  } else if (input.length > 0) {
    out.push(K + "\n");
    out.push(
      chalk.dim(" Press Enter to classify this prompt") + K + "\n"
    );
  } else {
    out.push(K + "\n");
    out.push(
      chalk.dim(" Start typing a prompt to classify...") + K + "\n"
    );
    out.push(K + "\n");
    out.push(
      chalk.dim(
        " The autocomplete model will suggest completions as you type."
      ) +
        K +
        "\n"
    );
    out.push(
      chalk.dim(" Press Tab to accept a suggestion, Enter to classify.") +
        K +
        "\n"
    );
  }

  if (status) {
    out.push(K + "\n");
    out.push(chalk.dim(` ${status}`) + K + "\n");
  }

  // Clear everything below the last written line
  out.push("\x1b[J");

  // Position cursor at end of last input line using absolute coordinates
  out.push(`\x1b[${cursorRow};${cursorCol}H`);
  out.push("\x1b[5 q"); // blinking bar cursor
  out.push("\x1b[?25h"); // show cursor

  process.stdout.write(out.join(""));
}
