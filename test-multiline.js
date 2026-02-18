// Minimal test: type text, press Enter for new lines, Ctrl+C to quit
// Run: node test-multiline.js

process.stdout.write("\x1b[?1049h"); // enter alt screen buffer
process.stdin.setRawMode(true);
process.stdin.resume();

let input = "";

function render() {
  const K = "\x1b[K";
  const lines = input.split("\n");

  let out = "\x1b[?25l"; // hide cursor
  out += "\x1b[H";       // cursor home

  out += "=== Multi-line test ===" + K + "\n";
  out += "Type text. Enter = new line. Ctrl+C = quit." + K + "\n";
  out += "───────────────────────────────────" + K + "\n";

  for (let i = 0; i < lines.length; i++) {
    const prefix = i === 0 ? " > " : "   ";
    out += prefix + lines[i] + K + "\n";
  }

  out += K + "\n";
  out += "Lines: " + lines.length + "  |  Bytes: " + Buffer.byteLength(input) + K + "\n";
  out += "\x1b[J";   // clear rest of screen
  out += "\x1b[?25h"; // show cursor

  process.stdout.write(out);
}

process.stdin.on("data", (key) => {
  const b = key[0];
  if (b === 0x03) {
    process.stdout.write("\x1b[?1049l");
    process.exit();
  }
  if (b === 0x0d || b === 0x0a) {
    input += "\n";
    render();
    return;
  }
  if (b === 0x7f || b === 0x08) {
    input = input.slice(0, -1);
    render();
    return;
  }
  if (b === 0x1b) return; // ignore escape sequences
  if (b < 0x20) return;   // ignore other control chars
  input += key.toString("utf-8");
  render();
});

render();
