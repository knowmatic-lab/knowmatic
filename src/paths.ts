import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const __here = dirname(fileURLToPath(import.meta.url));
export const MODELS_DIR = resolve(__here, "..", "models");
