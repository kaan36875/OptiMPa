// Copies the web-ifc WebAssembly file into public/ so the BIM page can load it.
// Runs automatically before `npm run dev` and `npm run build`.
import { copyFileSync, mkdirSync } from "node:fs";

const target = new URL("../public/wasm/", import.meta.url);
mkdirSync(target, { recursive: true });
copyFileSync(new URL("../node_modules/web-ifc/web-ifc.wasm", import.meta.url), new URL("web-ifc.wasm", target));
console.log("web-ifc.wasm -> public/wasm/");
