# ☠ AI CORE POLICY CHANGER — NIGHTMARE EDITION ☠

> A full-featured Electron GUI for inspecting, editing, and rebuilding GGUF model files.
> Dark. Brutal. Functional.

---

## Features

| Panel | What it does |
|---|---|
| **CORE** | Dashboard — system telemetry, quick stats, quick actions |
| **DECRYPT** | Drag-and-drop or browse to load any `.gguf` file (v1 / v2 / v3) |
| **POLICIES** | View, search, add, edit, and delete metadata key-value pairs |
| **TENSORS** | Browse all tensor names, shapes, types, element counts and offsets |
| **HEX** | Raw hex dump viewer with paged navigation |
| **FORGE** | Rebuild and export the modified GGUF with streaming tensor copy |
| **TERMINAL** | Live colour-coded log output |
| **SETTINGS** | Toggle visual effects, tune performance |

### Nightmare visuals
- Blood-red matrix rain canvas background
- CRT scanlines overlay + vignette
- Glitch text animations
- Pulsing neon glow on active elements
- Custom frameless window with draggable title bar

---

## Quick Start

```bash
npm install
npm start
```

## Build distributable

```bash
npm run dist
```

Outputs to `dist/`.

---

## GGUF Support

- ✅ GGUF v1 / v2 / v3
- ✅ All metadata value types (UINT8 → FLOAT64, STRING, ARRAY, BOOL)
- ✅ All GGML quantization types (F32, F16, Q4_0 … BF16)
- ✅ Streaming rebuild — handles multi-GB files without loading tensor data into RAM
- ✅ Verify magic bytes after write
- ✅ Export metadata as JSON

---

## ⚠ Warning

Modifying GGUF metadata may corrupt your model files.
**Always keep a backup before forging.**

---

*NightmareDesigns — the void stares back ☠*
