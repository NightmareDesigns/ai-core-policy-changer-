# ai-core-policy-changer

A command-line tool for inspecting, modifying policy-related metadata,
decrypting, and rebuilding **GGUF** (GPT-Generated Unified Format) model files.

---

## Features

| Feature | Description |
|---------|-------------|
| **Inspect** | List every metadata key-value pair stored in a GGUF file, with policy-relevant fields highlighted. |
| **Get / Set** | Read or update any metadata field – particularly chat templates, system prompts, and instruct templates. |
| **Decrypt** | Remove single-byte XOR obfuscation that some tools apply before writing a GGUF file. The XOR key is auto-detected or can be supplied manually. |
| **Rebuild** | Re-parse and re-serialise a file; useful for fixing alignment/padding and applying a chain of edits. |

---

## Requirements

- Python 3.8 or newer (no third-party dependencies)

---

## Usage

```
python gguf_policy_changer.py <subcommand> [options]
```

### Subcommands

#### `info` – print all metadata

```bash
python gguf_policy_changer.py info model.gguf
```

Output:
```
GGUF version : 3
Alignment    : 32
Tensors      : 291
Metadata KVs : 24

Metadata:
  general.name = "MyModel-7B"
  tokenizer.chat_template [POLICY] = "{% for msg in messages %}…"
  llama.context_length = 4096
  ...
```

Fields marked `[POLICY]` are the ones most likely to affect model behaviour.

---

#### `get` – print a single metadata value

```bash
python gguf_policy_changer.py get model.gguf tokenizer.chat_template
```

---

#### `set` – change a metadata value and rebuild

```bash
# Overwrite the file in-place
python gguf_policy_changer.py set model.gguf tokenizer.chat_template "New template here"

# Write to a new file
python gguf_policy_changer.py set model.gguf llama.system_prompt "Be concise." -o model_modified.gguf
```

---

#### `decrypt` – remove XOR obfuscation

```bash
# Auto-detect the XOR key
python gguf_policy_changer.py decrypt obfuscated.gguf -o plain.gguf

# Supply the key explicitly (decimal or hex)
python gguf_policy_changer.py decrypt obfuscated.gguf -k 0xA5 -o plain.gguf
```

---

#### `rebuild` – re-serialise a GGUF file

```bash
python gguf_policy_changer.py rebuild model.gguf -o model_rebuilt.gguf
```

---

## Common policy-relevant keys

| Key | Description |
|-----|-------------|
| `tokenizer.chat_template` | Jinja2 template used to format messages |
| `tokenizer.ggml.instruct_template` | Alternative instruct prompt template |
| `llama.system_prompt` | Default system prompt embedded in the model |
| `general.name` | Human-readable model name |
| `general.description` | Model description |
| `general.author` | Author / organisation |
| `general.license` | License identifier |
| `general.tags` | Tag list |

---

## Running the tests

```bash
python -m unittest test_gguf_policy_changer -v
```

---

## GGUF format overview

GGUF is a self-contained binary format created by the
[llama.cpp](https://github.com/ggerganov/llama.cpp) project.  Every file
contains:

1. **Header** – magic number (`GGUF`), version, tensor count, metadata count.
2. **Metadata KV pairs** – typed key-value pairs (strings, ints, floats, bools,
   arrays).
3. **Tensor info** – name, shape, data-type, and data offset for every tensor.
4. **Padding** – zero bytes to align the start of tensor data to the configured
   boundary (default 32 bytes).
5. **Tensor data** – raw quantised weight bytes.

Supported versions: **1**, **2**, **3**.
