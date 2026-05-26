/* ──────────────────────────────────────────────────────────────────────────────
   GGUFParser — parses GGUF v1 / v2 / v3 headers from an ArrayBuffer.
   GGUFWriter — serialises a (possibly modified) header back to bytes.
   ────────────────────────────────────────────────────────────────────────────── */

const GGUF_MAGIC_LE = 0x46554747; // 'GGUF' little-endian uint32

// Value type IDs
const GGUFType = Object.freeze({
  UINT8: 0, INT8: 1, UINT16: 2, INT16: 3,
  UINT32: 4, INT32: 5, FLOAT32: 6, BOOL: 7,
  STRING: 8, ARRAY: 9, UINT64: 10, INT64: 11, FLOAT64: 12,
});

const GGUFTypeName = Object.freeze({
  0:'UINT8', 1:'INT8', 2:'UINT16', 3:'INT16',
  4:'UINT32', 5:'INT32', 6:'FLOAT32', 7:'BOOL',
  8:'STRING', 9:'ARRAY', 10:'UINT64', 11:'INT64', 12:'FLOAT64',
});

const GGMLTypeName = Object.freeze({
  0:'F32', 1:'F16', 2:'Q4_0', 3:'Q4_1',
  6:'Q5_0', 7:'Q5_1', 8:'Q8_0', 9:'Q8_1',
  10:'Q2_K', 11:'Q3_K_S', 12:'Q3_K_M', 13:'Q3_K_L',
  14:'Q4_K_S', 15:'Q4_K_M', 16:'Q5_K_S', 17:'Q5_K_M',
  18:'Q6_K', 19:'Q8_K', 20:'IQ2_XXS', 21:'IQ2_XS',
  22:'IQ3_XXS', 23:'IQ1_S', 24:'IQ4_NL', 25:'IQ3_S',
  26:'IQ2_S', 27:'IQ4_XS', 28:'I8', 29:'I16',
  30:'I32', 31:'I64', 32:'F64', 33:'IQ1_M', 34:'BF16',
});

// ─────────────────────────────────────────────────────────────────────────────
// Reader
// ─────────────────────────────────────────────────────────────────────────────

class GGUFParser {
  constructor (buffer) {
    // buffer may be ArrayBuffer or a plain-array converted from main process
    if (Array.isArray(buffer)) {
      const u8 = new Uint8Array(buffer);
      this._buf  = u8.buffer;
      this._view = new DataView(u8.buffer);
    } else {
      this._buf  = buffer;
      this._view = new DataView(buffer);
    }
    this._off = 0;
    this.version = 0;
  }

  // ── Primitives ──────────────────────────────────────────────────────────────

  _u8  () { const v = this._view.getUint8   (this._off); this._off += 1; return v; }
  _i8  () { const v = this._view.getInt8    (this._off); this._off += 1; return v; }
  _u16 () { const v = this._view.getUint16  (this._off, true); this._off += 2; return v; }
  _i16 () { const v = this._view.getInt16   (this._off, true); this._off += 2; return v; }
  _u32 () { const v = this._view.getUint32  (this._off, true); this._off += 4; return v; }
  _i32 () { const v = this._view.getInt32   (this._off, true); this._off += 4; return v; }
  _f32 () { const v = this._view.getFloat32 (this._off, true); this._off += 4; return v; }
  _f64 () { const v = this._view.getFloat64 (this._off, true); this._off += 8; return v; }
  _u64 () { const v = this._view.getBigUint64(this._off, true); this._off += 8; return v; }
  _i64 () { const v = this._view.getBigInt64 (this._off, true); this._off += 8; return v; }

  _str () {
    const len = this.version === 1 ? this._u32() : Number(this._u64());
    const raw = new Uint8Array(this._buf, this._off, len);
    this._off += len;
    return new TextDecoder().decode(raw);
  }

  // For v1, counts are uint32; for v2/v3 they are uint64
  _count () { return this.version === 1 ? BigInt(this._u32()) : this._u64(); }

  _value (type) {
    switch (type) {
      case GGUFType.UINT8:   return this._u8();
      case GGUFType.INT8:    return this._i8();
      case GGUFType.UINT16:  return this._u16();
      case GGUFType.INT16:   return this._i16();
      case GGUFType.UINT32:  return this._u32();
      case GGUFType.INT32:   return this._i32();
      case GGUFType.FLOAT32: return this._f32();
      case GGUFType.BOOL:    return this._u8() !== 0;
      case GGUFType.STRING:  return this._str();
      case GGUFType.ARRAY:   return this._array();
      case GGUFType.UINT64:  return this._u64();
      case GGUFType.INT64:   return this._i64();
      case GGUFType.FLOAT64: return this._f64();
      default: throw new Error(`Unknown GGUF value type: ${type}`);
    }
  }

  _array () {
    const itemType = this._u32();
    const count    = Number(this._count());
    const items    = [];
    for (let i = 0; i < count; i++) items.push(this._value(itemType));
    return { itemType, items };
  }

  // ── Public API ──────────────────────────────────────────────────────────────

  parse () {
    // Magic
    const magic = this._view.getUint32(0, true);
    if (magic !== GGUF_MAGIC_LE) {
      throw new Error(
        `Invalid GGUF magic: expected 0x${GGUF_MAGIC_LE.toString(16).toUpperCase()} ` +
        `got 0x${magic.toString(16).toUpperCase()}`
      );
    }
    this._off = 4;

    this.version = this._u32();
    if (this.version < 1 || this.version > 3) {
      throw new Error(`Unsupported GGUF version: ${this.version} (supported: 1–3)`);
    }

    const tensorCount = Number(this._count());
    const metaCount   = Number(this._count());

    // Metadata KVs
    const metadata = new Map();
    for (let i = 0; i < metaCount; i++) {
      const key       = this._str();
      const valueType = this._u32();
      const value     = this._value(valueType);
      metadata.set(key, {
        key,
        valueType,
        typeName: GGUFTypeName[valueType] ?? `UNKNOWN(${valueType})`,
        value,
        modified: false,
      });
    }

    // Tensor infos
    const tensors = [];
    for (let i = 0; i < tensorCount; i++) {
      const name   = this._str();
      const nDims  = this._u32();
      const dims   = [];
      for (let d = 0; d < nDims; d++) dims.push(this._u64());
      const ggmlType = this._u32();
      const offset   = this._u64();
      tensors.push({
        name,
        dims,
        ggmlType,
        typeName: GGMLTypeName[ggmlType] ?? `UNKNOWN(${ggmlType})`,
        offset,
        // derived
        numElements: dims.reduce((a, d) => a * d, 1n),
      });
    }

    const headerEnd   = this._off;
    const alignment   = Number(metadata.get('general.alignment')?.value ?? 32n) || 32;
    const dataOffset  = Math.ceil(headerEnd / alignment) * alignment;

    return {
      version:     this.version,
      tensorCount,
      metaCount,
      metadata,   // Map<string, entry>
      tensors,
      headerEnd,
      dataOffset,
      alignment,
    };
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Writer — converts parsed (+ modified) data back to a Uint8Array header
// ─────────────────────────────────────────────────────────────────────────────

class GGUFWriter {
  constructor (parsed) {
    this._parsed  = parsed;
    this._version = parsed.version;
    this._chunks  = [];
  }

  // ── Primitives ──────────────────────────────────────────────────────────────

  _pu8  (v) { const b = new Uint8Array(1); b[0] = v & 0xFF; this._chunks.push(b); }
  _pi8  (v) { this._pu8(v); }
  _pu16 (v) { const b = new Uint8Array(2); new DataView(b.buffer).setUint16(0, v, true); this._chunks.push(b); }
  _pi16 (v) { const b = new Uint8Array(2); new DataView(b.buffer).setInt16(0, v, true); this._chunks.push(b); }
  _pu32 (v) { const b = new Uint8Array(4); new DataView(b.buffer).setUint32(0, v, true); this._chunks.push(b); }
  _pi32 (v) { const b = new Uint8Array(4); new DataView(b.buffer).setInt32(0, v, true); this._chunks.push(b); }
  _pf32 (v) { const b = new Uint8Array(4); new DataView(b.buffer).setFloat32(0, v, true); this._chunks.push(b); }
  _pf64 (v) { const b = new Uint8Array(8); new DataView(b.buffer).setFloat64(0, v, true); this._chunks.push(b); }
  _pu64 (v) { const b = new Uint8Array(8); new DataView(b.buffer).setBigUint64(0, typeof v === 'bigint' ? v : BigInt(v), true); this._chunks.push(b); }
  _pi64 (v) { const b = new Uint8Array(8); new DataView(b.buffer).setBigInt64(0, typeof v === 'bigint' ? v : BigInt(v), true); this._chunks.push(b); }

  _pStr (s) {
    const enc = new TextEncoder().encode(s);
    if (this._version === 1) this._pu32(enc.length);
    else                      this._pu64(BigInt(enc.length));
    this._chunks.push(enc);
  }

  _pCount (n) {
    if (this._version === 1) this._pu32(n);
    else                      this._pu64(BigInt(n));
  }

  _pValue (type, value) {
    switch (type) {
      case GGUFType.UINT8:   this._pu8(value);  break;
      case GGUFType.INT8:    this._pi8(value);  break;
      case GGUFType.UINT16:  this._pu16(value); break;
      case GGUFType.INT16:   this._pi16(value); break;
      case GGUFType.UINT32:  this._pu32(value); break;
      case GGUFType.INT32:   this._pi32(value); break;
      case GGUFType.FLOAT32: this._pf32(value); break;
      case GGUFType.BOOL:    this._pu8(value ? 1 : 0); break;
      case GGUFType.STRING:  this._pStr(value); break;
      case GGUFType.ARRAY:
        this._pu32(value.itemType);
        this._pCount(value.items.length);
        for (const item of value.items) this._pValue(value.itemType, item);
        break;
      case GGUFType.UINT64:  this._pu64(value); break;
      case GGUFType.INT64:   this._pi64(value); break;
      case GGUFType.FLOAT64: this._pf64(value); break;
      default: throw new Error(`Cannot encode type ${type}`);
    }
  }

  // ── Public API ──────────────────────────────────────────────────────────────

  build () {
    const p = this._parsed;

    // Magic
    this._pu32(GGUF_MAGIC_LE);
    // Version
    this._pu32(p.version);
    // Counts
    this._pCount(p.tensorCount);
    this._pCount(p.metadata.size);

    // Metadata
    for (const [key, entry] of p.metadata) {
      this._pStr(key);
      this._pu32(entry.valueType);
      this._pValue(entry.valueType, entry.value);
    }

    // Tensor infos
    for (const t of p.tensors) {
      this._pStr(t.name);
      this._pu32(t.dims.length);
      for (const d of t.dims) this._pu64(d);
      this._pu32(t.ggmlType);
      this._pu64(t.offset);
    }

    // Concatenate
    const total = this._chunks.reduce((s, c) => s + c.length, 0);
    const out   = new Uint8Array(total);
    let   off   = 0;
    for (const c of this._chunks) { out.set(c, off); off += c.length; }
    return out;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers exported to renderer
// ─────────────────────────────────────────────────────────────────────────────

function formatBytes (n) {
  if (n < 1024)                 return `${n} B`;
  if (n < 1024 * 1024)         return `${(n / 1024).toFixed(2)} KiB`;
  if (n < 1024 * 1024 * 1024)  return `${(n / 1048576).toFixed(2)} MiB`;
  return `${(n / 1073741824).toFixed(2)} GiB`;
}

function formatBigInt (v) {
  return typeof v === 'bigint' ? v.toString() : String(v);
}

function valueToDisplayString (entry) {
  const { valueType, value } = entry;
  if (valueType === GGUFType.ARRAY) {
    return `[${value.itemType in GGUFTypeName ? GGUFTypeName[value.itemType] : '?'} × ${value.items.length}]`;
  }
  if (typeof value === 'bigint') return value.toString();
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  if (typeof value === 'number') {
    return Number.isInteger(value) ? value.toString() : value.toPrecision(7);
  }
  return String(value);
}
