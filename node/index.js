import { readFileSync } from "node:fs";
import { resolve, dirname, isAbsolute } from "node:path";
import { createHash } from "node:crypto";
import { gunzipSync } from "node:zlib";
import { blake2b } from "@noble/hashes/blake2b";

const folds = JSON.parse(
  readFileSync(new URL("./casefold.json", import.meta.url), "utf8"),
);
const fold = (text) =>
  Array.from(text.normalize("NFC"), (c) => folds[c] ?? c).join("");
const compare = (a, b) => {
  const x = Array.from(a),
    y = Array.from(b);
  for (let i = 0; i < Math.min(x.length, y.length); i++) {
    const d = x[i].codePointAt(0) - y[i].codePointAt(0);
    if (d) return d;
  }
  return x.length - y.length;
};
function words(text, ignore) {
  let normalized = fold(text);
  for (const template of ignore)
    normalized = normalized.split(fold(template)).join(" ");
  return normalized.match(/[\p{L}\p{N}_]+/gu) ?? [];
}
function shingles(text, n, ignore) {
  const tokens = words(text, ignore),
    result = new Set();
  for (let i = 0; i <= tokens.length - n; i++)
    result.add(tokens.slice(i, i + n).join(" "));
  return result;
}
export function parseRecord(record, id) {
  if (!record || typeof record !== "object" || Array.isArray(record))
    throw Error("record must be an object");
  const string = (value) => {
    if (typeof value !== "string") throw Error("content must be a string");
    return value;
  };
  let messages;
  if ("messages" in record || "conversations" in record) {
    const share = !("messages" in record),
      rows = record[share ? "conversations" : "messages"];
    if (!Array.isArray(rows) || !rows.length)
      throw Error("conversation must be a nonempty list");
    messages = rows.map((row) => {
      if (!row || typeof row !== "object")
        throw Error("message must be an object");
      const raw = string(row[share ? "from" : "role"]),
        role = { human: "user", gpt: "assistant" }[raw] ?? raw;
      if (!["system", "user", "assistant", "tool"].includes(role))
        throw Error("unknown role");
      return { role, content: string(row[share ? "value" : "content"]) };
    });
  } else if ("instruction" in record && "output" in record) {
    const input = string(record.input === undefined ? "" : record.input);
    messages = [
      {
        role: "user",
        content: string(record.instruction) + (input ? "\n" + input : ""),
      },
      { role: "assistant", content: string(record.output) },
    ];
  } else throw Error("expected openai-chat, alpaca or sharegpt record");
  return { id, messages };
}
export function readJsonl(path) {
  if (/\.(xz|bz2)$/.test(path))
    throw Error(
      "Node reader supports plain JSONL and gzip; use Python for xz/bz2",
    );
  let bytes = readFileSync(path);
  if (path.endsWith(".gz")) bytes = gunzipSync(bytes);
  const text = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  return text.split(/\r?\n/).flatMap((line, i) => {
    if (!line.trim()) return [];
    try {
      return [parseRecord(JSON.parse(line), `${path}:${i + 1}`)];
    } catch (error) {
      throw Error(`${path}:${i + 1}: ${error.message}`);
    }
  });
}
const text = (e) => e.messages.map((m) => m.content).join("\n");
function keys(group, numPerm, bands) {
  const signature = [];
  for (let perm = 0; perm < numPerm; perm++) {
    const salt = new Uint8Array(16);
    salt[0] = perm >> 8;
    salt[1] = perm & 255;
    let min = null;
    for (const gram of group) {
      const digest = blake2b(new TextEncoder().encode(gram), {
        dkLen: 8,
        salt,
      });
      const val = Buffer.from(digest).toString("hex");
      if (min === null || val < min) min = val;
    }
    signature.push(min);
  }
  const rows = numPerm / bands;
  return Array.from(
    { length: bands },
    (_, b) => `${b}:${signature.slice(b * rows, (b + 1) * rows).join(",")}`,
  );
}
export function checkContamination(
  train,
  evaluation,
  {
    ngram = 13,
    threshold = 0.8,
    shingle = 5,
    numPerm = 64,
    bands = 16,
    ignoreTemplate = [],
    failOn = ["EXACT", "NEAR"],
  } = {},
) {
  if (
    !Number.isInteger(ngram) ||
    ngram < 1 ||
    !Number.isInteger(shingle) ||
    shingle < 1 ||
    !Number.isInteger(numPerm) ||
    numPerm < 1 ||
    numPerm > 65536 ||
    !Number.isInteger(bands) ||
    bands < 1 ||
    numPerm % bands ||
    !(threshold > 0 && threshold <= 1)
  )
    throw Error("invalid matcher parameters");
  if (!failOn.every((level) => ["EXACT", "NEAR"].includes(level)))
    throw Error("invalid failOn");
  for (const side of [train, evaluation])
    if (new Set(side.map((e) => e.id)).size !== side.length)
      throw Error("duplicate example IDs");
  const hits = [],
    index = new Map();
  evaluation.forEach((e, i) => {
    for (const gram of shingles(text(e), ngram, ignoreTemplate)) {
      if (!index.has(gram)) index.set(gram, []);
      index.get(gram).push(i);
    }
  });
  for (const e of train) {
    const evidence = new Map();
    for (const gram of [...shingles(text(e), ngram, ignoreTemplate)].sort(
      compare,
    ))
      for (const i of index.get(gram) ?? [])
        if (!evidence.has(i)) evidence.set(i, gram);
    for (const i of [...evidence.keys()].sort((a, b) => a - b))
      hits.push({
        train_id: e.id,
        eval_id: evaluation[i].id,
        level: "EXACT",
        value: 1.0,
        evidence: evidence.get(i),
      });
  }
  const groups = evaluation.map((e) =>
      shingles(text(e), shingle, ignoreTemplate),
    ),
    buckets = new Map();
  groups.forEach((group, i) => {
    if (group.size)
      for (const key of keys(group, numPerm, bands)) {
        if (!buckets.has(key)) buckets.set(key, []);
        buckets.get(key).push(i);
      }
  });
  for (const e of train) {
    const group = shingles(text(e), shingle, ignoreTemplate);
    if (!group.size) continue;
    const candidates = new Set(
      keys(group, numPerm, bands).flatMap((key) => buckets.get(key) ?? []),
    );
    for (const i of [...candidates].sort((a, b) => a - b)) {
      const shared = [...group].filter((s) => groups[i].has(s)),
        similarity =
          shared.length / (group.size + groups[i].size - shared.length);
      if (similarity >= threshold)
        hits.push({
          train_id: e.id,
          eval_id: evaluation[i].id,
          level: "NEAR",
          value: similarity,
          evidence: shared.sort(compare)[0],
        });
    }
  }
  return {
    hits,
    train_records: train.length,
    eval_records: evaluation.length,
    fail_on: failOn,
    ok: !hits.some((h) => failOn.includes(h.level)),
  };
}
export function serializeReport(report) {
  const sorted = (value) =>
    Array.isArray(value)
      ? value.map(sorted)
      : value && typeof value === "object"
        ? Object.fromEntries(
            Object.keys(value)
              .sort()
              .map((key) => [key, sorted(value[key])]),
          )
        : value;
  // Python's float fields retain .0 in golden reports.
  return (
    JSON.stringify(sorted(report), null, 2).replace(
      /("value": )(-?\d+)(,?\n)/g,
      "$1$2.0$3",
    ) + "\n"
  );
}
export function verifyManifest(path) {
  const data = JSON.parse(readFileSync(path, "utf8"));
  const entries = [
      ...data.inputs,
      ...data.eval_sets,
      ...data.outputs,
      ...(data.parent ? [data.parent] : []),
    ],
    errors = [];
  for (const entry of entries) {
    const target =
      data.path_base === "manifest" && !isAbsolute(entry.path)
        ? resolve(dirname(path), entry.path)
        : entry.path;
    try {
      const hash = createHash("sha256")
        .update(readFileSync(target))
        .digest("hex");
      if (hash !== entry.sha256) errors.push(`hash mismatch: ${entry.path}`);
    } catch {
      errors.push(`unreadable: ${entry.path}`);
    }
  }
  return { ok: !errors.length, checked: entries.length, errors };
}
