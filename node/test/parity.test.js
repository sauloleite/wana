import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  checkContamination,
  readJsonl,
  serializeReport,
  parseRecord,
} from "../index.js";
import { fileURLToPath } from "node:url";
import { resolve } from "node:path";
process.chdir(fileURLToPath(new URL("../../", import.meta.url)));
test("shared golden report is byte identical", () => {
  const report = checkContamination(
    readJsonl("tests/fixtures/train.jsonl"),
    readJsonl("tests/fixtures/valid.jsonl"),
  );
  assert.equal(
    serializeReport(report),
    readFileSync("tests/fixtures/contamination.json", "utf8"),
  );
});
test("normalization, params, template removal, empty input", () => {
  const a = parseRecord({ instruction: "STRAẞE CAFÉ Ⅳ", output: "ok" }, "a");
  const b = parseRecord(
    { instruction: "strasse cafe\u0301 ⅳ", output: "ok" },
    "b",
  );
  assert.equal(
    checkContamination([a], [b], { ngram: 2 }).hits[0].level,
    "EXACT",
  );
  assert.equal(checkContamination([], []).ok, true);
  assert.throws(() => checkContamination([a, a], [b]));
  assert.throws(() => checkContamination([a], [b], { ngram: 0 }));
  assert.throws(() =>
    parseRecord({ messages: [{ role: "user", content: null }] }, "bad"),
  );
  assert.equal(
    checkContamination([a], [b], {
      ngram: 2,
      shingle: 2,
      ignoreTemplate: ["STRASSE CAFÉ Ⅳ"],
    }).hits.length,
    0,
  );
});
test("shared multilingual and near-match fixtures", () => {
  const cases = JSON.parse(readFileSync("tests/fixtures/parity.json", "utf8"));
  for (const fixture of cases) {
    assert.deepEqual(
      checkContamination(
        fixture.train.map((r) => parseRecord(r, "a")),
        fixture.evaluation.map((r) => parseRecord(r, "b")),
        fixture.options,
      ),
      fixture.report,
    );
  }
});
test("manifest hash verification detects tampering", async () => {
  const { mkdtempSync, writeFileSync, rmSync } = await import("node:fs");
  const { tmpdir } = await import("node:os");
  const { createHash } = await import("node:crypto");
  const { verifyManifest } = await import("../index.js");
  const dir = mkdtempSync(resolve(tmpdir(), "wana-node-"));
  try {
    const artifact = resolve(dir, "data.jsonl"),
      manifest = resolve(dir, "manifest.json");
    writeFileSync(artifact, "hello\n");
    writeFileSync(
      manifest,
      JSON.stringify({
        inputs: [],
        eval_sets: [],
        outputs: [
          {
            path: artifact,
            sha256: createHash("sha256").update("hello\n").digest("hex"),
          },
        ],
      }),
    );
    assert.equal(verifyManifest(manifest).ok, true);
    writeFileSync(artifact, "tampered");
    assert.equal(verifyManifest(manifest).ok, false);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
