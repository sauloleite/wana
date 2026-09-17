#!/usr/bin/env node
import { writeFileSync } from "node:fs";
import {
  readJsonl,
  checkContamination,
  serializeReport,
  verifyManifest,
} from "./index.js";
try {
  const args = process.argv.slice(2),
    command = args.shift();
  if (command === "--version") console.log("wana 0.5.0");
  else if (command === "verify") {
    if (args.length !== 1) throw Error("usage: wana verify manifest.json");
    const report = verifyManifest(args[0]);
    console.log(JSON.stringify(report, null, 2));
    process.exitCode = report.ok ? 0 : 1;
  } else if (command === "check") {
    const train = [],
      evaluation = [],
      options = {},
      ignoreTemplate = [];
    let output;
    const value = () => {
      if (!args.length) throw Error("missing option value");
      return args.shift();
    };
    while (args.length) {
      const arg = args.shift();
      if (arg === "--eval") evaluation.push(value());
      else if (arg === "-o") output = value();
      else if (arg === "--ngram") options.ngram = Number(value());
      else if (arg === "--shingle") options.shingle = Number(value());
      else if (arg === "--threshold") options.threshold = Number(value());
      else if (arg === "--ignore-template") ignoreTemplate.push(value());
      else if (arg === "--fail-on")
        options.failOn = value().split(",").filter(Boolean);
      else if (arg.startsWith("-")) throw Error(`unknown option: ${arg}`);
      else train.push(arg);
    }
    if (!train.length || !evaluation.length)
      throw Error(
        "usage: wana check train.jsonl --eval valid.jsonl [-o report.json]",
      );
    const report = checkContamination(
      train.flatMap(readJsonl),
      evaluation.flatMap(readJsonl),
      { ...options, ignoreTemplate },
    );
    const serialized = serializeReport(report);
    if (output) {
      const { resolve } = await import("node:path");
      if ([...train, ...evaluation].some((p) => resolve(p) === resolve(output)))
        throw Error("output would overwrite input");
      writeFileSync(output, serialized);
    } else process.stdout.write(serialized);
    process.exitCode = report.ok ? 0 : 1;
  } else throw Error("usage: wana check | verify | --version");
} catch (error) {
  console.error(`wana: ${error.message}`);
  process.exitCode = 2;
}
