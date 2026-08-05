import { readFile } from "node:fs/promises";
import { pathToFileURL } from "node:url";

export function validateDashboardData(data) {
  const errors = [];
  if (!data || typeof data !== "object") return ["root must be an object"];
  if (data.schemaVersion !== 1) errors.push("schemaVersion must equal 1");
  if (!data.dataset || typeof data.dataset !== "object") errors.push("dataset must be an object");
  if (!Array.isArray(data.analyses) || data.analyses.length === 0) {
    errors.push("analyses must contain at least one verified result");
  }

  for (const [index, analysis] of (data.analyses ?? []).entries()) {
    if (!analysis?.transcript?.id) errors.push(`analyses[${index}].transcript.id is required`);
    for (const field of ["positiveProbability", "neutralProbability", "negativeProbability"]) {
      const value = analysis?.finbert?.[field];
      if (value !== null && (!Number.isFinite(value) || value < 0 || value > 1)) {
        errors.push(`analyses[${index}].finbert.${field} must be null or a probability`);
      }
    }
  }

  function checkFinite(value, path = "root") {
    if (typeof value === "number" && !Number.isFinite(value)) {
      errors.push(`${path} contains a non-finite number`);
      return;
    }
    if (Array.isArray(value)) {
      value.forEach((item, index) => checkFinite(item, `${path}[${index}]`));
    } else if (value && typeof value === "object") {
      Object.entries(value).forEach(([key, item]) => checkFinite(item, `${path}.${key}`));
    }
  }

  checkFinite(data);
  return errors;
}

export async function validateFile(path) {
  const data = JSON.parse(await readFile(path, "utf8"));
  const errors = validateDashboardData(data);
  if (errors.length) throw new Error(`Invalid dashboard data:\n- ${errors.join("\n- ")}`);
  return data;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const path = process.argv[2] ?? "public/data/dashboard.json";
  await validateFile(path);
  console.log(`Validated ${path}`);
}
