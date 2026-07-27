// Realistic VectorSeam tuner sample data.
// Each entry mirrors the real `RoundOutput` JSON the tuner publishes to
// calibrations/<cohort>/latest.json (+ historical round-*.json). Swap this
// out by setting window.VECTORSEAM_CONFIG = { baseUrl, cohorts: [...] }.

const EF_GRID = [10, 20, 40, 60, 80, 100, 150, 200, 300, 400];

function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const r4 = (x) => Math.round(x * 1e4) / 1e4;
const r3 = (x) => Math.round(x * 1e3) / 1e3;
const r2 = (x) => Math.round(x * 1e2) / 1e2;
const iso = (sec) => new Date(sec * 1000).toISOString().replace(/\.\d{3}Z$/, "Z");
const ASSURANCE = 0.9;
const confidenceFor = (quantile, n) => 1 / (1 + Math.exp(-(quantile - 0.9) * Math.sqrt(Math.max(1, n)) * 5));

// Compliance quantile recall as a function of ef. The generated training
// confidence tightens as samples accumulate, allowing progressively lower ef.
function recallAt(ef, targetEf) {
  const x = ef / targetEf;
  return Math.min(0.999, 0.99 - 0.45 * Math.exp(-1.9 * x));
}

const END_BASE = Math.floor(Date.parse("2026-07-24T07:49:00Z") / 1000);
const INTERVAL = 60;
const ROUNDS = 28;

function genCohort(cfg) {
  const rnd = mulberry32(cfg.seed);
  const out = [];
  let effective = null;

  for (let i = 0; i < ROUNDS; i++) {
    const roundEnd = END_BASE - (ROUNDS - 1 - i) * INTERVAL;
    const computedAt = iso(roundEnd);
    const windowStart = iso(roundEnd - 600);
    const windowEnd = iso(roundEnd);

    // Unique samples grow as closed storage windows accumulate, saturating at
    // the deduplicated query-pool ceiling.
    const windowsAvail = Math.min(10, i);
    const raw = windowsAvail * cfg.perWindow;
    let unique = Math.round(cfg.cap * (1 - Math.exp(-raw / cfg.cap)));
    if (i === 0) unique = Math.round(4 + 4 * rnd());
    unique = Math.min(cfg.cap, unique);

    // Per-ef sweep, with a small per-round wobble on the effective target so
    // recommended_ef occasionally steps by one grid position while sparse.
    const jitter = (i < 6 ? (rnd() - 0.5) * 0.5 : (rnd() - 0.5) * 0.12);
    const effTarget = cfg.targetEf * (1 + jitter);
    const per_ef = EF_GRID.map((ef) => {
      const q = Math.min(0.999, Math.max(0.4, recallAt(ef, effTarget) + (rnd() - 0.5) * 0.008));
      const mean = Math.min(0.999, q + 0.006 + 0.012 * rnd());
      const lat = (cfg.latBase + cfg.latSlope * ef) * (0.95 + 0.1 * rnd());
      return { ef, quantile_recall: r4(q), mean_recall: r4(mean), latency_p50_ms: r2(lat) };
    });

    const transient = cfg.transientAt === i;
    const failed = transient ? Math.round(18 + 8 * rnd()) : Math.round(1 + 4 * rnd());
    const measured = Math.round(unique * (1 + 0.08 * rnd()));
    const available = measured + failed + Math.round(unique * 0.05);
    const train = Math.round(unique * 0.7);
    const test = unique - train;
    const trainCeiling = 1 - Math.pow(0.9, train + 1);
    const testCeiling = 1 - Math.pow(0.9, test + 1);
    const insufficient = transient || train === 0 || test === 0 || trainCeiling < ASSURANCE || testCeiling < ASSURANCE;
    const trainScores = per_ef.map((p) => ({ ef: p.ef, confidence: confidenceFor(p.quantile_recall, train) }));
    const cleared = trainScores.find((p) => p.confidence >= ASSURANCE);
    const highestClears = trainScores[trainScores.length - 1].confidence >= ASSURANCE;
    const recommended = highestClears ? cleared.ef : EF_GRID[EF_GRID.length - 1];
    const recSummary = per_ef.find((p) => p.ef === recommended);
    const selectedTrain = trainScores.find((p) => p.ef === recommended);
    const status = insufficient ? "insufficient_samples" : highestClears ? "ok" : "target_unmet";

    let confidence = null;
    let train_confidence = null;
    let train_compliance = null;
    let test_compliance = null;
    let recommended_ef = null;
    let train_q = null;
    let test_q = null;
    let error = null;

    if (!insufficient) {
      recommended_ef = recommended;
      train_confidence = r4(selectedTrain.confidence);
      train_q = r4(Math.min(0.999, recSummary.quantile_recall + 0.004 + 0.01 * rnd()));
      test_q = r4(Math.max(0.85, recSummary.quantile_recall - 0.002 - 0.005 * rnd()));
      train_compliance = r4(Math.max(0, Math.min(1, 0.9 + (train_q - 0.9) * 1.5)));
      test_compliance = r4(Math.max(0, Math.min(1, 0.9 + (test_q - 0.9) * 1.5)));
      confidence = r4(Math.min(0.995, confidenceFor(test_q, test) * (0.985 + 0.03 * rnd())));
      if (status === "ok" && confidence >= ASSURANCE) {
        effective = { recommended_ef, confidence, source_round: computedAt, carried: false };
      } else if (effective) effective = { ...effective, carried: true };
    } else {
      if (transient) {
        error = `holdout validation failed: ${failed} statement timeouts left only ${Math.max(0, unique - failed)} measured samples`;
      }
      if (effective) effective = { ...effective, carried: true };
    }

    const windowsWithParts = i === 0 ? 0 : Math.min(10, Math.max(1, Math.round(10 * Math.min(1, windowsAvail / 10 + 0.05))));

    out.push({
      format_version: 1,
      cohort: cfg.name,
      computed_at: computedAt,
      window: { start: windowStart, end: windowEnd, duration_seconds: 600 },
      target: { name: "demo_recall", k: 10, value: 0.9, percentile: 0.9, confidence: ASSURANCE },
      index: cfg.index,
      ef_grid: EF_GRID.slice(),
      status,
      error,
      recommended_ef,
      confidence,
      train_confidence,
      train_compliance,
      train_quantile_recall: train_q,
      test_compliance,
      test_quantile_recall: test_q,
      effective: effective ? { ...effective } : null,
      samples: { available, measured, failed, unique, train, test },
      dropped_frame_fraction: r4(rnd() * 0.014),
      coverage: {
        empty_window_fraction: r3(1 - windowsWithParts / 10),
        windows_in_scope: 10,
        windows_with_parts: windowsWithParts,
      },
      parts_used: windowsWithParts,
      incompatible_parts: 0,
      ground_truth_latency_mean_ms: r2(cfg.gtLat * (0.95 + 0.1 * rnd())),
      per_ef,
    });
  }
  return out;
}

const CONFIGS = [
  { name: "superuser", index: "superuser", targetEf: 60, latBase: 6, latSlope: 0.055, cap: 2000, perWindow: 300, gtLat: 430, seed: 11, transientAt: null },
  { name: "products/search", index: "products", targetEf: 100, latBase: 11, latSlope: 0.11, cap: 2600, perWindow: 300, gtLat: 610, seed: 23, transientAt: null },
  { name: "support/tickets", index: "tickets", targetEf: 40, latBase: 4, latSlope: 0.04, cap: 780, perWindow: 90, gtLat: 280, seed: 37, transientAt: 20 },
];

export const SAMPLE_COHORTS = CONFIGS.map((c) => c.name);
export const SAMPLE_STORE = CONFIGS.reduce((acc, cfg) => {
  const rounds = genCohort(cfg);
  acc[cfg.name] = { rounds, latest: rounds[rounds.length - 1] };
  return acc;
}, {});
export const EF_GRID_VALUES = EF_GRID;
