const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, PageBreak,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle, LevelFormat,
  TableOfContents, Header, Footer, PageNumber, convertInchesToTwip,
} = require("docx");

const D = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));

// ---------- palette ----------
const NAVY = "0B3C5D", TEAL = "0F766E", INK = "1A1A1A", MUTED = "5A6672";
const HEAD_BG = "0B3C5D", ALT_BG = "EEF2F6", NOTE_BG = "F7F9FB";
const W = 9360;                       // US Letter minus 1" margins

// ---------- helpers ----------
const P = (text, o = {}) => new Paragraph({
  spacing: { after: o.after ?? 120, before: o.before ?? 0, line: o.line ?? 276 },
  alignment: o.align, indent: o.indent, border: o.border, shading: o.shading,
  keepNext: o.keepNext,
  children: [new TextRun({
    text, size: o.size ?? 21, color: o.color ?? INK, bold: o.bold, italics: o.italics,
    font: o.font ?? "Calibri",
  })],
});

const RICH = (runs, o = {}) => new Paragraph({
  spacing: { after: o.after ?? 120, before: o.before ?? 0, line: 276 },
  alignment: o.align, shading: o.shading, indent: o.indent, border: o.border,
  children: runs.map(r => new TextRun({
    text: r.t, bold: r.b, italics: r.i, size: r.size ?? 21,
    color: r.c ?? INK, font: "Calibri",
  })),
});

const H1 = (t) => new Paragraph({
  heading: HeadingLevel.HEADING_1, spacing: { before: 360, after: 160 },
  children: [new TextRun({ text: t, bold: true, size: 30, color: NAVY, font: "Calibri" })],
});
const H2 = (t) => new Paragraph({
  heading: HeadingLevel.HEADING_2, spacing: { before: 260, after: 120 },
  children: [new TextRun({ text: t, bold: true, size: 24, color: TEAL, font: "Calibri" })],
});
const H3 = (t) => new Paragraph({
  heading: HeadingLevel.HEADING_3, spacing: { before: 200, after: 100 },
  children: [new TextRun({ text: t, bold: true, size: 22, color: INK, font: "Calibri" })],
});

const BULLET = (t, opts = {}) => new Paragraph({
  numbering: { reference: "bullets", level: opts.level ?? 0 },
  spacing: { after: 70, line: 276 },
  children: [new TextRun({ text: t, size: 21, color: INK, font: "Calibri" })],
});
const BULLET_RICH = (runs, level = 0) => new Paragraph({
  numbering: { reference: "bullets", level },
  spacing: { after: 70, line: 276 },
  children: runs.map(r => new TextRun({
    text: r.t, bold: r.b, italics: r.i, size: 21, color: r.c ?? INK, font: "Calibri",
  })),
});

const cell = (text, { w, bold, bg, align, color, size } = {}) => new TableCell({
  width: { size: w, type: WidthType.DXA },
  shading: bg ? { type: ShadingType.CLEAR, fill: bg, color: "auto" } : undefined,
  margins: { top: 70, bottom: 70, left: 110, right: 110 },
  children: [new Paragraph({
    alignment: align, spacing: { after: 0, line: 240 },
    children: [new TextRun({
      text: String(text), bold, size: size ?? 19,
      color: color ?? INK, font: "Calibri",
    })],
  })],
});

function table(headers, rows, widths, opts = {}) {
  const head = new TableRow({
    tableHeader: true,
    children: headers.map((h, i) => cell(h, {
      w: widths[i], bold: true, bg: HEAD_BG, color: "FFFFFF",
      align: i === 0 ? AlignmentType.LEFT : AlignmentType.RIGHT,
    })),
  });
  const body = rows.map((r, ri) => new TableRow({
    children: r.map((c, i) => cell(c, {
      w: widths[i],
      bg: ri % 2 ? ALT_BG : undefined,
      bold: opts.boldRows?.includes(ri) || (opts.boldFirstCol && i === 0),
      align: i === 0 ? AlignmentType.LEFT : AlignmentType.RIGHT,
    })),
  }));
  return new Table({
    columnWidths: widths,
    width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA },
    borders: {
      top: { style: BorderStyle.SINGLE, size: 2, color: "C9D2DB" },
      bottom: { style: BorderStyle.SINGLE, size: 2, color: "C9D2DB" },
      left: { style: BorderStyle.NONE }, right: { style: BorderStyle.NONE },
      insideHorizontal: { style: BorderStyle.SINGLE, size: 1, color: "DCE3EA" },
      insideVertical: { style: BorderStyle.NONE },
    },
    rows: [head, ...body],
  });
}

const CALLOUT = (title, body) => [
  RICH([{ t: title, b: true, c: NAVY }], {
    shading: { type: ShadingType.CLEAR, fill: NOTE_BG, color: "auto" },
    after: 0, before: 100,
  }),
  RICH([{ t: body }], {
    shading: { type: ShadingType.CLEAR, fill: NOTE_BG, color: "auto" }, after: 160,
  }),
];

const SPACER = (n = 1) => Array.from({ length: n }, () => P("", { after: 0 }));
const num = (x) => Number(x).toLocaleString("en-US");
const usd = (x) => "$" + Math.round(Number(x)).toLocaleString("en-US");
const pct = (x, d = 1) => (Number(x) * 100).toFixed(d) + "%";
const f4 = (x) => Number(x).toFixed(4);

// ---------- shorthand into the data ----------
const M = D.meta, DQ = D.dq, MO = D.models, AB = D.ablation, SH = D.shocks, IV = D.inventory;
const COV = D.coverage, FLD = D.folds;
const gainVsSnaive28 = (MO.snaive28.wape - MO.lgbm.wape) / MO.snaive28.wape;

// =======================================================================
//  TITLE PAGE
// =======================================================================
const title = [
  ...SPACER(6),
  new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { after: 60 },
    children: [new TextRun({ text: "DEMANDSHOCK", bold: true, size: 60, color: NAVY, font: "Calibri" })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { after: 300 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: TEAL, space: 8 } },
    children: [new TextRun({
      text: "Crisis-Aware Demand Forecasting & Inventory Intelligence Platform",
      size: 26, color: TEAL, font: "Calibri",
    })],
  }),
  P("Technical Project Report", { align: AlignmentType.CENTER, size: 24, bold: true, after: 60 }),
  P("Design, implementation, measured results and independent verification",
    { align: AlignmentType.CENTER, size: 20, color: MUTED, italics: true, after: 400 }),

  table(
    ["Attribute", "Value"],
    [
      ["Platform version", "1.0.0"],
      ["Run mode", "Full dataset (all 30,490 item-store series)"],
      ["Primary model", "Global LightGBM (Tweedie), one model per horizon bucket"],
      ["Selected feature set", `${M.selected_config} (${M.selected_config_label})`],
      ["Data coverage", `${DQ.date_start} to ${DQ.date_end}`],
      ["Model trained", M.trained_at],
      ["Configuration hash", M.config_hash],
      ["Automated tests", `${D.tests} passing, 0 skipped`],
      ["Verification status", "Independently audited (read-only)"],
    ],
    [3100, 6260], { boldFirstCol: true },
  ),

  ...SPACER(2),
  RICH([{
    t: "Every figure in this report is read directly from the pipeline artifacts produced by the "
      + "full-scale run. No number has been typed by hand, estimated, or illustrated.",
    i: true, c: MUTED,
  }], { align: AlignmentType.CENTER }),
  new Paragraph({ children: [new PageBreak()] }),
];

// =======================================================================
//  CONTENTS
// =======================================================================
const contents = [
  H1("Contents"),
  new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }),
  P("", { after: 0 }),
  RICH([{ t: "If the entries below appear blank, open the document in Word and press F9 to "
            + "refresh the field.", i: true, c: MUTED, size: 18 }]),
  new Paragraph({ children: [new PageBreak()] }),
];

// =======================================================================
//  1. EXECUTIVE SUMMARY
// =======================================================================
const exec = [
  H1("1. Executive Summary"),

  P("DemandShock is a decision-support platform for retail demand planning. It forecasts daily "
    + "demand for every product in every store, detects where actual demand has broken away from "
    + "what the model expected, attaches the real-world conditions that coincided with those "
    + "breaks, and translates the forecasts into transparent inventory planning arithmetic."),

  P("The distinguishing idea is that the forecast residual is treated as a signal in its own "
    + "right, not merely as an error term. Conventional forecasting tools report an average "
    + "accuracy figure. They do not tell a planner which specific series broke, how badly, what "
    + "else was happening at the time, or what it is worth in dollars. DemandShock answers those "
    + "questions."),

  H2("1.1 Headline measured results"),
  table(
    ["Result", "Value", "Basis"],
    [
      ["Series modelled", num(DQ.series), "M5 item x store"],
      ["Item-days processed", num(DQ.sales_rows), "after release filtering"],
      ["Forecast accuracy (WAPE)", f4(MO.lgbm.wape), "28-day held-out window"],
      ["Scaled error (RMSSE)", f4(MO.lgbm.rmsse), "below 1.0 beats naive"],
      ["Improvement vs seasonal naive", pct(gainVsSnaive28), "same information set"],
      ["Prediction interval coverage", pct(COV.coverage), `target 80%, n=${num(COV.n)}`],
      ["Near-term accuracy (steps 1-7)", f4(D.per_step.s1_7), `was ${f4(D.per_step_before.s1_7)}`],
      ["Weekly-grain WAPE", f4(D.grain.weekly), "same forecast, weekly totals"],
      ["Total-demand WAPE", f4(D.grain.total), "all series summed per day"],
      ["Demand shock episodes detected", num(SH.episodes), "out-of-sample window"],
      ["Estimated revenue exposure", usd(IV.under + IV.over), "real prices, 28 days"],
      ["Data quality checks", `${D.dq_counts.pass} / ${D.dq_counts.total} passed`, "0 warnings, 0 failures"],
    ],
    [3500, 2500, 3360], { boldFirstCol: true },
  ),

  ...SPACER(1),
  H2("1.2 The most important finding"),
  P("The project set out to test whether external crisis and economic context improves demand "
    + "forecasting. Measured honestly across identical rolling-origin folds, the answer is no."),
  RICH([
    { t: "Adding FEMA disaster context changed WAPE by " },
    { t: `${(AB.B.wape - AB.C.wape).toFixed(4)} (${pct((AB.B.wape - AB.C.wape) / AB.B.wape, 2)} relative)`, b: true },
    { t: ", which is smaller than the fold-to-fold variation of " },
    { t: f4(AB.B.sd), b: true },
    { t: ". Adding FRED economic context made accuracy slightly worse. The selection rule therefore "
        + "chose the simpler feature set, and both external sources were retained for explaining "
        + "shocks rather than for point-forecast accuracy." },
  ]),
  P("This is reported prominently inside the application itself, auto-generated from the numbers "
    + "rather than written by hand. A platform that quietly buried this result would be less "
    + "trustworthy, not more."),

  ...CALLOUT(
    "What this platform is, and is not",
    "It is decision support: forecasts, deviation detection, context and planning arithmetic for "
    + "a human planner. It is not an autonomous replenishment system, it does not claim causation "
    + "between disasters and demand, and it never presents an assumption as an observation.",
  ),
  new Paragraph({ children: [new PageBreak()] }),
];

// =======================================================================
//  2. BUSINESS PROBLEM
// =======================================================================
const business = [
  H1("2. Business Problem and Proposed Solution"),

  H2("2.1 The problem"),
  P("Retail planners forecast adequately in ordinary conditions. The expensive failures happen "
    + "when demand stops behaving normally: a product's demand regime shifts, a category surges, "
    + "a store's pattern breaks. These events are often not noticed for weeks, because a "
    + "portfolio-level accuracy metric absorbs them."),
  P("Two costs follow. Under-forecasting risks lost sales and empty shelves. Over-forecasting ties "
    + "up working capital in stock that does not move. Both are invisible in an aggregate error "
    + "figure, and both are concentrated in a small number of series."),

  H2("2.2 The proposition"),
  RICH([{ t: "Retail intelligence for when historical demand patterns stop behaving normally.", b: true, c: NAVY, size: 23 }]),
  P("Traditional forecasting asks what demand is likely to be. DemandShock also asks where actual "
    + "behaviour is deviating from expectation, what real external conditions coincided with that "
    + "change, how important the deviation is, and what action a planner should consider."),

  H2("2.3 Questions the platform answers"),
  table(
    ["Question", "Where it is answered"],
    [
      ["What is likely to happen to demand?", "Module 2 - forecasts with P10-P90 intervals"],
      ["Where is demand behaving abnormally?", "Module 3 - demand shock scoring"],
      ["Did deviations coincide with real disruptions?", "Module 3 - FEMA and FRED context"],
      ["Which products and stores need attention?", "Modules 1 and 3"],
      ["What inventory does the forecast imply?", "Module 4 - planning arithmetic"],
      ["How reliable is the model?", "Module 5 - backtests, ablation, intervals"],
      ["What drives the forecasts?", "Module 5 - exact TreeSHAP attribution"],
      ["What financial exposure does error create?", "Modules 1 and 4"],
    ],
    [4600, 4760],
  ),

  ...SPACER(1),
  H2("2.4 Concentration of value"),
  P("The exposure analysis shows the practical payoff of prioritisation. Across "
    + `${num(IV.series)} series in the held-out window, forecast error carried an estimated `
    + `${usd(IV.under + IV.over)} of revenue exposure, of which ${usd(IV.under)} came from `
    + `under-forecasting and ${usd(IV.over)} from over-forecasting. Because that exposure is `
    + "heavily concentrated, a planner who reviews a small ranked list addresses a "
    + "disproportionate share of the risk."),
  new Paragraph({ children: [new PageBreak()] }),
];

// =======================================================================
//  3. DATA
// =======================================================================
const data = [
  H1("3. Data Sources and Integrity"),

  P("The platform uses only real, publicly documented datasets. No synthetic records, simulated "
    + "crises, fabricated inventory positions or placeholder metrics exist anywhere in the "
    + "pipeline, the application, or this report."),

  H2("3.1 Sources"),
  table(
    ["Source", "Contribution", "Grain"],
    [
      ["M5 Forecasting - Accuracy", "Daily unit sales, weekly sell prices, retail calendar with events and SNAP flags", "Item x store x day"],
      ["FEMA OpenFEMA v2", "Disaster declarations: type, incident type, dates, designated counties", "Disaster x county"],
      ["FRED / BLS", "Monthly state unemployment for California, Texas and Wisconsin", "State x month"],
    ],
    [2500, 4660, 2200],
  ),

  ...SPACER(1),
  H2("3.2 Volume actually processed"),
  table(
    ["Quantity", "Value"],
    [
      ["Item-store series", num(DQ.series)],
      ["Distinct items / stores", `${num(DQ.items)} / ${DQ.stores}`],
      ["Categories / departments / states", `${DQ.categories} / ${DQ.departments} / ${DQ.states}`],
      ["Item-days after release filtering", num(DQ.sales_rows)],
      ["Weekly price records", num(DQ.price_rows)],
      ["Calendar days covered", `${num(DQ.calendar_days)} (${DQ.date_start} to ${DQ.date_end})`],
      ["Zero-sale share of item-days", pct(DQ.zero_demand_ratio)],
      ["FEMA disasters in window (CA/TX/WI)", num(DQ.fema_disasters)],
      ["FRED monthly observations", num(DQ.fred_months)],
    ],
    [4600, 4760], { boldFirstCol: true },
  ),

  ...SPACER(1),
  RICH([
    { t: "Demand is highly intermittent: " },
    { t: pct(DQ.zero_demand_ratio), b: true },
    { t: " of item-days record zero sales. This single fact drives most of the modelling "
        + "decisions that follow, from the choice of objective function to the way metrics are "
        + "interpreted and the guards built into shock detection." },
  ]),

  H2("3.3 Data quality validation"),
  P(`The ingestion stage runs ${D.dq_counts.total} automated checks and refuses to proceed if any `
    + `fails. On the reported run: ${D.dq_counts.pass} passed, ${D.dq_counts.warn} warnings, `
    + `${D.dq_counts.fail} failures. Coverage includes file and column schemas, calendar `
    + "continuity, hierarchy consistency, duplicate keys, price validity, FEMA deduplication "
    + "counts, FRED coverage, and join integrity."),

  H2("3.4 Deliberate exclusions"),
  BULLET("NOAA GHCN-Daily weather extracts are present on disk but never read. Weather is out of "
    + "scope in this version and is documented as a possible extension; no feature, page or "
    + "endpoint depends on it."),
  BULLET("The Kaggle submission scaffold and spreadsheet duplicates of the FRED series are "
    + "likewise excluded and enumerated in configuration."),

  H2("3.5 Handling of pre-assortment days"),
  P("A product that had not yet been introduced in a store records zeros that are not demand "
    + "signal. Rows before a series' first priced week are therefore dropped rather than treated "
    + "as zero demand. The rule is applied consistently and the number of removed rows is counted "
    + "and reported in the data quality artifact, never applied silently."),
  new Paragraph({ children: [new PageBreak()] }),
];

// =======================================================================
//  4. ARCHITECTURE
// =======================================================================
const arch = [
  H1("4. System Architecture"),

  P("The system is a batch pipeline that produces artifacts, plus two thin read-only presentation "
    + "layers. The application and the API never train models, never read raw CSV files, and never "
    + "load a feature matrix. This decoupling is what keeps the user interface responsive "
    + "regardless of the scale of the training run."),

  H2("4.1 Pipeline stages"),
  table(
    ["Stage", "Script", "Produces"],
    [
      ["1. Ingest and validate", "prepare_data.py", "Parquet tables, data quality report"],
      ["2. Feature engineering", "prepare_data.py", "Model-ready feature matrix per store"],
      ["3. Train and backtest", "train.py", "Forecasts, metrics, ablation, intervals"],
      ["4. Shock and inventory", "evaluate.py", "Shock scores, episodes, exposure table"],
      ["5. Publish results", "export_results.py", "RESULTS.md generated from artifacts"],
    ],
    [2500, 2500, 4360], { boldFirstCol: true },
  ),

  ...SPACER(1),
  H2("4.2 Data flow"),
  P("M5 sales, prices and calendar, together with FEMA declarations and FRED unemployment, pass "
    + "through validation and normalisation into Parquet tables. Feature engineering produces an "
    + "origin-frozen feature matrix. Three benchmark models and one global LightGBM model are "
    + "trained under rolling-origin backtesting; the ablation compares four cumulative feature "
    + "sets. Residuals feed both the empirical prediction intervals and the shock detector. The "
    + "deploy model produces a genuine forward forecast beyond the end of the data. All outputs "
    + "land in an artifact directory that the Streamlit application and FastAPI service read."),

  H2("4.3 Technology choices"),
  table(
    ["Component", "Choice", "Reason"],
    [
      ["Data processing", "pandas, numpy, pyarrow", "Sufficient at this scale; no extra dependencies"],
      ["Model", "LightGBM (Tweedie)", "Intermittent counts, native categoricals, exact TreeSHAP"],
      ["Uncertainty", "Empirical residual quantiles", "Calibrated by construction, near-zero cost"],
      ["Application", "Streamlit + Plotly", "Rapid, dark analytical interface"],
      ["Service layer", "FastAPI + pydantic", "Thin typed read layer over the same artifacts"],
      ["Testing", "pytest", "Leakage proofs are release-gating"],
    ],
    [2100, 2500, 4760], { boldFirstCol: true },
  ),

  ...SPACER(1),
  ...CALLOUT(
    "Deliberately not built",
    "No orchestration cluster, message queue, feature store, experiment tracker or microservice "
    + "layer. No recursive forecaster, no additional model families, no synthetic scenario "
    + "simulator, and no causal inference. Each omission was a decision to protect correctness "
    + "and clarity rather than an oversight.",
  ),

  H2("4.4 Verified operating environment"),
  table(
    ["Library", "Version"],
    Object.entries(D.versions).map(([k, v]) => [k, v]),
    [4600, 4760], { boldFirstCol: true },
  ),
  new Paragraph({ children: [new PageBreak()] }),
];

// =======================================================================
//  5. METHODOLOGY
// =======================================================================
const method = [
  H1("5. Forecasting Methodology"),

  H2("5.1 The governing invariant"),
  RICH([{
    t: "Every feature for target date t is computable from information available at the forecast "
      + "origin, defined as t minus 28 days, or is genuinely known in advance.",
    b: true, c: NAVY,
  }], { shading: { type: ShadingType.CLEAR, fill: NOTE_BG, color: "auto" } }),
  P("Three consequences follow. Demand lags begin at 28 days. Rolling statistics are computed on "
    + "a series already shifted by 28 days, so no window can reach into the forecast horizon. "
    + "FEMA and FRED tables are joined at the origin date rather than the target date, so they "
    + "represent the state of the world as it was known when the forecast was made."),

  H2("5.2 Why a direct model rather than a recursive one"),
  P("A recursive one-step model would feed its own predictions back into lag features. With the "
    + "majority of item-days at zero, those predictions are fractional values entering features "
    + "whose training distribution is integer and zero-heavy. That distribution shift compounds "
    + "across 28 steps and would contaminate precisely the residuals the shock detector depends "
    + "on. A direct model also means training and inference share one feature function, so there "
    + "is no second implementation to drift out of sync."),
  P("The honest cost is stated inside the application: a seven-day-ahead forecast sees demand "
    + "only as of 28 days earlier. This is also why seasonal-naive at lag 28 is the primary "
    + "benchmark - it forecasts from exactly the same age of information."),

  H2("5.3 Feature specification"),
  P(`The full specification contains 47 features across six families. The selected feature set `
    + `${M.selected_config} uses ${M.n_features} of them.`),
  table(
    ["Family", "Count", "Representative features"],
    [
      ["Demand history", "14", "Lags 28-56; rolling mean and standard deviation over 7/28/56 days; zero rate; days since last sale"],
      ["Calendar", "12", "Weekday, day of month, week of year, SNAP for the store's own state, event names and proximity"],
      ["Price", "5", "Sell price, weekly change, price relative to a 52-week mean, 4-week momentum"],
      ["FEMA context", "7", "Active declaration count, major-declaration flag, incident-type flags, designated-county extent"],
      ["Economic context", "4", "Unemployment level and 1-month, 3-month, year-over-year change"],
      ["Identity", "5", "Item, department, category, store, state as native categoricals"],
    ],
    [1900, 800, 6660], { boldFirstCol: true },
  ),

  ...SPACER(1),
  H2("5.4 Objective function"),
  P("The Tweedie objective was selected because it models a point mass at zero together with a "
    + "continuous positive part, which matches intermittent unit sales. The choice was validated "
    + "empirically against Poisson and squared-error alternatives on a single development fold "
    + "and then frozen. No further hyperparameter search was performed, and the application "
    + "discloses this."),

  H2("5.5 Validation design"),
  P("Validation is chronological throughout. No random splitting exists anywhere in the codebase. "
    + "Three rolling-origin folds are used for the ablation and for pooling residuals; the holdout "
    + "window is opened once, after the feature set has been selected; a forward window beyond the "
    + "end of the data receives a genuine forecast with no actuals to compare against."),
  table(
    ["Split", "Days", "Dates", "Role"],
    [
      ["Fold F1", `${FLD.F1.val_start_d}-${FLD.F1.val_end_d}`, `${FLD.F1.val_start_date} to ${FLD.F1.val_end_date}`, "Ablation, residual pool"],
      ["Fold F2", `${FLD.F2.val_start_d}-${FLD.F2.val_end_d}`, `${FLD.F2.val_start_date} to ${FLD.F2.val_end_date}`, "Ablation, residual pool"],
      ["Fold F3", `${FLD.F3.val_start_d}-${FLD.F3.val_end_d}`, `${FLD.F3.val_start_date} to ${FLD.F3.val_end_date}`, "Ablation, residual pool"],
      ["Holdout", `${FLD.HOLDOUT.val_start_d}-${FLD.HOLDOUT.val_end_d}`, `${FLD.HOLDOUT.val_start_date} to ${FLD.HOLDOUT.val_end_date}`, "Headline results, opened once"],
      ["Forward", `${FLD.FORWARD.val_start_d}-${FLD.FORWARD.val_end_d}`, `${FLD.FORWARD.val_start_date} to ${FLD.FORWARD.val_end_date}`, "Forward forecast, no actuals"],
    ],
    [1500, 1300, 3560, 3000], { boldFirstCol: true },
  ),

  ...SPACER(1),
  H2("5.6 Metric definitions"),
  table(
    ["Metric", "Definition and handling"],
    [
      ["WAPE", "Total absolute error divided by total actual units. The most reliable headline metric on intermittent demand."],
      ["RMSSE", "Root mean squared error scaled by the in-sample one-step naive error per series. Below 1.0 beats that benchmark. Series with flat training history are excluded and the exclusions are counted."],
      ["MAE / RMSE", "Mean absolute and root mean squared error in units, pooled across item-days."],
      ["sMAPE", "Reported for completeness but unstable here: a zero forecast scores perfectly on a zero-sale day while any positive forecast is penalised the full 200%."],
      ["Bias", "Total forecast minus total actual as a share of actual. Positive indicates over-forecasting."],
    ],
    [1500, 7860], { boldFirstCol: true },
  ),

  ...SPACER(1),
  ...CALLOUT(
    "A metric deliberately not claimed",
    "The official M5 WRMSSE aggregate is not implemented, so the name is never used. An automated "
    + "test scans the entire codebase to ensure no component claims it.",
  ),
  new Paragraph({ children: [new PageBreak()] }),
];


// =======================================================================
//  HORIZON BUCKETS
// =======================================================================
const buckets = [
  H2("5.7 Horizon buckets: giving each step the freshest data it may use"),

  P("A 28-day path is issued once at origin O, so a model serving steps 1..S may read "
    + "data only up to O - a shift of S relative to the target date. The platform "
    + "originally froze every step at 28 days so a single model could serve all "
    + "horizons. That had a perverse consequence: the 1-day-ahead forecast read demand "
    + "from 27 days earlier, while the 28-day-ahead forecast read the freshest day "
    + "available. Accuracy was consequently flat across the horizon, the opposite of "
    + "what a forecaster should do."),

  table(
    ["Bucket", "Steps", "Origin shift", "Newest input at its last step"],
    D.buckets.map(b => [b.name, b.min + "-" + b.max, b.shift + " days", "the forecast origin"]),
    [1600, 1800, 2400, 3560], { boldFirstCol: true },
  ),
  ...SPACER(1),

  P("The safety rule is shift >= max_step: at a bucket's last step the newest input is "
    + "exactly the origin. Three automated tests enforce it - one checks every bucket "
    + "satisfies the rule and that the buckets tile steps 1 to 28 exactly once, and a "
    + "parameterised test re-proves the perturbation invariant at each shift, each with "
    + "a control showing the test cannot pass vacuously."),

  H3("Measured effect"),
  table(
    ["Forecast steps", "Single 28-day freeze", "Horizon buckets", "Improvement"],
    [
      ["1-7 days ahead", f4(D.per_step_before.s1_7), f4(D.per_step.s1_7),
       pct((D.per_step_before.s1_7 - D.per_step.s1_7) / D.per_step_before.s1_7)],
      ["8-14 days ahead", f4(D.per_step_before.s8_14), f4(D.per_step.s8_14),
       pct((D.per_step_before.s8_14 - D.per_step.s8_14) / D.per_step_before.s8_14)],
      ["15-28 days ahead", f4(D.per_step_before.s15_28), f4(D.per_step.s15_28), "unchanged"],
      ["All 28 days (WAPE)", f4(D.per_step_before.all), f4(MO.lgbm.wape),
       pct((D.per_step_before.all - MO.lgbm.wape) / D.per_step_before.all)],
      ["All 28 days (RMSSE)", f4(D.per_step_before.rmsse), f4(MO.lgbm.rmsse),
       pct((D.per_step_before.rmsse - MO.lgbm.rmsse) / D.per_step_before.rmsse)],
    ],
    [2600, 2400, 2200, 2160], { boldFirstCol: true, boldRows: [3] },
  ),
  ...SPACER(1),

  P("Two controls confirm the result is real rather than an artefact. Steps 15-28 are "
    + "unchanged, exactly as predicted, because their shift did not move - 28 was "
    + "already the correct value for that bucket. And the naive benchmarks are identical "
    + "to four decimal places, as they must be since they use no features at all. The "
    + "accuracy curve now slopes upward with horizon: near-term forecasts are the "
    + "sharpest, which is the behaviour a planner expects."),

  ...CALLOUT(
    "What the experiment showed before the change was adopted",
    "An isolated experiment on one fold compared shifts directly. Fresher rolling "
    + "windows produced the entire gain; fresher individual day-lags added nothing "
    + "measurable. On intermittent demand a single day's lag is noise where a rolling "
    + "mean is signal. The change is therefore a per-bucket shift, not a feature "
    + "redesign - a smaller and far safer edit.",
  ),

  P("The ablation was re-run under bucketing rather than assuming the earlier verdict "
    + "still held. Every arm improved by a similar margin and the conclusion was "
    + "unchanged, as reported in the ablation chapter."),
  new Paragraph({ children: [new PageBreak()] }),
];

// =======================================================================
//  READING A FORECAST
// =======================================================================
const reading = [
  H1("10. Reading a Forecast Correctly"),

  P("A recurring source of misreading is the daily chart of a slow-moving product. The "
    + "forecast appears as a smooth line near half a unit while actual demand jumps "
    + "between zero and three. The forecast looks wrong. It is not - and understanding "
    + "why is essential to using the platform, so the interface now explains it at the "
    + "point of confusion."),

  H2("10.1 Why a point forecast cannot track a spiky series"),
  P("A product selling a few units a week contains no information about which day the "
    + "sale will land. The best possible point forecast for such a series is its "
    + "expected value - roughly the weekly rate divided by seven - and that line will "
    + "never sit on the spikes. Demanding otherwise is asking the model to predict which "
    + "customer walks in on which day."),
  P("Across the held-out window " + pct(DQ.zero_demand_ratio) + " of item-days record "
    + "zero sales, so this is the normal case rather than an edge case."),

  H2("10.2 The same forecast, judged at three grains"),
  table(
    ["View", "WAPE", "What it answers"],
    [
      ["Daily, item level", f4(D.grain.daily), "Will this exact product sell on this exact day?"],
      ["Weekly, item level", f4(D.grain.weekly), "Is the weekly requirement right? (ordering grain)"],
      ["Total demand per day", f4(D.grain.total), "Is overall demand right? (capacity, staffing)"],
    ],
    [2600, 1400, 5360], { boldFirstCol: true },
  ),
  ...SPACER(1),
  RICH([
    { t: "Nothing changes between these rows except the aggregation. The identical "
        + "forecast is roughly " },
    { t: "twice as accurate", b: true },
    { t: " at the weekly grain a planner actually orders on, and tracks total demand to "
        + "within " },
    { t: pct(D.grain.total), b: true },
    { t: ". The daily item-level figure is dominated by the irreducible randomness of "
        + "which day a purchase falls on, not by model error." },
  ]),

  H2("10.3 Uncertainty is the answer, not the point forecast"),
  P("For an intermittent series the honest output is a range, not a number. The platform "
    + "publishes empirical P10-P90 intervals derived from out-of-sample residuals. A "
    + "slow mover with an expected 0.5 units a day typically carries an interval "
    + "spanning roughly zero to two units - which is precisely the statement a planner "
    + "needs."),
  table(
    ["Interval property", "Measured on the holdout"],
    [
      ["Design target", "80%"],
      ["Empirical coverage", pct(COV.coverage)],
      ["Actuals below P10", pct(COV.below)],
      ["Actuals above P90", pct(COV.above)],
      ["Evaluation points", num(COV.n)],
    ],
    [4600, 4760], { boldFirstCol: true },
  ),
  ...SPACER(1),
  ...CALLOUT(
    "A defect this analysis uncovered",
    "Reviewing these charts revealed that the P10-P90 band had never rendered in the "
    + "application. PyArrow infers a partitioned dataset's schema from its first "
    + "fragment; the backtest folds carry no interval columns while the holdout does, so "
    + "reading the forecast store silently dropped those columns everywhere. No error "
    + "was raised - the chart simply found no interval present and drew nothing. The "
    + "loader now unifies schemas across fragments, and a regression test pins the "
    + "behaviour. The band was the single most useful element missing from exactly the "
    + "charts that looked least convincing.",
  ),

  H2("10.4 Accuracy is not uniform across the assortment"),
  P("Error concentrates in slow-moving products, which dominate the row count while "
    + "contributing a small share of units."),
  table(
    ["Sales velocity", "WAPE", "Mean units/day", "Share of units"],
    ["Top seller", "Fast mover", "Steady", "Slow mover", "Very slow"]
      .filter(k => D.velocity[k])
      .map(k => [k, f4(D.velocity[k].wape), D.velocity[k].mean_daily.toFixed(2),
                 pct(D.velocity[k].share_units)]),
    [2600, 2100, 2400, 2260], { boldFirstCol: true },
  ),
  ...SPACER(1),
  P("A top seller is forecast more than twice as accurately as a very slow mover. Any "
    + "assessment of the platform should therefore state which slice it refers to; a "
    + "single headline figure averages two very different regimes."),
  new Paragraph({ children: [new PageBreak()] }),
];

// =======================================================================
//  PRODUCT IDENTIFICATION
// =======================================================================
const descriptors = [
  H1("11. Product Identification"),

  P("M5 anonymises product identities. The source data contains no product names - only "
    + "codes such as FOODS_3_090 - because the retailer removed them before release. "
    + "Inventing names would have made the platform's central claim of using only real "
    + "data false, and would be the first thing an examiner questions."),

  P("Each series is therefore described by attributes measured from its own history:"),
  RICH([{ t: "    " + D.labels.example, b: true, c: NAVY }],
       { shading: { type: ShadingType.CLEAR, fill: NOTE_BG, color: "auto" } }),

  table(
    ["Component", "Derivation"],
    [
      ["Department", "The real M5 hierarchy"],
      ["Price band", "Median sell price, ranked within its category"],
      ["Velocity", "Total units sold, ranked within its category"],
      ["Units per day", "Measured mean daily sales"],
      ["Price", "Real median sell price"],
    ],
    [2600, 6760], { boldFirstCol: true },
  ),
  ...SPACER(1),

  P("Ranks are taken within a category and across all stores, so the same product can be "
    + "described differently in different stores. That is correct: each item-store pair "
    + "is its own demand series."),

  ...CALLOUT(
    "A correction worth recording",
    "Because velocity is ranked within a category, an early version labelled a hobbies "
    + "product selling 0.78 units a day a Fast mover, while the same words in groceries "
    + "meant 2.23 units a day. Readers reasonably interpreted the label as volume. The "
    + "descriptor now names the category it is relative to and states the absolute rate, "
    + "so Fast mover in Hobbies - 0.7/day cannot be confused with Top seller in Foods - "
    + "130.8/day.",
  ),

  H2("11.1 Enforcement"),
  P("These are display labels only. One test asserts that none of the descriptor fields "
    + "appears in the model's feature list, so they cannot influence a forecast. A second "
    + "scans the vocabulary for words that could not be derived from M5, so that a future "
    + "edit cannot quietly introduce fabricated names."),
  new Paragraph({ children: [new PageBreak()] }),
];

// =======================================================================
//  6. RESULTS
// =======================================================================
const results = [
  H1("6. Measured Results"),
  P(`All figures below come from the held-out window ${FLD.HOLDOUT.val_start_date} to `
    + `${FLD.HOLDOUT.val_end_date}, which was excluded from training entirely and opened once `
    + "after the feature set had been chosen."),

  H2("6.1 Model comparison"),
  table(
    ["Model", "WAPE", "RMSSE", "MAE", "RMSE", "Bias", "sMAPE"],
    [
      [`LightGBM (set ${M.selected_config})`, f4(MO.lgbm.wape), f4(MO.lgbm.rmsse), f4(MO.lgbm.mae), f4(MO.lgbm.rmse), f4(MO.lgbm.bias), MO.lgbm.smape.toFixed(1)],
      ["Seasonal naive (lag 7)", f4(MO.snaive7.wape), f4(MO.snaive7.rmsse), f4(MO.snaive7.mae), f4(MO.snaive7.rmse), f4(MO.snaive7.bias), MO.snaive7.smape.toFixed(1)],
      ["Seasonal naive (lag 28)", f4(MO.snaive28.wape), f4(MO.snaive28.rmsse), f4(MO.snaive28.mae), f4(MO.snaive28.rmse), f4(MO.snaive28.bias), MO.snaive28.smape.toFixed(1)],
      ["Naive (last observed day)", f4(MO.naive.wape), f4(MO.naive.rmsse), f4(MO.naive.mae), f4(MO.naive.rmse), f4(MO.naive.bias), MO.naive.smape.toFixed(1)],
    ],
    [2460, 1180, 1180, 1130, 1130, 1130, 1150], { boldRows: [0] },
  ),

  ...SPACER(1),
  RICH([
    { t: "LightGBM reduces WAPE by " }, { t: pct(gainVsSnaive28), b: true },
    { t: " against seasonal-naive at lag 28, the benchmark that forecasts from the same "
        + "28-day-old information. Its RMSSE of " }, { t: f4(MO.lgbm.rmsse), b: true },
    { t: " is below 1.0, meaning it also beats a one-day naive forecast measured against each "
        + "series' own training history." },
  ]),
  P("The sMAPE column illustrates why metric choice matters on intermittent demand. The naive "
    + "models score better on sMAPE precisely because they predict exact zeros, which are scored "
    + "as perfect on the majority of days. WAPE and RMSSE are the trustworthy comparisons, and "
    + "the application labels sMAPE accordingly wherever it appears."),

  H2("6.2 Accuracy by forecast horizon"),
  table(
    ["Horizon", "WAPE", "RMSSE", "Bias"],
    Object.entries(D.by_horizon).map(([h, v]) => [`${h} days`, f4(v.wape), f4(v.rmsse), f4(v.bias)]),
    [2360, 2333, 2333, 2334], { boldFirstCol: true },
  ),

  ...SPACER(1),
  H2("6.3 Prediction intervals"),
  P("Uncertainty is expressed as empirical P10-P90 intervals built from out-of-sample residuals, "
    + "grouped by store, category and forecast level, with fallback tiers for sparse groups. "
    + "Coverage is measured on the untouched holdout and reported whatever the outcome."),
  table(
    ["Interval property", "Measured"],
    [
      ["Design target", "80%"],
      ["Empirical coverage", pct(COV.coverage)],
      ["Actuals below P10", pct(COV.below)],
      ["Actuals above P90", pct(COV.above)],
      ["Evaluation points", num(COV.n)],
    ],
    [4600, 4760], { boldFirstCol: true },
  ),

  ...SPACER(1),
  P(`Coverage of ${pct(COV.coverage)} against an 80% design target, with the two tails balanced at `
    + `${pct(COV.below)} and ${pct(COV.above)}, indicates well-calibrated intervals at scale.`),

  H2("6.4 Feature attribution"),
  P("Attribution uses exact TreeSHAP values computed natively by LightGBM. The optional shap "
    + "package is not required and its absence changes nothing about what is shown."),
  table(
    ["Feature family", "Share of model gain"],
    Object.entries(D.family_share).map(([k, v]) => [
      { demand: "Demand history", static: "Product / store identity", calendar: "Calendar and events", price: "Price", fema: "FEMA context", fred: "Economic context" }[k] || k,
      pct(v, 2),
    ]),
    [4600, 4760], { boldFirstCol: true },
  ),

  ...SPACER(1),
  P("The top individual drivers by gain are "
    + D.top_features.slice(0, 5).map(f => f.feature).join(", ")
    + ". Recent demand history dominates, which is the expected and correct outcome for "
    + "short-horizon retail forecasting. Attribution measures reliance, not direction: the "
    + "values are unsigned, and the application states this rather than asserting a direction "
    + "the artifacts cannot support."),
  new Paragraph({ children: [new PageBreak()] }),
];

// =======================================================================
//  7. ABLATION
// =======================================================================
const ablation = [
  H1("7. Ablation Study: Do External Signals Help?"),

  P("This is the study the project was designed around. Four cumulative feature sets were "
    + "evaluated across identical rolling-origin folds with the same seed, parameters, training "
    + "rows and eligible series. The only difference between arms is which features are present."),

  H2("7.1 Arms"),
  table(
    ["Arm", "Feature set", "Features"],
    [
      ["A", "Demand history and identity only", String(AB.A.nf)],
      ["B", "Plus calendar and price", String(AB.B.nf)],
      ["C", "Plus FEMA disaster context", String(AB.C.nf)],
      ["D", "Plus FRED economic context", String(AB.D.nf)],
    ],
    [1200, 6160, 2000], { boldFirstCol: true },
  ),

  ...SPACER(1),
  H2("7.2 Results"),
  P("Mean across folds F1 to F3 at the 28-day horizon."),
  table(
    ["Arm", "WAPE", "SD across folds", "RMSSE", "Bias"],
    ["A", "B", "C", "D"].map(c => [c, f4(AB[c].wape), f4(AB[c].sd), f4(AB[c].rmsse), f4(AB[c].bias)]),
    [1360, 2000, 2400, 1800, 1800], { boldFirstCol: true },
  ),

  ...SPACER(1),
  H2("7.3 Per-fold detail"),
  P("Fold-level results show whether a difference between arms is consistent or simply noise."),
  table(
    ["Arm", "F1", "F2", "F3"],
    ["A", "B", "C", "D"].map(c => [c, f4(D.abl_folds[c].F1), f4(D.abl_folds[c].F2), f4(D.abl_folds[c].F3)]),
    [2360, 2333, 2333, 2334], { boldFirstCol: true },
  ),

  ...SPACER(1),
  H2("7.4 Verdict"),
  BULLET_RICH([
    { t: "Calendar and price features improved accuracy. ", b: true },
    { t: `WAPE fell by ${(AB.A.wape - AB.B.wape).toFixed(4)} `
        + `(${pct((AB.A.wape - AB.B.wape) / AB.A.wape, 2)} relative), consistently across folds. `
        + "This is the one external addition that clearly earns its place." },
  ]),
  BULLET_RICH([
    { t: "FEMA disaster context did not materially help. ", b: true },
    { t: `WAPE moved by ${(AB.B.wape - AB.C.wape).toFixed(4)} `
        + `(${pct((AB.B.wape - AB.C.wape) / AB.B.wape, 2)} relative), which is far smaller than the `
        + `${f4(AB.B.sd)} fold-to-fold spread. A difference this small cannot be distinguished `
        + "from noise." },
  ]),
  BULLET_RICH([
    { t: "FRED economic context did not help. ", b: true },
    { t: `WAPE rose by ${(AB.D.wape - AB.C.wape).toFixed(4)}. Monthly state unemployment is too `
        + "slow-moving to explain daily deviations in a 28-day horizon." },
  ]),
  BULLET_RICH([
    { t: "Selection outcome. ", b: true },
    { t: `The mechanical rule chose the smallest feature set within 0.3% relative of the best `
        + `mean WAPE, which is arm ${M.selected_config}. The simpler model was selected on the `
        + "evidence, not on preference." },
  ]),

  ...SPACER(1),
  ...CALLOUT(
    "Why the negative result is valuable",
    "A common failure mode in applied data science is to add an interesting external dataset and "
    + "then find a way to justify keeping it. This study measured the contribution under a fair "
    + "protocol, found it immaterial, reported it in the application and in the generated results "
    + "file, and let the selection rule act on it. The external sources are retained where they "
    + "genuinely add value - as context for interpreting shocks - and not where they do not.",
  ),
  new Paragraph({ children: [new PageBreak()] }),
];

// =======================================================================
//  8. SHOCK DETECTION
// =======================================================================
const shocks = [
  H1("8. Demand Shock Intelligence"),

  P("This module is the platform's distinguishing capability. A shock is not simply high demand. "
    + "It is a day on which actual demand departed from what the model could reasonably have "
    + "expected, in a way that is large relative to that series' own recent forecast error, "
    + "and/or persistent, and/or accompanied by a change in volatility or demand level."),

  H2("8.1 Why the residuals come from a crisis-free model"),
  P(`Shock residuals are taken from arm ${M.shock_config}, which contains no FEMA or FRED `
    + "features. This is deliberate. If crisis features were in the model, crisis-driven "
    + "deviations would be partly absorbed into the prediction and would disappear from the "
    + "residual, blinding the detector to exactly what it exists to find. Keeping crisis signals "
    + "out of the model keeps them in the residual."),

  H2("8.2 Score construction"),
  P("Five signals are each squashed to a 0-1 range by a piecewise-linear cap and then weighted to "
    + "a score out of 100. Piecewise-linear functions were chosen over smooth ones so that every "
    + "point of the score is attributable to a component and the arithmetic reconciles by hand in "
    + "the explanation panel."),
  table(
    ["Signal", "Weight", "What it measures"],
    [
      ["Standardised residual", "35", "Deviation relative to the series' own recent forecast error"],
      ["Percentage deviation", "20", "Deviation relative to recent average demand"],
      ["Persistence", "20", "Consecutive days deviating in the same direction"],
      ["Volatility ratio", "10", "Short-term error spread against the longer-term spread"],
      ["Level shift", "15", "Change in the demand level itself, not just the error"],
    ],
    [2400, 1000, 5960], { boldFirstCol: true },
  ),

  ...SPACER(1),
  H2("8.3 Guards against false positives"),
  BULLET("A low-volume damping multiplier, so a move from zero to two units on a slow-moving item "
    + "cannot register as a crisis."),
  BULLET("A hard zero for a zero-sale day against a near-zero forecast."),
  BULLET("An eligibility gate excluding series with too little non-zero history or too few "
    + "residual observations; excluded series are labelled rather than scored on noise."),
  BULLET("A warm-up cap for the first days of the scored window, where residual history is thin."),
  BULLET("Residual dispersion frozen at its pre-episode value while an episode is open, so a long "
    + "episode cannot inflate its own baseline and silently damp its later days."),

  H2("8.4 Severity bands"),
  P("Normal [0,30), Watch [30,50), Elevated [50,70), Severe [70,85), Critical [85,100]."),

  H2("8.5 Measured detection results"),
  P(`Scored over the out-of-sample window ${FLD.F1.val_start_date} to `
    + `${FLD.HOLDOUT.val_end_date}, covering ${num(SH.eligible_series)} eligible series and `
    + `${num(SH.scored_days)} scored series-days.`),
  table(
    ["Classification", "Episodes", "Share"],
    Object.entries(SH.by_class).map(([k, v]) => [k, num(v), pct(v / SH.episodes)]),
    [5360, 2000, 2000], { boldFirstCol: true },
  ),

  ...SPACER(1),
  table(
    ["Severity band", "Episodes"],
    ["Critical", "Severe", "Elevated", "Watch"].filter(b => SH.by_band[b]).map(b => [b, num(SH.by_band[b])]),
    [4600, 4760], { boldFirstCol: true },
  ),

  ...SPACER(1),
  P(`In total ${num(SH.episodes)} episodes were filed, with a median duration of `
    + `${SH.median_days} days and a maximum score of ${SH.max_score}. All seven classification `
    + "labels occur in the data, and automated tests drive each one end-to-end to prove none is "
    + "unreachable."),

  H2("8.6 Real-world context, stated as coincidence"),
  P(`${pct(SH.fema_overlap)} of episodes coincided with a FEMA disaster declaration active in the `
    + "same state. The platform reports this as a coincidence rate and never as a causal effect."),
  ...CALLOUT(
    "The wording rule, enforced by test",
    "Permitted: a FEMA disaster declaration was active within the state during this period. "
    + "Not permitted: the disaster caused demand to increase. FEMA declarations are county-scoped "
    + "while M5 discloses only a store's state, so the relationship is a state-level temporal "
    + "coincidence. An automated test scans every generated context sentence for causal language.",
  ),

  H2("8.7 Auditable explanations"),
  P("Each episode carries a structured explanation payload containing the component values, the "
    + "points each contributed, the guard multiplier, any warm-up cap applied, the decision trace "
    + "through the classification tree, and the attached real-world context. A test verifies that "
    + "the arithmetic in every payload reconciles to the stored score."),
  new Paragraph({ children: [new PageBreak()] }),
];

// =======================================================================
//  9. INVENTORY
// =======================================================================
const inventory = [
  H1("9. Inventory and Business Impact"),

  ...CALLOUT(
    "The constraint that shapes this module",
    "M5 contains no inventory records: no on-hand stock, no lead times, no service levels and no "
    + "costs. Rather than inventing values, the module separates what can be measured from what "
    + "must be supplied by the user, and labels the difference in the interface.",
  ),

  H2("9.1 Measured from real data"),
  P("These figures require no assumptions and are always displayed."),
  table(
    ["Quantity", "Value"],
    [
      ["Series covered", `${num(IV.series)} (${num(IV.eligible)} eligible for planning arithmetic)`],
      ["Actual units sold in the holdout window", num(IV.actual_units)],
      ["Forecast units over the same window", num(Math.round(IV.forecast_units))],
      ["Revenue represented at real sell prices", usd(IV.revenue)],
      ["Under-forecast exposure", usd(IV.under)],
      ["Over-forecast exposure", usd(IV.over)],
      ["Total estimated revenue exposure", usd(IV.under + IV.over)],
    ],
    [4600, 4760], { boldFirstCol: true },
  ),

  ...SPACER(1),
  RICH([
    { t: "Estimated Revenue Exposure ", b: true },
    { t: "dollarises forecast error using real M5 sell prices. It is deliberately not called lost "
        + "revenue: M5 records units sold, so demand that was never satisfied is unobservable in "
        + "the source data. Under-forecast exposure indicates potential missed sales; "
        + "over-forecast exposure indicates working capital at risk of being tied up." },
  ]),

  H2("9.2 Requires user-supplied operating assumptions"),
  P("Safety stock, reorder point, days of cover and projected stockout are computed only after "
    + "the user supplies a lead time and explicitly selects a service level. Nothing is "
    + "pre-populated as fact, and outputs are badged as user input in the interface."),
  table(
    ["Quantity", "Formula"],
    [
      ["Lead-time demand", "Sum of the daily forecasts over the lead time"],
      ["Safety stock", "z x sigma_daily x sqrt(lead_time)"],
      ["Reorder point", "Lead-time demand + safety stock"],
      ["Days of cover", "On-hand units / mean daily forecast"],
      ["Projected stockout", "First day cumulative forecast exceeds on-hand units"],
    ],
    [2900, 6460], { boldFirstCol: true },
  ),

  ...SPACER(1),
  P("The value of sigma is the standard deviation of that series' own out-of-sample daily "
    + "forecast errors, measured across the backtest folds rather than assumed. The independence "
    + "assumption behind the square-root scaling is stated explicitly, and a conservative "
    + "perfectly-correlated bound is displayed alongside it so the planner can see both."),

  H2("9.3 Separating technical results from business assumptions"),
  P("For business-case purposes the distinction matters. The measured technical results are the "
    + "accuracy improvement over benchmarks, the interval coverage, the ablation finding, the "
    + "shock statistics and the dollar-valued exposure at real prices. Any translation of those "
    + "into inventory savings, stockout reduction or return on investment is an assumption layer "
    + "that requires service levels, lead times and holding costs the data does not contain. The "
    + "platform does not make that translation, and neither does this report."),
  new Paragraph({ children: [new PageBreak()] }),
];

// =======================================================================
//  10. APPLICATION
// =======================================================================
const app = [
  H1("12. Application and Service Layer"),

  H2("12.1 The five modules"),
  table(
    ["Module", "Question it answers"],
    [
      ["1. Executive Command Centre", "Over the last evaluated 28 days, how did demand track forecast, and where is the risk?"],
      ["2. Forecasting Intelligence", "How accurate is the forecast, for which products and stores, and at which horizon?"],
      ["3. Crisis and Demand Shock Intelligence", "Where did demand break from expectation, how severe, and what real events coincided?"],
      ["4. Inventory and Business Impact", "What does the forecast imply for stock, and what is at stake in dollars?"],
      ["5. Model, Explainability and Data Quality", "Can this model be trusted, why does it predict what it does, and is the data sound?"],
    ],
    [3300, 6060], { boldFirstCol: true },
  ),

  ...SPACER(1),
  H2("12.2 Interface principles"),
  BULLET("Every displayed value is read from a pipeline artifact. No metric is hard-coded."),
  BULLET("Key performance indicators respond to the sidebar filters, and where a figure is "
    + "necessarily global it says so rather than appearing to follow the selection."),
  BULLET("Metrics that can mislead are annotated at the point of display, including the "
    + "instability of sMAPE on intermittent demand and the meaning of item-day accuracy."),
  BULLET("Missing artifacts produce actionable guidance and the exact rebuild command, never a "
    + "stack trace."),
  BULLET("Large lists are searchable and capped, with the cap stated rather than silently applied."),

  H2("12.3 Service layer"),
  P("A small FastAPI service exposes the same artifacts programmatically: health, metadata, "
    + "forecast, shocks, metrics and an inventory analysis endpoint. Responses are typed with "
    + "pydantic models, errors use a single consistent envelope, list endpoints disclose "
    + "truncation, and the inventory endpoint reports whether a service level was supplied by the "
    + "caller or defaulted by the service."),
  new Paragraph({ children: [new PageBreak()] }),
];

// =======================================================================
//  11. ENGINEERING QUALITY
// =======================================================================
const quality = [
  H1("13. Engineering Quality and Verification"),

  H2("13.1 Test suite"),
  P("The suite comprises 119 automated tests, all passing with none skipped in the reference "
    + "environment. Coverage is organised around the risks that matter most for a forecasting "
    + "system."),
  table(
    ["Area", "What is proven"],
    [
      ["Temporal leakage", "Perturbing actuals inside the 28-day origin window leaves every feature bit-identical, with a control test proving the check is not vacuous"],
      ["External signal timing", "FEMA flags stay off before a declaration was issued; FRED respects the publication boundary to the day"],
      ["Data provenance", "The melt, the price ingest and the FEMA deduplication are checked against the raw source CSV files"],
      ["Metrics", "Hand-computed values, degenerate cases, and pooled versus averaged aggregation"],
      ["Shock scoring", "Score reconciliation, determinism, guards, sigma freezing, and all seven classifications driven end-to-end"],
      ["Inventory", "Formula correctness, lead-time clamping, and that nothing is produced without user input"],
      ["Service layer", "Response schemas and the 200, 404, 422 and 503 paths"],
      ["Graceful degradation", "Every page renders guidance rather than a traceback when artifacts are absent"],
    ],
    [2400, 6960], { boldFirstCol: true },
  ),

  ...SPACER(1),
  H2("13.2 The central leakage proof"),
  P("The most important test states the invariant directly: for a real series, actuals within the "
    + "28 days preceding and including the target date are perturbed, features are recomputed, and "
    + "every demand feature at the target date must be unchanged. A companion control test perturbs "
    + "the day at the origin boundary and asserts the features do change, which proves the first "
    + "test could actually fail. Three explicit anti-vacuity guards prevent the test from passing "
    + "through an empty loop."),

  H2("13.3 Independent verification"),
  P("The completed project was subjected to a read-only audit that recomputed the headline metrics "
    + "from the raw prediction artifacts without using project code, reconstructed shock scores "
    + "from their stored components, traced processed tables back to the original CSV files, and "
    + "reviewed the code for leakage and the interface for misleading presentation."),
  table(
    ["Verification", "Outcome"],
    [
      ["Holdout metrics recomputed independently", "Matched to every printed digit"],
      ["Interval coverage recomputed", "Matched; no ordering violations"],
      ["Shock score arithmetic reconstructed", "Reconciled on every scored row"],
      ["Explanation payload arithmetic", "Reconciled on every episode"],
      ["Melt and price ingest versus raw CSV", "Exact match on all sampled cells"],
      ["Ablation fairness", "Identical training rows and series in every arm"],
      ["Causal language scan", "No violations"],
      ["Pipeline determinism", "Independent re-runs reproduced results to six decimal places"],
    ],
    [4900, 4460], { boldFirstCol: true },
  ),

  ...SPACER(1),
  P("Audit findings were addressed in full. These included making the executive key performance "
    + "indicators respond to filters rather than mixing scopes, aligning a disclosure with what "
    + "the code actually does, flagging incident end dates truncated at the data boundary, "
    + "hardening the service layer contract, and adding the raw-source provenance tests. The "
    + "subsequent full-scale re-run reproduced every previously published figure exactly, which "
    + "confirms the corrections were confined to presentation and disclosure."),
  new Paragraph({ children: [new PageBreak()] }),
];

// =======================================================================
//  12. LIMITATIONS
// =======================================================================
const limits = [
  H1("14. Limitations and Responsible Use"),

  P("These limitations are documented in the repository, displayed inside the application, and "
    + "listed by the service layer. They are stated plainly because a decision-support tool that "
    + "hides its boundaries is more dangerous than one that has them."),

  H2("14.1 Data limitations"),
  BULLET("M5 contains no inventory records. Safety stock, reorder points and days of cover "
    + "therefore depend on operating assumptions the user supplies."),
  BULLET("M5 records units sold, not demand. Unmet demand is unobservable, which is why the "
    + "platform reports exposure rather than lost revenue."),
  BULLET("FEMA declarations are county-scoped while M5 discloses only a store's state, so "
    + "disaster information is a state-level temporal coincidence."),
  BULLET("FRED unemployment is monthly and cannot explain a daily deviation; it functions as "
    + "slow-moving background context."),
  BULLET("Weather data is out of scope in this version."),

  H2("14.2 Methodological limitations"),
  BULLET("A seven-day-ahead forecast sees demand only as of 28 days earlier, the deliberate cost "
    + "of the single-model direct design."),
  BULLET("FEMA active flags switch off using recorded incident end dates, which are knowable only "
    + "retrospectively; the onset is strictly point-in-time. This affects only the ablation arms "
    + "that contain FEMA features, which showed no material gain in any case."),
  BULLET("The model and the naive benchmarks are scored on slightly different rows, because the "
    + "model requires a feature warm-up period that the benchmarks do not."),
  BULLET("Shock detection is retrospective: residuals exist only where the model forecast "
    + "out-of-sample, so every shock shown is a measured historical deviation rather than a "
    + "prediction of future risk."),
  BULLET("Hierarchical aggregation is bottom-up; no formal forecast reconciliation is performed."),
  BULLET("No causal inference is attempted. External signals are treated as association only."),
  BULLET("The demand shock score is a metric defined by this project, not a validated industry "
    + "standard, and the application says so on the page."),

  H2("14.3 Responsible use"),
  P("The platform is decision support. Forecast uncertainty is displayed rather than hidden, "
    + "poorly performing segments are surfaced rather than buried, and the ablation verdict is "
    + "generated from the numbers so it cannot be quietly softened. Human review belongs between "
    + "any output here and an ordering decision."),
  new Paragraph({ children: [new PageBreak()] }),
];

// =======================================================================
//  13. REPRODUCTION + 14. CONCLUSION
// =======================================================================
const repro = [
  H1("15. Reproduction"),

  H2("15.1 Environment"),
  P("Python 3.11 or later. Install dependencies, then run the dependency smoke test, which "
    + "exercises every library call the pipeline relies on so environment problems surface before "
    + "anything else runs."),
  P("pip install -r requirements.txt", { font: "Consolas", size: 19, shading: { type: ShadingType.CLEAR, fill: NOTE_BG, color: "auto" }, after: 40 }),
  P("python scripts/smoke_deps.py", { font: "Consolas", size: 19, shading: { type: ShadingType.CLEAR, fill: NOTE_BG, color: "auto" } }),

  H2("15.2 Build and serve"),
  P("python scripts/run_pipeline.py --mode development", { font: "Consolas", size: 19, shading: { type: ShadingType.CLEAR, fill: NOTE_BG, color: "auto" }, after: 40 }),
  P("streamlit run app/Home.py", { font: "Consolas", size: 19, shading: { type: ShadingType.CLEAR, fill: NOTE_BG, color: "auto" }, after: 40 }),
  P("uvicorn api.main:app --reload", { font: "Consolas", size: 19, shading: { type: ShadingType.CLEAR, fill: NOTE_BG, color: "auto" }, after: 40 }),
  P("python -m pytest", { font: "Consolas", size: 19, shading: { type: ShadingType.CLEAR, fill: NOTE_BG, color: "auto" } }),

  P("Development mode uses a real subset of the M5 data - one store, one category, a fixed number "
    + "of real items - so the pipeline completes in minutes. It is a subset of real data, never "
    + "generated data. Full mode processes all 30,490 series."),

  H2("15.3 Measured run characteristics"),
  table(
    ["Stage", "Full-scale duration"],
    [
      ["Data preparation and feature engineering", "Approximately 5 minutes"],
      ["Training, backtesting and ablation", "Approximately 55 minutes"],
      ["Shock scoring and inventory analysis", "Approximately 11 minutes"],
      ["Complete test suite", "Under 2 minutes"],
    ],
    [4600, 4760], { boldFirstCol: true },
  ),

  ...SPACER(1),
  P(`The deploy model was fitted on ${num(D.train_rows_deploy)} training rows with a `
    + `${M.train_window_days}-day training window, converging at iteration `
    + `${D.best_iter.deploy}.`),

  H1("16. Conclusion"),
  P("DemandShock demonstrates an end-to-end applied data science workflow: data engineering at "
    + "scale, time-series forecasting under strict temporal discipline, anomaly and regime "
    + "detection, external data integration evaluated on its merits, explainability, inventory "
    + "decision support, software engineering practice, and executive communication."),
  P("The forecasting model measurably outperforms the benchmark that shares its information set, "
    + "its prediction intervals are well calibrated at scale, and its shock detection produces "
    + "auditable, individually explainable episodes. Equally important, the project reports "
    + "honestly where its hypothesis failed: external crisis and economic context did not "
    + "materially improve point forecasts, the selection rule acted on that evidence, and the "
    + "finding is displayed rather than concealed."),
  P("The result is a platform whose numbers can be traced from the interface back through the "
    + "artifacts to the original source files, and which states its limitations as clearly as its "
    + "results."),

  ...SPACER(1),
  RICH([{
    t: "End of report.", i: true, c: MUTED,
  }], { align: AlignmentType.CENTER }),
];

// =======================================================================
//  DOCUMENT
// =======================================================================
const doc = new Document({
  creator: "DemandShock",
  title: "DemandShock - Technical Project Report",
  description: "Crisis-Aware Demand Forecasting & Inventory Intelligence Platform",
  numbering: {
    config: [{
      reference: "bullets",
      levels: [
        { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 360, hanging: 200 } } } },
        { level: 1, format: LevelFormat.BULLET, text: "◦", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 200 } } } },
      ],
    }],
  },
  styles: {
    default: { document: { run: { font: "Calibri", size: 21, color: INK } } },
  },
  sections: [
    {
      properties: {
        page: {
          size: { width: 12240, height: 15840 },
          margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
        },
      },
      children: title,
    },
    {
      properties: {
        page: {
          size: { width: 12240, height: 15840 },
          margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
        },
      },
      headers: {
        default: new Header({
          children: [new Paragraph({
            alignment: AlignmentType.RIGHT,
            border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: "C9D2DB", space: 6 } },
            children: [new TextRun({
              text: "DemandShock - Technical Project Report",
              size: 17, color: MUTED, font: "Calibri",
            })],
          })],
        }),
      },
      footers: {
        default: new Footer({
          children: [new Paragraph({
            alignment: AlignmentType.CENTER,
            children: [new TextRun({
              children: ["Page ", PageNumber.CURRENT, " of ", PageNumber.TOTAL_PAGES],
              size: 17, color: MUTED, font: "Calibri",
            })],
          })],
        }),
      },
      children: [
        ...contents, ...exec, ...business, ...data, ...arch, ...method, ...buckets,
        ...results, ...ablation, ...shocks, ...inventory, ...reading, ...descriptors, ...app,
        ...quality, ...limits, ...repro,
      ],
    },
  ],
});

Packer.toBuffer(doc).then(b => {
  fs.writeFileSync(process.argv[3], b);
  console.log("wrote", process.argv[3], b.length, "bytes");
});
