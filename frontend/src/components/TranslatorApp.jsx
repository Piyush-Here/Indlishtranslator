import { useState, useRef } from "react";
import styles from "../styles/TranslatorApp.module.css";

const API_URL = "http://localhost:5000/translate";

const LANG_CONFIG = {
  en_hi: {
    from: "English",
    to: "हिन्दी",
    fromFlag: "🇬🇧",
    toFlag: "🇮🇳",
    placeholder: "Type an English sentence...",
    outputLabel: "हिन्दी output",
  },
  hi_en: {
    from: "हिन्दी",
    to: "English",
    fromFlag: "🇮🇳",
    toFlag: "🇬🇧",
    placeholder: "हिन्दी वाक्य लिखें...",
    outputLabel: "English output",
  },
};

const MODE_CONFIG = {
  baseline: { emoji: "⚡", label: "Baseline",     desc: "Basic Seq2Seq, greedy" },
  attention: { emoji: "🔍", label: "Attention",   desc: "Bahdanau attention, greedy" },
  beam:      { emoji: "🔦", label: "Beam Search", desc: "Attention + beam" },
};

const BEAM_OPTIONS = [3, 5, 8, 10];

export default function TranslatorApp() {
  const [direction, setDirection] = useState("en_hi");
  const [input, setInput]         = useState("");
  const [output, setOutput]       = useState("");
  const [mode, setMode]           = useState("beam");
  const [beamK, setBeamK]         = useState(5);
  const [loading, setLoading]     = useState(false);
  const [error, setError]         = useState("");
  const [flipping, setFlipping]   = useState(false);
  const textareaRef = useRef(null);

  const L = LANG_CONFIG[direction];

  const handleFlip = () => {
    setFlipping(true);
    setTimeout(() => {
      setDirection(d => (d === "en_hi" ? "hi_en" : "en_hi"));
      setInput(output);
      setOutput(input);
      setError("");
      setFlipping(false);
    }, 300);
  };

  const handleTranslate = async () => {
    if (!input.trim()) return;
    setLoading(true);
    setError("");
    setOutput("");

    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: input.trim(), direction, mode, beam_k: beamK }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.error || `Server error ${res.status}`);
      }
      const data = await res.json();
      setOutput(data.translation || "");
    } catch (err) {
      setError(
        err.message.includes("fetch")
          ? "⚠️ Cannot reach backend — make sure app.py is running on port 5000."
          : err.message
      );
    } finally {
      setLoading(false);
    }
  };

  const handleKey = e => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) handleTranslate();
  };

  const modeDesc =
    mode === "beam"
      ? `Attention + beam k=${beamK}`
      : MODE_CONFIG[mode].desc;

  return (
    <div className={styles.page}>
      {/* Ambient blobs */}
      <div className={`${styles.blob} ${styles.blobTopLeft}`} />
      <div className={`${styles.blob} ${styles.blobBottomRight}`} />

      {/* Header */}
      <header className={styles.header}>
        <p className={styles.subtitle}>Seq2Seq Neural Translator</p>
        <div className={`${styles.directionRow} ${flipping ? styles.flipping : ""}`}>
          <span className={styles.langLabel}>{L.fromFlag} {L.from}</span>
          <span className={styles.arrow}>→</span>
          <span className={styles.langLabel}>{L.to} {L.toFlag}</span>
        </div>
        <p className={styles.poweredBy}>Powered by your trained LSTM model</p>
      </header>

      {/* Main card */}
      <main className={styles.card}>

        {/* Mode tabs */}
        <div className={styles.modeTabs}>
          {Object.entries(MODE_CONFIG).map(([key, cfg]) => (
            <button
              key={key}
              onClick={() => setMode(key)}
              className={`${styles.modeTab} ${mode === key ? styles.modeTabActive : ""}`}
            >
              <span className={styles.modeEmoji}>{cfg.emoji}</span>
              {cfg.label}
            </button>
          ))}
        </div>

        {/* Mode description + beam controls */}
        <div className={styles.modeBar}>
          <span className={styles.modeDesc}>{modeDesc}</span>
          {mode === "beam" && (
            <span className={styles.beamControls}>
              k:
              {BEAM_OPTIONS.map(k => (
                <button
                  key={k}
                  onClick={() => setBeamK(k)}
                  className={`${styles.beamBtn} ${beamK === k ? styles.beamBtnActive : ""}`}
                >
                  {k}
                </button>
              ))}
            </span>
          )}
        </div>

        {/* Translation panel */}
        <div className={styles.panel}>

          {/* Input side */}
          <div className={styles.panelSide}>
            <p className={styles.panelLabel}>{L.fromFlag} {L.from}</p>
            <textarea
              ref={textareaRef}
              value={input}
              onChange={e => { setInput(e.target.value); setError(""); }}
              onKeyDown={handleKey}
              placeholder={L.placeholder}
              rows={5}
              className={styles.textarea}
            />
            <p className={styles.hint}>Ctrl+Enter to translate</p>
          </div>

          {/* Centre controls */}
          <div className={styles.panelControls}>
            <button
              onClick={handleFlip}
              title="Swap languages"
              className={`${styles.iconBtn} ${flipping ? styles.iconBtnFlipping : ""}`}
            >
              ⇄
            </button>
            <button
              onClick={handleTranslate}
              disabled={loading || !input.trim()}
              title="Translate"
              className={`${styles.iconBtn} ${styles.iconBtnTranslate} ${loading || !input.trim() ? styles.iconBtnDisabled : ""}`}
            >
              {loading
                ? <span className={styles.spinner} />
                : "→"}
            </button>
          </div>

          {/* Output side */}
          <div className={styles.panelSide}>
            <p className={`${styles.panelLabel} ${styles.panelLabelOutput}`}>
              {L.toFlag} {L.outputLabel}
            </p>
            {error ? (
              <p className={styles.errorText}>{error}</p>
            ) : output ? (
              <p className={styles.outputText}>{output}</p>
            ) : (
              <p className={styles.placeholder}>
                {loading ? "Decoding..." : "Translation appears here"}
              </p>
            )}
            {output && !loading && (
              <button
                onClick={() => navigator.clipboard?.writeText(output)}
                className={styles.copyBtn}
              >
                copy
              </button>
            )}
          </div>
        </div>

        {/* Full-width translate button */}
        <div className={styles.cardFooter}>
          <button
            onClick={handleTranslate}
            disabled={loading || !input.trim()}
            className={`${styles.translateBtn} ${loading || !input.trim() ? styles.translateBtnDisabled : ""}`}
          >
            {loading ? "Translating..." : `Translate ${L.fromFlag} → ${L.toFlag}`}
          </button>
        </div>

        {/* Card footer meta */}
        <div className={styles.cardMeta}>
          <span>Backend: <code className={styles.code}>localhost:5000</code></span>
          <span>EN ↔ HI · Seq2Seq + Attention + Beam Search</span>
        </div>
      </main>

      {/* Setup box */}
      <div className={styles.setupBox}>
        <p className={styles.setupTitle}>🛠 Quick setup</p>
        <pre className={styles.setupPre}>{`pip install flask flask-cors tensorflow numpy\npython app.py     # starts on port 5000`}</pre>
        <p className={styles.setupNote}>
          Place <code className={styles.codeAccent}>app.py</code> alongside all{" "}
          <code className={styles.codeAccent}>.keras</code> and{" "}
          <code className={styles.codeAccent}>.pkl</code> files.
          Run Section 15 of the notebook to enable HI→EN direction.
        </p>
      </div>
    </div>
  );
}
