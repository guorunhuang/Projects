import React, { useEffect, useRef } from 'react';
import { Link } from 'react-router-dom';

// Animated ticker numbers that cycle through digits
function TickerCell({ value, delay = 0 }) {
  return (
    <div className="ticker-cell" style={{ animationDelay: `${delay}s` }}>
      <span className="ticker-value">{value || ''}</span>
    </div>
  );
}

const PUZZLE = [
  [5,3,0,6,0,8,0,0,0],
  [0,7,0,1,9,5,0,0,0],
  [0,0,8,0,0,0,0,6,0],
  [8,0,0,0,6,0,0,0,3],
  [4,0,0,8,0,3,0,0,1],
  [7,0,0,0,2,0,0,0,6],
  [0,6,0,0,0,0,2,8,0],
  [0,0,0,4,1,9,0,0,5],
  [0,0,0,0,8,0,0,7,9],
];

export default function Home() {
  const scanRef = useRef(null);

  // Animate scan line
  useEffect(() => {
    const el = scanRef.current;
    if (!el) return;
    let pos = 0;
    const tick = () => {
      pos = (pos + 0.3) % 100;
      el.style.top = `${pos}%`;
    };
    const id = setInterval(tick, 16);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="home-page">
      {/* ── Left panel: giant live grid ── */}
      <div className="home-left">
        <div className="scan-container">
          <div className="scan-line" ref={scanRef} />
          <div className="big-grid">
            {PUZZLE.flat().map((v, i) => (
              <TickerCell key={i} value={v} delay={(i % 9) * 0.07 + Math.floor(i / 9) * 0.04} />
            ))}
          </div>
          {/* Corner labels */}
          <div className="grid-corner corner-tl">ROW</div>
          <div className="grid-corner corner-tr">COL</div>
          <div className="grid-corner corner-bl">BOX</div>
          <div className="grid-corner corner-br">NUM</div>
        </div>

        {/* Vertical marquee strip */}
        <div className="marquee-strip">
          <div className="marquee-inner">
            {Array(4).fill('GO · SUDOKU · THINK · SOLVE · WIN · ').join('')}
          </div>
        </div>
      </div>

      {/* ── Divider ── */}
      <div className="home-divider">
        <div className="divider-line" />
        <div className="divider-dot" />
        <div className="divider-line" />
      </div>

      {/* ── Right panel: editorial content ── */}
      <div className="home-right">
        <div className="home-eyebrow">
          <span className="eyebrow-line" />
          <span className="eyebrow-text">NUMBER PUZZLE SYSTEM</span>
        </div>

        <h1 className="home-headline">
          <span className="hl-numer">Go</span>
          <span className="hl-ix">Sudoku</span>
        </h1>

        <p className="home-descriptor">
          This is a web that rapidly produces<br />
          unique puzzles for you.
        </p>

        <div className="home-pill-row">
          <span className="home-pill">9 × 9 GRID</span>
          <span className="home-pill">6 × 6 GRID</span>
          <span className="home-pill">CUSTOM PUZZLES</span>
        </div>

        <div className="home-cta-stack">
          <Link to="/games" className="cta-primary">
            <span>PLAY NOW</span>
            <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
              <path d="M4 10h12M11 5l5 5-5 5" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
            </svg>
          </Link>
          <Link to="/rules" className="cta-ghost">HOW IT WORKS</Link>
        </div>

        <div className="home-data-row">
          <div className="data-block">
            <div className="data-num">∞</div>
            <div className="data-label">Unique Puzzles</div>
          </div>
          <div className="data-sep" />
          <div className="data-block">
            <div className="data-num">01</div>
            <div className="data-label">Valid Solution</div>
          </div>
          <div className="data-sep" />
          <div className="data-block">
            <div className="data-num">02</div>
            <div className="data-label">Difficulty Modes</div>
          </div>
        </div>

        <div className="home-footnote">
          © 2026 GoSudoku — Backtracking-verified puzzles
        </div>
      </div>
    </div>
  );
}
