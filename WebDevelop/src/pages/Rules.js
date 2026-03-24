import React from 'react';
import './Rules.css';

export default function Rules() {
  return (
    <div className="rules-page">
      <div className="page-header">
        <h1 className="page-title">How to <span className="accent">Play</span></h1>
        <p className="page-sub">Everything you need to know about Go Sudoku.</p>
      </div>

      <div className="rules-grid">
        <section className="rules-card main-rules">
          <h2 className="section-title">The Rules</h2>
          <div className="rule-list">
            {[
              { icon: '🔢', title: 'Fill the Grid', desc: 'Each cell must contain exactly one number. No cell may be left empty.' },
              { icon: '↔️', title: 'Unique Rows', desc: 'Every row must contain each number (1–9 in normal mode, 1–6 in easy) exactly once.' },
              { icon: '↕️', title: 'Unique Columns', desc: 'Every column must also contain each number exactly once — no duplicates.' },
              { icon: '⬛', title: 'Unique Boxes', desc: 'Each bold-bordered sub-grid (3×3 or 2×3) must contain each number exactly once.' },
              { icon: '🔒', title: 'Given Clues', desc: 'Gray cells with pre-filled numbers are fixed — you cannot change them.' },
              { icon: '①', title: 'One Solution', desc: 'Every generated puzzle has exactly one valid solution. No guessing required.' },
            ].map(({ icon, title, desc }) => (
              <div className="rule-item" key={title}>
                <div className="rule-icon">{icon}</div>
                <div>
                  <div className="rule-title">{title}</div>
                  <div className="rule-desc">{desc}</div>
                </div>
              </div>
            ))}
          </div>
        </section>

        <div className="rules-sidebar">
          <section className="rules-card">
            <h2 className="section-title">Modes</h2>
            <div className="mode-card easy-mode">
              <div className="mode-name">Easy</div>
              <div className="mode-grid-label">6 × 6 Grid</div>
              <div className="mode-detail">18 clues given (half the board). Sub-grids are 2×3.</div>
            </div>
            <div className="mode-card normal-mode">
              <div className="mode-name">Normal</div>
              <div className="mode-grid-label">9 × 9 Grid</div>
              <div className="mode-detail">29 clues given. Sub-grids are 3×3. Classic Sudoku experience.</div>
            </div>
          </section>

          <section className="rules-card">
            <h2 className="section-title">Controls</h2>
            <div className="control-list">
              {[
                { key: '1 – 9', action: 'Input a number' },
                { key: 'Backspace', action: 'Erase a cell' },
                { key: '↑↓←→', action: 'Navigate cells' },
                { key: 'Click', action: 'Select a cell' },
                { key: 'Hint', action: 'Reveal a forced cell' },
                { key: 'Reset', action: 'Restore original puzzle' },
                { key: 'New Game', action: 'Generate fresh puzzle' },
              ].map(({ key, action }) => (
                <div className="control-row" key={key}>
                  <kbd className="kbd">{key}</kbd>
                  <span className="control-action">{action}</span>
                </div>
              ))}
            </div>
          </section>

          <section className="rules-card credits-card">
            <h2 className="section-title">Credits</h2>
            <p className="credits-text">Made with ❤️ by <strong>Guorun Huang</strong></p>
            <div className="credit-links">
              <a href="mailto:guorun@example.com" className="credit-link">📧 Email</a>
              <a href="https://github.com/guorunhuang/Projects" target="_blank" rel="noreferrer" className="credit-link">🐙 GitHub</a>
              <a href="https://www.linkedin.com/in/guorun-huang-159663369/" target="_blank" rel="noreferrer" className="credit-link">💼 LinkedIn</a>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}
