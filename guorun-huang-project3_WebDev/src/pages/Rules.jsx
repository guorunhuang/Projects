import React, { useRef, useEffect, useState } from 'react';

const RULES = [
  {
    num: '01',
    title: 'Fill Every Cell',
    desc: 'Every cell in the grid must contain exactly one number. No empty spaces when you\'re done.',
    tag: 'MANDATORY',
  },
  {
    num: '02',
    title: 'Unique Rows',
    desc: 'Each row must contain each number exactly once — 1 through 9 in normal mode, 1 through 6 in easy mode.',
    tag: 'ROWS',
  },
  {
    num: '03',
    title: 'Unique Columns',
    desc: 'Every column must also contain each number exactly once. No duplicates top to bottom.',
    tag: 'COLS',
  },
  {
    num: '04',
    title: 'Unique Boxes',
    desc: 'Each bold-bordered sub-grid (3×3 in normal, 2×3 in easy) must hold each number exactly once.',
    tag: 'BOXES',
  },
  {
    num: '05',
    title: 'Clues Are Fixed',
    desc: 'Pre-filled numbers are permanent — they are the anchors your logic must work around.',
    tag: 'GIVEN',
  },
  {
    num: '06',
    title: 'One Solution Only',
    desc: 'Every puzzle is backtracking-verified to have exactly one valid solution. No guesswork needed.',
    tag: 'UNIQUE',
  },
];

const CONTROLS = [
  { keys: ['1','–','9'], action: 'Enter a digit into selected cell' },
  { keys: ['⌫'], action: 'Erase the selected cell' },
  { keys: ['↑','↓','←','→'], action: 'Move between cells' },
  { keys: ['Click'], action: 'Select any cell' },
  { keys: ['Hint'], action: 'Highlight a cell with only one valid number' },
  { keys: ['Reset'], action: 'Restore puzzle to its original state' },
];

function RuleRow({ num, title, desc, tag, index }) {
  const ref = useRef(null);
  const [vis, setVis] = useState(false);
  useEffect(() => {
    const el = ref.current;
    const obs = new IntersectionObserver(([e]) => { if (e.isIntersecting) setVis(true); }, { threshold: 0.15 });
    if (el) obs.observe(el);
    return () => obs.disconnect();
  }, []);

  return (
    <div
      ref={ref}
      className={`rule-row-new ${vis ? 'rule-visible' : ''}`}
      style={{ transitionDelay: `${index * 0.07}s` }}
    >
      <div className="rule-num-col">
        <span className="rule-big-num">{num}</span>
        {index < RULES.length - 1 && <div className="rule-spine-line" />}
      </div>
      <div className="rule-content-col">
        <div className="rule-tag">{tag}</div>
        <h3 className="rule-title-new">{title}</h3>
        <p className="rule-desc-new">{desc}</p>
      </div>
    </div>
  );
}

export default function Rules() {
  return (
    <div className="rules-page-new">

      {/* ── Hero banner ── */}
      <div className="rules-hero">
        <div className="rules-hero-bg">
          {Array.from({length: 9}, (_,i) => (
            <div key={i} className="rules-hero-col" style={{left:`${(i/9)*100}%`}} />
          ))}
          {Array.from({length: 9}, (_,i) => (
            <div key={i} className="rules-hero-row" style={{top:`${(i/9)*100}%`}} />
          ))}
        </div>
        <div className="rules-hero-content">
          <p className="rules-hero-eyebrow">THE SYSTEM</p>
          <h1 className="rules-hero-title">HOW TO<br/><em>PLAY</em></h1>
          <p className="rules-hero-sub">Six constraints. One solution. Infinite puzzles.</p>
        </div>
        <div className="rules-hero-grid-preview">
          {[1,0,0,4,0,0,7,0,0,
            0,2,0,0,5,0,0,8,0,
            0,0,3,0,0,6,0,0,9].map((v,i) => (
            <div key={i} className={`rh-cell ${v ? 'rh-filled' : ''}`}>{v || ''}</div>
          ))}
        </div>
      </div>

      {/* ── Rules timeline ── */}
      <div className="rules-body">
        <div className="rules-timeline">
          <div className="timeline-label">RULES</div>
          {RULES.map((r, i) => (
            <RuleRow key={r.num} {...r} index={i} />
          ))}
        </div>

        {/* ── Right column: modes + controls + credits ── */}
        <div className="rules-aside">

          <div className="aside-block">
            <div className="aside-label">DIFFICULTY</div>
            <div className="mode-band easy-band">
              <div className="mode-band-left">
                <div className="mode-band-name">EASY</div>
                <div className="mode-band-grid">6 × 6</div>
              </div>
              <div className="mode-band-right">
                <div className="mode-band-clues">18 clues</div>
                <div className="mode-band-desc">Sub-grids are 2 × 3 boxes</div>
              </div>
            </div>
            <div className="mode-band normal-band">
              <div className="mode-band-left">
                <div className="mode-band-name">NORMAL</div>
                <div className="mode-band-grid">9 × 9</div>
              </div>
              <div className="mode-band-right">
                <div className="mode-band-clues">29 clues</div>
                <div className="mode-band-desc">Classic 3 × 3 sub-grids</div>
              </div>
            </div>
          </div>

          <div className="aside-block">
            <div className="aside-label">CONTROLS</div>
            <div className="controls-table">
              {CONTROLS.map(({ keys, action }) => (
                <div className="ctrl-row" key={action}>
                  <div className="ctrl-keys">
                    {keys.map(k => <kbd key={k} className="ctrl-key">{k}</kbd>)}
                  </div>
                  <div className="ctrl-action">{action}</div>
                </div>
              ))}
            </div>
          </div>

          <div className="aside-block credits-block">
            <div className="aside-label">CREDITS</div>
            <p className="credits-name">Guorun Huang</p>
            <p className="credits-role">Developer & Designer</p>
            <div className="credits-links">
              <a href="huang.guor@northeastern.edu" className="credits-link-new">EMAIL</a>
              <a href="https://github.com" target="_blank" rel="noreferrer" className="credits-link-new">GITHUB</a>
              <a href="https://linkedin.com" target="_blank" rel="noreferrer" className="credits-link-new">LINKEDIN</a>
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}
