import React from 'react';
import { Link } from 'react-router-dom';
import './Home.css';

export default function Home() {
  return (
    <div className="home-page">
      <div className="home-bg">
        <div className="grid-pattern" />
        <div className="glow-orb orb1" />
        <div className="glow-orb orb2" />
      </div>
      <div className="home-content">
        <div className="home-badge">Number Puzzle Game</div>
        <h1 className="home-title">
          Go<span className="accent">Sudoku</span>
        </h1>
        <p className="home-sub">
          Go solve the sudoku puzzle.
        </p>
        <div className="home-demo-grid">
          {[1,3,0,0,7,0,0,0,0,
            6,0,0,1,9,5,0,0,0,
            0,9,8,0,0,0,0,6,0].map((v, i) => (
            <div key={i} className={`demo-cell ${v ? 'filled' : ''}`}>
              {v || ''}
            </div>
          ))}
        </div>
        <div className="home-actions">
          <Link to="/games/easy" className="btn-primary">Play Easy</Link>
          <Link to="/games/normal" className="btn-secondary">Play Normal</Link>
        </div>
        <div className="home-stats">
          <div className="stat"><div className="stat-num">2</div><div className="stat-label">Modes</div></div>
          <div className="stat-divider" />
          <div className="stat"><div className="stat-num">∞</div><div className="stat-label">Puzzles</div></div>
          <div className="stat-divider" />
          <div className="stat"><div className="stat-num">1</div><div className="stat-label">Solution</div></div>
        </div>
      </div>
    </div>
  );
}
