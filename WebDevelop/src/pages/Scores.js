import React from 'react';
import './Scores.css';

const SCORES = [
  { rank:1,  username:'pixel_logic',      solved:3142, avgTime:'02:58', streak:63 },
  { rank:2,  username:'nightowl_sudo',    solved:2891, avgTime:'03:11', streak:49 },
  { rank:3,  username:'caffeine_grids',   solved:2634, avgTime:'03:27', streak:38 },
  { rank:4,  username:'methodical_m',     solved:2201, avgTime:'03:54', streak:30 },
  { rank:5,  username:'zeroerr0r',        solved:1876, avgTime:'04:16', streak:25 },
  { rank:6,  username:'quietstorm99',     solved:1543, avgTime:'04:45', streak:21 },
  { rank:7,  username:'debugmode_on',     solved:1287, avgTime:'05:08', streak:17 },
  { rank:8,  username:'block_by_block',   solved:1043, avgTime:'05:39', streak:13 },
  { rank:9,  username:'singular_focus',   solved: 892, avgTime:'06:02', streak: 9 },
  { rank:10, username:'tenacity_five',    solved: 711, avgTime:'06:44', streak: 6 },
];

const medals = ['🥇','🥈','🥉'];

export default function Scores() {
  return (
    <div className="scores-page">
      <div className="page-header">
        <h1 className="page-title">High <span className="accent">Scores</span></h1>
        <p className="page-sub">Top puzzle solvers on the Go Sudoku leaderboard.</p>
      </div>

      <div className="scores-container">
        <div className="scores-legend-row">
          <span>Player</span>
          <span>Solved</span>
          <span>Avg Time</span>
          <span>Streak</span>
        </div>
        {SCORES.map(e => (
          <div className={`score-row ${e.rank <= 3 ? 'top-three' : ''}`} key={e.rank}>
            <div className="rank-cell">
              {e.rank <= 3
                ? <span className="medal">{medals[e.rank-1]}</span>
                : <span className="rank-num">#{e.rank}</span>
              }
              <span className="username">{e.username}</span>
            </div>
            <div className="stat-cell">{e.solved.toLocaleString()}</div>
            <div className="stat-cell mono">{e.avgTime}</div>
            <div className="stat-cell streak">🔥 {e.streak}</div>
          </div>
        ))}
      </div>
      <p className="scores-note">* Mock data. Login to track your own progress.</p>
    </div>
  );
}
