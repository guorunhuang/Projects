import React, { useEffect, useState } from 'react';

const medals = ['🥇', '🥈', '🥉'];

export default function Scores() {
  const [scores,  setScores]  = useState([]);
  const [loading, setLoading] = useState(true);
  const [error,   setError]   = useState('');

  useEffect(() => {
    fetch('/api/highscore')
      .then(r => r.json())
      .then(data => setScores(data))
      .catch(() => setError('Failed to load scores.'))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="scores-page">
      <div className="page-header">
        <h1 className="page-title">High <span className="accent">Scores</span></h1>
        <p className="page-sub">Top puzzle solvers on the GoSudoku leaderboard.</p>
      </div>

      <div className="scores-container">
        {loading ? (
          <div className="loading-state">Loading scores…</div>
        ) : error ? (
          <div className="error-state">{error}</div>
        ) : scores.length === 0 ? (
          <div className="empty-state">No scores yet — complete a game to appear here!</div>
        ) : (
          <>
            <div className="scores-legend-row">
              <span>Player</span>
              <span>Wins</span>
            </div>
            {scores.map((entry, i) => (
              <div className={`score-row ${i < 3 ? 'top-three' : ''}`} key={entry.username}>
                <div className="rank-cell">
                  {i < 3
                    ? <span className="medal">{medals[i]}</span>
                    : <span className="rank-num">#{i + 1}</span>
                  }
                  <span className="username">{entry.username}</span>
                </div>
                <div className="stat-cell wins-cell">🏆 {entry.wins}</div>
              </div>
            ))}
          </>
        )}
      </div>
      <p className="scores-note">* Only users with at least one win are shown.</p>
    </div>
  );
}
