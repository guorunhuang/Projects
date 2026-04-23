import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../sudoku/AuthContext.jsx';

export default function Register() {
  const navigate = useNavigate();
  const { register } = useAuth();
  const [username, setUsername]   = useState('');
  const [password, setPassword]   = useState('');
  const [confirm,  setConfirm]    = useState('');
  const [error,    setError]      = useState('');
  const [loading,  setLoading]    = useState(false);

  const disabled = !username.trim() || !password.trim() || !confirm.trim() || loading;

  async function handleSubmit() {
    if (disabled) return;
    if (password !== confirm) {
      setError('Passwords do not match.');
      return;
    }
    setLoading(true);
    setError('');
    try {
      await register(username.trim(), password);
      navigate('/games');
    } catch (err) {
      setError(err.message || 'Registration failed.');
    } finally {
      setLoading(false);
    }
  }

  const handleKey = (e) => { if (e.key === 'Enter') handleSubmit(); };

  return (
    <div className="auth-page">
      <div className="auth-card">
        <div className="auth-brand">Go<span className="accent">Sudoku</span></div>
        <h1 className="auth-title">Create account</h1>
        <p className="auth-sub">Join GoSudoku and start solving puzzles today.</p>

        {error && <div className="form-error">{error}</div>}

        <div className="auth-form">
          <div className="field-group">
            <label className="field-label" htmlFor="reg-username">Username</label>
            <input
              id="reg-username"
              type="text"
              className="field-input"
              placeholder="choose_a_username"
              autoComplete="username"
              value={username}
              onChange={e => setUsername(e.target.value)}
              onKeyDown={handleKey}
            />
          </div>

          <div className="field-group">
            <label className="field-label" htmlFor="reg-password">Password</label>
            <input
              id="reg-password"
              type="password"
              className="field-input"
              placeholder="••••••••"
              autoComplete="new-password"
              value={password}
              onChange={e => setPassword(e.target.value)}
              onKeyDown={handleKey}
            />
          </div>

          <div className="field-group">
            <label className="field-label" htmlFor="reg-confirm">Confirm Password</label>
            <input
              id="reg-confirm"
              type="password"
              className="field-input"
              placeholder="••••••••"
              autoComplete="new-password"
              value={confirm}
              onChange={e => setConfirm(e.target.value)}
              onKeyDown={handleKey}
            />
          </div>

          <button
            className="auth-submit"
            onClick={handleSubmit}
            disabled={disabled}
          >
            {loading ? 'Creating…' : 'Create Account'}
          </button>
        </div>

        <p className="auth-switch">
          Already have an account? <Link to="/login" className="auth-link">Sign in →</Link>
        </p>
      </div>
    </div>
  );
}
