import React from 'react';
import { Link } from 'react-router-dom';
import './Auth.css';

export default function Login() {
  return (
    <div className="auth-page">
      <div className="auth-card">
        <div className="auth-brand">Numer<span className="accent">iX</span></div>
        <h1 className="auth-title">Welcome back</h1>
        <p className="auth-sub">Sign in to track your progress and compete on the leaderboard.</p>

        <div className="auth-form">
          <div className="field-group">
            <label className="field-label" htmlFor="username">Username</label>
            <input
              id="username"
              type="text"
              className="field-input"
              placeholder="your_username"
              autoComplete="username"
            />
          </div>

          <div className="field-group">
            <label className="field-label" htmlFor="password">Password</label>
            <input
              id="password"
              type="password"
              className="field-input"
              placeholder="••••••••"
              autoComplete="current-password"
            />
          </div>

          <button className="auth-submit" type="button">Sign In</button>
        </div>

        <p className="auth-switch">
          Don't have an account? <Link to="/register" className="auth-link">Register →</Link>
        </p>
      </div>
    </div>
  );
}
