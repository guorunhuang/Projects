import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import './Navbar.css';

export default function Navbar() {
  const location = useLocation();
  const [open, setOpen] = useState(false);

  const links = [
    { to: '/',       label: 'Home'   },
    { to: '/games',  label: 'Games'  },
    { to: '/rules',  label: 'Rules'  },
    { to: '/scores', label: 'Scores' },
    { to: '/login',  label: 'Login'  },
  ];

  return (
    <nav className="navbar">
      <Link to="/" className="navbar-brand">
        Go<span className="brand-accent">Sudoku</span>
      </Link>

      <button
        className="navbar-burger"
        onClick={() => setOpen(o => !o)}
        aria-label="Toggle menu"
      >
        <span className={`burger-bar ${open ? 'open' : ''}`} />
        <span className={`burger-bar ${open ? 'open' : ''}`} />
        <span className={`burger-bar ${open ? 'open' : ''}`} />
      </button>

      <ul className={`navbar-links ${open ? 'open' : ''}`}>
        {links.map(({ to, label }) => (
          <li key={to}>
            <Link
              to={to}
              className={`nav-link ${location.pathname === to ? 'active' : ''}`}
              onClick={() => setOpen(false)}
            >
              {label}
            </Link>
          </li>
        ))}
        <li>
          <Link to="/register" className="nav-btn" onClick={() => setOpen(false)}>
            Register
          </Link>
        </li>
      </ul>
    </nav>
  );
}
