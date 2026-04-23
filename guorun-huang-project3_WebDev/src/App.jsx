import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { SudokuProvider } from './sudoku/SudokuContext.jsx';
import { AuthProvider }   from './sudoku/AuthContext.jsx';
import Navbar    from './components/Navbar.jsx';
import Home      from './pages/Home.jsx';
import Select    from './pages/Select.jsx';
import Game      from './pages/Game.jsx';
import Custom    from './pages/Custom.jsx';
import Rules     from './pages/Rules.jsx';
import Scores    from './pages/Scores.jsx';
import Login     from './pages/Login.jsx';
import Register  from './pages/Register.jsx';

function Footer() {
  return (
    <footer className="site-footer">
      <div className="footer-brand">
      Go<span style={{ color: 'var(--accent)' }}>Sudoku</span>
      </div>
      <div className="footer-copy">© 2026 GoSudoku · All rights reserved</div>
    </footer>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <SudokuProvider>
          <Navbar />
          <Routes>
            <Route path="/"          element={<Home />} />
            <Route path="/games"     element={<Select />} />
            <Route path="/game/:gameid" element={<Game />} />
            <Route path="/custom"    element={<Custom />} />
            <Route path="/rules"     element={<Rules />} />
            <Route path="/scores"    element={<Scores />} />
            <Route path="/login"     element={<Login />} />
            <Route path="/register"  element={<Register />} />
          </Routes>
          <Footer />
        </SudokuProvider>
      </AuthProvider>
    </BrowserRouter>
  );
}
