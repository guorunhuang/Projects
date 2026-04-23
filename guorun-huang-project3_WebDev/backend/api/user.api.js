import express from 'express';
import bcrypt  from 'bcrypt';
import User    from './db/schema/user.schema.js';

const router = express.Router();
const SALT_ROUNDS = 10;
const COOKIE_NAME = 'GoSudoku_user';

// ── POST /api/users/register ───────────────────────────────────────────────
router.post('/register', async (req, res) => {
  try {
    const { username, password } = req.body;
    if (!username || !password)
      return res.status(400).json({ error: 'Username and password are required.' });

    const existing = await User.findOne({ username });
    if (existing)
      return res.status(409).json({ error: 'Username already exists.' });

    const hashed = await bcrypt.hash(password, SALT_ROUNDS);
    const user   = await new User({ username, password: hashed }).save();

    res.cookie(COOKIE_NAME, username, {
      httpOnly: true,
      sameSite: 'lax',
      maxAge: 7 * 24 * 60 * 60 * 1000, // 7 days
    });

    return res.status(201).json({ username: user.username });
  } catch (err) {
    console.error(err);
    return res.status(500).json({ error: 'Server error.' });
  }
});

// ── POST /api/users/login ──────────────────────────────────────────────────
router.post('/login', async (req, res) => {
  try {
    const { username, password } = req.body;
    if (!username || !password)
      return res.status(400).json({ error: 'Username and password are required.' });

    const user = await User.findOne({ username });
    if (!user)
      return res.status(401).json({ error: 'Invalid username or password.' });

    const match = await bcrypt.compare(password, user.password);
    if (!match)
      return res.status(401).json({ error: 'Invalid username or password.' });

    res.cookie(COOKIE_NAME, username, {
      httpOnly: true,
      sameSite: 'lax',
      maxAge: 7 * 24 * 60 * 60 * 1000,
    });

    return res.json({ username: user.username });
  } catch (err) {
    console.error(err);
    return res.status(500).json({ error: 'Server error.' });
  }
});

// ── POST /api/users/logout ────────────────────────────────────────────────
router.post('/logout', (req, res) => {
  res.clearCookie(COOKIE_NAME);
  return res.json({ message: 'Logged out.' });
});

// ── GET /api/users/me ─────────────────────────────────────────────────────
// Returns currently logged-in user from cookie (used by frontend on load)
router.get('/me', async (req, res) => {
  const username = req.cookies[COOKIE_NAME];
  if (!username) return res.status(401).json({ error: 'Not logged in.' });

  const user = await User.findOne({ username }, 'username wins');
  if (!user) {
    res.clearCookie(COOKIE_NAME);
    return res.status(401).json({ error: 'User not found.' });
  }

  return res.json({ username: user.username, wins: user.wins });
});

export default router;
