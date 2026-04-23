import Sudoku from '../schema/sudoku.schema.js';
import User   from '../schema/user.schema.js';

// ── Create a new game ──────────────────────────────────────────────────────
export async function createGame({ name, difficulty, puzzle, solution, createdBy }) {
  const game = new Sudoku({ name, difficulty, puzzle, solution, createdBy });
  return game.save();
}

// ── Get all games (list view) ──────────────────────────────────────────────
export async function getAllGames() {
  return Sudoku.find({}, 'name difficulty createdBy createdAt').sort({ createdAt: -1 });
}

// ── Get a single game by ID ────────────────────────────────────────────────
export async function getGameById(id) {
  return Sudoku.findById(id);
}

// ── Save user progress on a game ──────────────────────────────────────────
export async function saveProgress(gameId, username, board, isComplete) {
  const game = await Sudoku.findById(gameId);
  if (!game) throw new Error('Game not found');

  game.userProgress.set(username, { board, isComplete });
  await game.save();

  return game;
}

// ── Delete a game & roll back every completer's win count ─────────────────
export async function deleteGame(id) {
  const game = await Sudoku.findById(id);
  if (!game) return null;

  // Collect every username who completed this game
  const completers = [];
  for (const [username, progress] of game.userProgress.entries()) {
    if (progress.isComplete) completers.push(username);
  }

  // Decrement each completer's win count (only when > 0 to avoid negatives)
  if (completers.length > 0) {
    await Promise.all(
      completers.map(username =>
        User.findOneAndUpdate(
          { username, wins: { $gt: 0 } },
          { $inc: { wins: -1 } }
        )
      )
    );
  }

  return Sudoku.findByIdAndDelete(id);
}

// ── Leaderboard: users with wins > 0, ordered by wins desc, then username ──
export async function getLeaderboard() {
  return User.find({ wins: { $gt: 0 } }, 'username wins')
    .sort({ wins: -1, username: 1 });
}

// ── Increment wins for a user (called by POST /api/highscore) ─────────────
export async function recordWin(username) {
  return User.findOneAndUpdate(
    { username },
    { $inc: { wins: 1 } },
    { new: true, projection: 'username wins' }
  );
}

// ── Get all users who completed a specific game, with their win counts ──────
export async function getScoresByGame(gameId) {
  const game = await Sudoku.findById(gameId);
  if (!game) return [];

  const completedUsers = [];
  for (const [username, progress] of game.userProgress.entries()) {
    if (progress.isComplete) completedUsers.push(username);
  }
  if (completedUsers.length === 0) return [];

  return User.find(
    { username: { $in: completedUsers } },
    'username wins'
  ).sort({ wins: -1, username: 1 });
}
