# GoSudoku (Web Dev Project 3, 2026)

A full-stack Sudoku web application built with React, Express, and MongoDB. 
Players can browse, create, and solve puzzles across two difficulty modes, create custom board layouts, track wins on a live leaderboard, and delete games they own.

## Local Development

// ── MongoDB Username r38932537_db_user Password vntYNvd4cUNjM7bA────────────────────────────────────────────────────────────────
// mongodb+srv://r38932537_db_user:dGdI12m8anymp4SX@sudokuprojectcluster.fq7dpgl.mongodb.net/?appName=SudokuProjectCluster
const MONGODB_URL = process.env.MONGODB_URL || 'mongodb+srv://r38932537_db_user:dGdI12m8anymp4SX@sudokuprojectcluster.fq7dpgl.mongodb.net/?appName=SudokuProjectCluster';

```bash
npm install
npm run server
npm run dev        # http://localhost:5173/
```

The included `render.yaml` handles the SPA rewrite rule so all routes
(`/games`, `/games/easy`, `/rules`, etc.) work correctly on reload.

## Bonus Points Completed
AI Survey - 1pt
Password Encryption - 2pts
Delete Game - 5pts
Custom Games - 10pts

## API Reference
### Sudoku Games — /api/sudoku
GET, /api/sudoku. List all games (name, difficulty, creator, date)
POST, /api/sudoku. Generate a new Easy or Normal puzzle; returns { id }
POST, /api/sudoku/custom. Validate & save a user-supplied puzzle; returns { id }
GET, /api/sudoku/:id. Get full game + caller's progress if logged in
PUT, /api/sudoku/:id. Save board state and completion status
DELETE, /api/sudoku/:id. Delete game and roll back all completers' win counts
### High Scores — /api/highscore
GET, /api/highscore. All users with wins > 0, sorted wins desc then username asc
POST, /api/highscore. Increment win count for the logged-in user
GET, /api/highscore/:gameId. Scores for users who completed a specific game
### Users — /api/users
POST, /api/users/register. Create account; sets GoSudoku_user cookie
POST, /api/users/login. Authenticate; sets GoSudoku_user cookie
POST, /api/users/logout. Clear cookie
GET, /api/users/me. Return current user from cookie

## Tech Stack

Frontend: React 18, React Router v6, Vite
Backend: Node.js, Express 4
Database: MongoDB via Mongoose 8
Auth: HTTP-only cookies + bcrypt password hashing

## Challenges

### Win-count consistency on delete. 
Fun thing is, I initially tried to use guest (anonymous) mode to create a game and had another player complete it. However, the project requirements specify that guests are not allowed to create games. I then updated the code to enforce this rule. In the Mongo database, I located the game object that should not have been created using its unique game ID and deleted it. After that, I edited the player’s high score in the user data accordingly.
When a game is deleted, every user who completed it must lose exactly one win. The deleteGame function in backend/api/db/model/sudoku.model.js reads the userProgress map, collects all completers, and issues parallel $inc: { wins: -1 } updates guarded by wins: { $gt: 0 } so no score can go negative. Getting the sequence (read → decrement → delete) correct without a transaction required careful ordering.
### Shared vs. per-user game state. 
Sudoku boards are shared objects — any user can view any game. But each user's in-progress board must be stored independently. This was solved by embedding a userProgress Mongoose Map keyed by username inside each game document, so a single game document holds every player's board state without separate progress collection.
### Guest vs. authenticated interaction boundaries. 
The spec required guests to see but not interact with any page. This meant threading a readOnly prop through SudokuBoard → SudokuCell and guarding every onClick / onKeyDown handler, while still rendering the full board visually. The Select page additionally needed to disable the Create buttons and game-card clicks without hiding them.
### Uniqueness verification performance. 
The countSolutions backtracking function stops as soon as it finds a second solution (early-exit at limit = 2), which keeps it fast for typical puzzles. However, for a very sparse custom board with almost no clues, the search tree explodes and the API call can take several seconds. The correct long-term fix is constraint propagation (naked singles / hidden singles) before backtracking, but that was out of scope for this project.

## Assumptions

MongoDB is running locally (or a URI is provided via MONGODB_URL). No setup beyond npm install and setting the connection string is required.
Usernames are the primary identity key. There is no email field. Username collisions are prevented by a unique index on the User schema, and the session cookie stores the raw username string.
No email verification. Account creation is instant. A real product would send a verification email before activating the account.

##Time Estimate: 40 hours

## Given More Time

Puzzle rating and comments. Let players rate a puzzle after completing it, with an average displayed on the game card. A comment thread per game would add social depth.
Real-time multiplayer. Using Socket.io, two players could race to solve the same board, with each other's progress shown on a ghost overlay.

Undo history. A small move stack in SudokuContext would let players step backward through their inputs, removing the need to reset the entire board after a mistake.

## Architecture

```
project-sudoku/
├── backend/
│   ├── api/
│   │   ├── db/
│   │   │   ├── model/sudoku.model.js   # Data access layer (CRUD + win rollback)
│   │   │   └── schema/
│   │   │       ├── sudoku.schema.js    # Mongoose schema for games + userProgress
│   │   │       └── user.schema.js      # Mongoose schema for users + win count
│   │   ├── sudoku.api.js               # GET/POST/PUT/DELETE for games + custom
│   │   ├── highscore.api.js            # GET/POST /api/highscore + /:gameId
│   │   └── user.api.js                 # Register, login, logout, /me
│   └── utils.js                        # Backtracking solver, generator, validator
├── src/
│   ├── components/
│   │   ├── Navbar.jsx                  # Auth-aware nav (username pill / logout)
│   │   ├── SudokuBoard.jsx             # Grid renderer, keyboard nav, readOnly prop
│   │   ├── SudokuCell.jsx              # Individual cell with all state classes
│   │   ├── NumberPad.jsx               # Mobile-friendly digit input
│   │   └── Timer.jsx                   # Elapsed time display
│   ├── pages/
│   │   ├── Home.jsx                    # Split-screen hero with live animated grid
│   │   ├── Select.jsx                  # Game list + create/custom buttons
│   │   ├── Game.jsx                    # Game play page (hint, reset, delete)
│   │   ├── Custom.jsx                  # Custom puzzle editor (9×9 blank board)
│   │   ├── Rules.jsx                   # Timeline/magazine layout rules page
│   │   ├── Scores.jsx                  # Live leaderboard from /api/highscore
│   │   ├── Login.jsx                   # Cookie-based auth form
│   │   └── Register.jsx                # Registration with duplicate-user guard
│   ├── sudoku/
│   │   ├── SudokuContext.jsx           # Global game state (useReducer + auto-save)
│   │   └── AuthContext.jsx             # Global auth state (cookie session check)
│   └── index.css                       # Full design system (1100+ lines)
├── server.js                           # Express entry point + MongoDB connection
├── vite.config.js                      # Dev proxy: /api → localhost:8000
└── package.json
```