# Go — Sudoku (Project 2)

React 18 + React Router v6 + Context API Sudoku game.

## Local Development

```bash
npm install
npm start        # http://localhost:3000
```

## Github

https://github.com/guorunhuang/Projects

## Video

https://northeastern-my.sharepoint.com/:v:/g/personal/huang_guor_northeastern_edu/IQDKV2l8etRjTbqCU9u25GVBAa-mStUOAOklC62bG0dOF3M?e=sX9F1O&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifX0%3D


## What were some challenges you faced while making this app?
The router, react state, and the render deployment.

## Given more time, what additional features, functional or design changes would you make
The local storage. Tried but can not compile, it is tricky. And more visual effects.

## What assumptions did you make while working on this assignment?
6*6 grid, 9*9 grid.

## How long did this assignment take to complete? (may less than a paragraph in length)
Around 30 hours.

## What bonus points did you accomplish?  Please link to code where relevant and add any required details.
(1)4 Bonus Points: Backtracking
src/utils/sudoku.js , func generatePuzzle() and func countSolutions()

Phase 1 — Build a complete solved board (solveSudoku): Starting from the top-left cell, find the first empty cell (value = 0). Shuffle the numbers 1–9 (or 1–6) randomly, then try each one. For each candidate, call isValidPlacement to check whether it conflicts with the same row, column, or box. If valid, place it and recurse to the next empty cell. If the recursion eventually fails (dead end), backtrack: reset the cell to 0 and try the next candidate. If all candidates fail, return false to trigger backtracking in the caller. If no empty cell is found, the board is full — return true.

Phase 2 — Remove cells while guaranteeing a unique solution (countSolutions): Shuffle all cell positions randomly. For each position, temporarily set it to 0, then run the solver again but stop as soon as it finds 2 solutions. If only 1 solution exists, keep the cell empty (the puzzle is still uniquely solvable). If 2 solutions exist, restore the original value. Repeat until enough cells are removed (18 clues kept for Easy, 29 for Normal).

(2)5 Bonus Points: Hint System
src/utils/sudoku.js , func findHintCell() and Context func SHOW_HINT action()

findHintCell scans every empty cell (value = 0) on the current user board. For each empty cell, it tests every number from 1 to size by calling isValidPlacement. It collects all numbers that pass — the valid candidates for that cell. If exactly one candidate passes, that cell has a forced answer (only one number can legally go there), so it is added to a hints list. After scanning the whole board, one cell is chosen randomly from hints and returned as { row, col, value }.

The Context stores this in hintCell. SudokuCell checks isHint and applies a green pulsing highlight — it shows the player where to look, but does not fill in the answer automatically. The player must type the number themselves.

## Deploy to Render (Static Site)

1. Push this folder to a GitHub repository.
2. Go to https://render.com → **New → Static Site**.
3. Connect GitHub repo.
4. Set these build settings:
   - **Build Command:** `npm install && npm run build`
   - **Publish Directory:** `build`
5. Click **Create Static Site**.
6. Render auto-deploys on every `git push`.

The included `render.yaml` handles the SPA rewrite rule so all routes
(`/games`, `/games/easy`, `/rules`, etc.) work correctly on reload.

## Routes

| URL              | Page           |
|------------------|----------------|
| `/`              | Home           |
| `/games`         | Game Selection |
| `/games/easy`    | Easy 6×6 Game  |
| `/games/normal`  | Normal 9×9 Game|
| `/rules`         | Rules + Credits|
| `/scores`        | Leaderboard    |
| `/login`         | Login          |
| `/register`      | Register       |

