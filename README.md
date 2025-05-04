# Mini AI Chess

Welcome to **Mini Chess**! This game implements an AI-driven adversarial search for a simplified version of chess. The game supports human and AI play, with the AI utilizing **Minimax and Alpha-Beta pruning**.

---

## 📌 Features
- **5x5 Mini Chess Board** with simplified chess rules
- **Multiple Game Modes**:
  - 🤜 **H-H** (Human vs Human)
  - 🤖 **H-AI** (Human vs AI)
  - 🤖🤖 **AI-AI** (AI vs AI)
- **Game Trace Logging** 📜
- **AI Evaluation Heuristics** 📈

---

## 🛠 How to Run
### Prerequisites
- Python 3.x
- Clone this repository:
  ```bash
  git clone https://github.com/Vivek-Patel123/Mini-AI-Chess.git
  cd MiniChess
  ```
- Run the game:
  ```bash
  python mini_chess.py
  ```

---

## 🎮 How to Play
- Moves must be entered in the format **"B2 B3"** (Start position → End position)
- To exit, type `exit`
- AI players automatically make moves within a time constraint
- The game ends when:
  - A **King is captured** (Win)
  - No pieces are captured for **10 consecutive turns** (Draw)

---

## 📜 Game Trace Output
The program generates a game trace file in the format:
```
gameTrace-<b>-<t>-<m>.txt
```
where:
- `b` = True/False (Alpha-Beta pruning on/off)
- `t` = Timeout in seconds
- `m` = Max number of turns

Example:
```
gameTrace-false-60-100.txt
```
---

## 📂 Project Structure
```
├── mini_chess.py  # Main game script
├── README.md      # This file
├── gameTrace-*.txt  # Generated game traces
```

---

## 🔥 Authors
- **Sarah Daccache** ✨
- **David Flores** 🚀
- **Vivek Patel** 💡

---

Enjoy the game, and may the best strategist win! ♟️🔥

