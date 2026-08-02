# Brick Blast Powerups Guide

In **Brick Blast**, powerups appear on the 8x10 grid as special blocks. When balls pass through or collide with powerups, they trigger powerful board-clearing effects and alter ball trajectories.

---

## 1. Redirect Powerup (`RedirectPowerup`)
- **Effect**: Changes the velocity vector of any ball that passes through or hits it.
- **Angles Supported**:
  - **45° Redirector**: Directs the ball diagonally upwards to the right (45 degrees).
  - **90° Redirector**: Directs the ball vertically straight up (90 degrees).
  - **145° (or 135°) Redirector**: Directs the ball diagonally upwards to the left (135–145 degrees).
- **Visual Icon**: Green triangular block with directional arrow indicator (`45°`, `90°`, or `145°`).
- **Tactical Advantage**: Helps balls escape traps at the bottom row and sends them climbing back to the top of the board for extended bounce combos.

---

## 2. Multiplier Powerup (`MultiplierPowerup`)
- **Effect**: **Redirect but balls * 3**.
- **Mechanic**: When a single ball passes through a Multiplier powerup, it redirects AND spawns **2 extra active balls**, sending a total of **3 balls** out simultaneously at **45°, 90°, and 145°**.
- **Visual Icon**: Yellow block with an `x3` or triple-arrow badge.
- **Tactical Advantage**: Exponentially increases the active ball count during a turn, allowing massive area damage across the entire 8x10 grid.

---

## 3. Laser Powerup (`LaserPowerup`)
- **Effect**: Shoots laser beams across the grid whenever a ball passes through it.
- **Variants**:
  - **Horizontal Laser**: Fires beams left and right across the entire row, damaging all bricks in that row.
  - **Vertical Laser**: Fires a beam upwards and downwards through the entire column, damaging all bricks in that column.
  - **Cross Laser (Both)**: Fires both horizontal and vertical lasers simultaneously (a 4-way cross beam).
- **Damage**: Deals **1 damage** to every brick caught in the laser beam per ball trigger.
- **Visual Icon**: Cyan/Red block with horizontal (`━`), vertical (`┃`), or cross (`╋`) laser indicators.
- **Tactical Advantage**: Perfect for thinning out high-HP bricks and clearing dangerous bottom rows before bricks touch down.
