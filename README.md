# 📘 Homework 3: DQN and its Variants

此專案為深度強化學習（Deep Reinforcement Learning）第三次作業，主要探討與實作 DQN (Deep Q-Network) 及其多種進階變體。我們使用 `Gridworld` 環境（包含 `static`、`player` 及 `random` 模式）來測試這些演算法的表現與學習穩定性。

## 專案結構與執行說明

請確保環境中已安裝 `torch`, `pytorch-lightning`, `numpy`, `matplotlib` 等套件。

```bash
pip install torch torchvision torchaudio pytorch-lightning numpy matplotlib
```

### 🧠 HW3-1: Naive DQN vs. Experience Replay (Static Mode)
**檔案**: `hw3_1_naive_dqn.py`

在完全靜態的 Gridworld 中（所有物件位置固定），實作並比較了最原始的 Q-Learning 結合神經網路（Naive DQN）與加入 Experience Replay Buffer 的 DQN。
- **Naive DQN**: 採用線上學習（Online Learning），容易受到樣本高度相關性的影響而導致訓練不穩定。
- **Experience Replay**: 藉由將經驗存入緩衝區並隨機抽樣進行批次訓練，有效打破了資料相關性，使神經網路收斂更為穩定。

**執行方式**:
```bash
python hw3_1_naive_dqn.py
```

### ⚖️ HW3-2: Enhanced DQN Variants (Player Mode)
**檔案**: `hw3_2_enhanced_dqn.py`

在 `player` 模式中（僅玩家起點隨機，其餘物件固定），我們進一步實作並比較了兩種 DQN 的改良版：
- **Double DQN**: 將「動作選擇」與「動作價值評估」分別交由 Main Network 與 Target Network 處理，藉此解決基本 DQN 容易發生「Q 值高估（Overestimation bias）」的問題。
- **Dueling DQN**: 改變網路架構，將輸出的 Q 值拆分為「狀態價值函數 (Value)」與「優勢函數 (Advantage)」，使得神經網路在面對多種動作選擇時，能更有效地專注於評估狀態本身的好壞，進而加速訓練。

**執行方式**:
```bash
python hw3_2_enhanced_dqn.py
```

### 🔁 HW3-3: Enhance DQN for random mode (PyTorch Lightning)
**檔案**: `hw3_3_lightning_dqn.py`

在最具挑戰性的 `random` 模式中（所有物件位置全隨機），為了提升模型的泛化能力與穩定性，我們將原本的 PyTorch DQN 模型轉換為 **PyTorch Lightning** 架構。
- 引入了 **Learning Rate Scheduler (`StepLR`)** 動態調整學習率。
- 在 Trainer 中啟用 **Gradient Clipping** 以防止梯度爆炸，使得訓練過程更加平穩。

**執行方式**:
```bash
python hw3_3_lightning_dqn.py
```

### 🌈 HW3-4: 加分題 - Rainbow DQN (Random Mode)
**檔案**: `hw3_4_rainbow_dqn.py` 及 `Report.md`

在全隨機的 Gridworld 中，我們分析了 Rainbow DQN 架構的六大組件（如 Prioritized Experience Replay, Distributional RL 等）的理論基礎，並於程式碼中實作了一個包含核心概念的**「彩虹縮水版」**：
- **Noisy Linear Layers**: 自動於權重中引入雜訊，取代傳統的 $\epsilon$-greedy 進行更有效的探索。
- **Dueling Architecture**: 評估狀態與動作優勢。
- **Multi-step (n-step) Return**: 結合未來 n 步的獎勵進行更新，加速學習。
- **Double DQN**: 減緩 Q 值高估。

詳細的理論分析請參考 `Report.md`。

**執行方式**:
```bash
python hw3_4_rainbow_dqn.py
```

## 📝 綜合分析報告
詳細的演算法原理解釋與學習心得，已整理至 `Report.md`。
