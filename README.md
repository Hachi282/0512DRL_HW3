# 📘 Homework 3: DQN and its Variants

此專案為深度強化學習（Deep Reinforcement Learning）第三次作業，主要探討與實作 DQN (Deep Q-Network) 及其多種進階變體。我們使用 `Gridworld` 環境（包含 `static`、`player` 及 `random` 模式）來測試這些演算法的表現與學習穩定性。

🌐 **[點此進入 Interactive Web Demo (GitHub Pages)](https://Hachi282.github.io/0512DRL_HW3/)**，可以直接在瀏覽器中手動操控 Agent 並視覺化理解各演算法的行為差異！

## 專案結構與執行說明

請確保環境中已安裝相關套件：
```bash
pip install torch torchvision torchaudio pytorch-lightning numpy matplotlib
```

---

### 🧠 HW3-1: Naive DQN vs. Experience Replay (Static Mode)
**檔案**: `hw3_1_naive_dqn.py`

在完全靜態的 Gridworld 中（所有物件位置固定），比較了原始 DQN 與加入 Experience Replay 的差異。

#### 技術與公式解析
- **Q-Learning 更新公式**:
  神經網路的目標是最小化預測 Q 值與 Target Q 值之間的均方誤差 (MSE Loss)。
  $$ L(\theta) = \mathbb{E} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta) - Q(s, a; \theta) \right)^2 \right] $$
- **Naive DQN**: 採用線上學習（Online Learning）。每走一步就用剛獲得的 $(s, a, r, s')$ 更新模型。由於連續步驟的狀態高度相關，容易導致神經網路訓練發散（Catastrophic Forgetting）。
- **Experience Replay**:
  將所有的 Transition 存入一個容量為 $N$ 的 Buffer 中 $D = \{e_1, \dots, e_N\}$。更新模型時，從 $D$ 中隨機抽取一個 Mini-batch 進行訓練。
  - **優勢**：打破資料的時間相關性（Temporal Correlation），使資料更符合 i.i.d. 假設；並提高資料的使用效率。

---

### ⚖️ HW3-2: Enhanced DQN Variants (Player Mode)
**檔案**: `hw3_2_enhanced_dqn.py`

在僅起點隨機的模式下，實作了 Double DQN 與 Dueling DQN 來強化模型的穩定性與收斂速度。

#### 1. Double DQN (DDQN)
傳統 DQN 常因為 $\max_{a'}$ 操作而發生「Q 值高估 (Overestimation Bias)」。DDQN 透過解耦「動作選擇」與「價值評估」來解決此問題：
- **動作選擇**：由 Main Network $\theta$ 決定下一個狀態的最佳動作 $a_{max} = \arg\max_{a'} Q(s', a'; \theta)$
- **價值評估**：由 Target Network $\theta^-$ 來評估該動作的 Q 值。
- **目標公式**:
  $$ Y^{DoubleDQN}_t = R_{t+1} + \gamma Q(S_{t+1}, \arg\max_a Q(S_{t+1}, a; \theta_t); \theta^-_t) $$

#### 2. Dueling DQN
改變了神經網路的末端架構，將預測拆分為兩條分支：**狀態價值 (State Value, $V$)** 與 **動作優勢 (Advantage, $A$)**。
- **架構公式**:
  $$ Q(s, a; \theta, \alpha, \beta) = V(s; \theta, \beta) + \left( A(s, a; \theta, \alpha) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s, a'; \theta, \alpha) \right) $$
- **優勢**：當環境中很多狀態下採取什麼動作並不重要（例如在無障礙的空地），神經網路不需要精確學習每個動作的 Q 值，只需準確評估狀態的價值 $V(s)$ 即可，這大幅加速了訓練。

---

### 🔁 HW3-3: Enhance DQN for random mode (PyTorch Lightning)
**檔案**: `hw3_3_lightning_dqn.py`

為了應付全隨機（Random Mode）的高難度環境，我們將模型架構轉換為 **PyTorch Lightning**，並引入了以下訓練穩定技術：
- **Gradient Clipping (梯度裁剪)**: 防止在複雜環境下 TD Error 過大導致梯度爆炸。
- **Learning Rate Scheduler (學習率調度器)**: 使用 `StepLR`，隨著 Epoch 增加逐漸降低學習率，讓模型在訓練後期能收斂到更精確的最佳解。

---

### 🌈 HW3-4: 加分題 - Rainbow DQN (Random Mode)
**檔案**: `hw3_4_rainbow_dqn.py` 及 `Report.md`

Rainbow DQN 整合了六大經典改良技術。我們實作了一個「彩虹縮水版」架構，包含了以下核心組件：
1. **Multi-step Return (n-step)**:
   比起只看下一步的 Reward，累積未來 $n$ 步的 Reward 可以加速信號傳遞，減少偏差。
   $$ R^{(n)}_t = \sum_{k=0}^{n-1} \gamma^k R_{t+k+1} + \gamma^n \max_{a} Q(S_{t+n}, a; \theta^-) $$
2. **Noisy Nets for Exploration**:
   放棄傳統的 $\epsilon$-greedy，改為在神經網路的全連接層權重中加入參數化的雜訊。讓模型自行學習何時該探索、何時該利用。
   $$ y = (b + Wx) + (\sigma^b \odot \epsilon^b + (\sigma^w \odot \epsilon^w)x) $$
3. **Dueling Architecture** & **Double DQN** 機制整合。

詳細的理論分析與程式邏輯探討請參考 [Report.md](Report.md)。
