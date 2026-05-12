# 📘 Homework 3: DQN and its Variants

此專案為深度強化學習（Deep Reinforcement Learning）第三次作業，主要探討與實作 DQN (Deep Q-Network) 及其多種進階變體。我們使用 `Gridworld` 環境（包含 `static`、`player` 及 `random` 模式）來測試這些演算法的表現與學習穩定性。

🌐 **[點此進入 Interactive Web Demo (GitHub Pages)](https://Hachi282.github.io/0512DRL_HW3/)**，可以直接在瀏覽器中手動操控 Agent 並視覺化理解各演算法的行為差異！

## 🌍 環境介紹: GridWorld (4x4)
本專案使用的環境為一個 4x4 的網格世界，包含四種基本物件：
- **Player (👤)**: Agent 控制的主角。
- **Goal (⭐)**: 目標，抵達可獲得 **+10 分** 並結束回合。
- **Pit (🔥)**: 陷阱，掉入會獲得 **-10 分** 並結束回合。
- **Wall (🧱)**: 障礙物，無法通行。
- **Step Penalty**: 每走一步（包含撞牆停留在原地）都會獲得 **-1 分** 的懲罰，藉此鼓勵 Agent 尋找最短路徑。
- **State Representation**: 狀態被表示為一個 64 維度的扁平化陣列（對應不同物件在 4x4 網格中的位置），並加入微小雜訊防止產生全 0 狀態。
- **Action Space**: 離散的 4 個動作空間（上、下、左、右）。

---

## 專案結構與執行說明

請確保環境中已安裝相關套件：
```bash
pip install torch torchvision torchaudio pytorch-lightning numpy matplotlib
```

---

### 🧠 HW3-1: Naive DQN vs. Experience Replay (Static Mode)
**檔案**: `hw3_1_naive_dqn.py`

在完全靜態的 Gridworld 中（所有物件位置固定：Player 在左下，Goal 在左上），比較了原始 DQN 與加入 Experience Replay 的差異。

#### 技術與公式解析
- **Q-Learning 更新公式**:
  神經網路的目標是最小化預測 Q 值與 Target Q 值之間的均方誤差 (MSE Loss)。
  $$ L(\theta) = \mathbb{E} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta) - Q(s, a; \theta) \right)^2 \right] $$
- **Naive DQN**: 採用線上學習（Online Learning）。每走一步就用剛獲得的 $(s, a, r, s')$ 更新模型。由於連續步驟的狀態高度相關，容易導致神經網路訓練發散（Catastrophic Forgetting）。
- **Experience Replay**:
  將所有的 Transition 存入一個容量為 $N$ 的 Buffer 中 $D = \{e_1, \dots, e_N\}$。更新模型時，從 $D$ 中隨機抽取一個 Mini-batch 進行訓練。
  - **優勢**：打破資料的時間相關性（Temporal Correlation），使資料更符合 i.i.d. 假設；並提高資料的使用效率。

#### 訓練參數與實際結果
- **Hyperparameters**: 
  - `gamma` = 0.9, `learning_rate` = 1e-3
  - `epochs` = 1000, `batch_size` = 32, `memory_size` = 1000
  - `epsilon` (探索率) 從 1.0 隨每步衰減至 0.1 (`decay`=0.999)。
- **實際訓練結果**: 
  - Naive DQN 的 Loss 震盪劇烈且不穩定，甚至偶爾會出現突波。
  - Experience Replay DQN 的 Loss 曲線明顯平滑許多，能穩定收斂並找到目標。
  ![HW3-1 訓練結果](hw3_1_losses.png)

---

### ⚖️ HW3-2: Enhanced DQN Variants (Player Mode)
**檔案**: `hw3_2_enhanced_dqn.py`

在「僅玩家起點隨機」的模式下，實作了 Double DQN 與 Dueling DQN 來強化模型的穩定性與收斂速度。

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

#### 訓練參數與實際結果
- **Hyperparameters**: 與 HW3-1 相同，另外增加了 `sync_freq` = 50（每 50 步更新一次 Target Network）。
- **實際訓練結果**: 
  - Double DQN 有效抑制了不尋常的超高 Loss。
  - Dueling DQN 在相同 Epoch 下，由於能更好評估狀態好壞，其 Loss 下降速度與收斂穩定性通常優於一般架構。
  ![HW3-2 訓練結果](hw3_2_losses.png)

---

### 🔁 HW3-3: Enhance DQN for random mode (PyTorch Lightning)
**檔案**: `hw3_3_lightning_dqn.py`

為了應付全隨機（Random Mode）的高難度環境，我們將模型架構轉換為 **PyTorch Lightning**，並引入了以下訓練穩定技術：
- **Gradient Clipping (梯度裁剪)**: 防止在複雜環境下 TD Error 過大導致梯度爆炸。設定 `gradient_clip_val=1.0`。
- **Learning Rate Scheduler (學習率調度器)**: 使用 `StepLR`，每 1000 步將學習率乘上 0.9，讓模型在訓練後期能收斂到更精確的最佳解。

#### 訓練參數與實際結果
- **Hyperparameters**:
  - `learning_rate` = 1e-3, `batch_size` = 32, `memory_size` = 1000.
  - **Learning Rate Scheduler**: `StepLR` (每 1000 步乘以 0.9)。
  - **Gradient Clipping**: `gradient_clip_val=1.0`。
  - **Max Steps**: 5000（模擬約數百個回合）。
- **實際訓練結果**: 
  - 由於導入了 Lightning 的 Trainer 管理與梯度裁剪，模型在面臨起點、目標、陷阱與牆壁皆隨機變化的嚴苛環境時，依舊能保持穩定的 Loss 衰減而不會發生 NaN 或爆炸的情況。透過 Learning Rate Scheduler 的幫助，後期的震盪幅度變得更小。
  ![HW3-3 訓練結果](hw3_3_losses.png)

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

#### 訓練參數與實際結果
- **Hyperparameters**:
  - `n_step` = 3 (Multi-step Return)
  - `std_init` = 0.4 (Noisy Nets 初始雜訊強度)
  - `gamma` = 0.99, `learning_rate` = 1e-3, `epochs` = 300, `batch_size` = 32
- **實際訓練結果**: 
  - 彩虹縮水版將多種技術整合，Noisy Nets 成功取代了 $\epsilon$-greedy，讓模型在隨機環境中自動調節探索程度。配合 3-step return，Loss 下降速度在初期非常迅速，展現了強大的學習潛力。
  ![HW3-4 訓練結果](hw3_4_losses.png)

詳細的理論分析與程式邏輯探討請參考 [Report.md](Report.md)。
