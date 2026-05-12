# 📘 Homework 3: DQN and its variants - Report

## 🧠 HW3-1: Naive DQN vs. Experience Replay (Static Mode)

**Understanding Report:**
在 HW3-1 中，我們實作了基本的 DQN（Deep Q-Network）以及帶有經驗回放緩衝區（Experience Replay Buffer）的 DQN 變體，並在 `static` 模式的 GridWorld 中進行測試。
- **Naive DQN**: 傳統的 Q-Learning 在結合深度神經網路時，若直接使用線上學習（Online Learning），也就是每走一步就拿當前的 `(state, action, reward, next_state)` 來更新神經網路，容易因為樣本之間的**高度相關性（Correlation）**而導致訓練不穩定甚至發散。
- **Experience Replay Buffer**: 為了解決這個問題，我們引入了 Experience Replay Buffer。Agent 在與環境互動時，會將每一步的經驗存入 Buffer 中。當要更新神經網路時，從 Buffer 中隨機抽取一個 Batch 的經驗來進行訓練。這樣做有兩個主要好處：
  1. **打破資料相關性**：隨機抽樣使得訓練數據更接近獨立同分佈（i.i.d），穩定神經網路的收斂。
  2. **提高數據使用效率**：過去的經驗可以被多次重複學習，而不是用過一次就丟棄。

## ⚖️ HW3-2: Enhanced DQN Variants (Player Mode)

在 HW3-2 中，環境變得更為複雜（`player` 模式：起點隨機）。我們比較了 Double DQN 與 Dueling DQN 兩種進階變體：
- **Double DQN (DDQN)**:
  - **改善點**：解決了基本 DQN 容易發生「高估 Q 值（Overestimation bias）」的問題。
  - **原理**：在計算 目標 Q 值（Target Q）時，DDQN 使用 Main Network 來選擇最佳動作（$argmax_a Q(s', a)$），並使用 Target Network 來評估這個動作的價值（$Q_{target}(s', a_{max})$）。這種解耦動作選擇與評估的機制，大幅降低了不切實際的高 Q 值。
- **Dueling DQN**:
  - **改善點**：加速訓練，特別是當動作的選擇對於當前狀態影響不大時。
  - **原理**：修改了神經網路的架構，將輸出的 Q 值拆分為「狀態價值函數（State Value Function, $V(s)$）」與「優勢函數（Advantage Function, $A(s,a)$）」。最終 $Q(s,a) = V(s) + \left(A(s,a) - \frac{1}{|A|} \sum_{a'} A(s,a')\right)$。這使得網路能更有效地學習哪些狀態本身是有價值的，而不必在每個狀態都學會每一個動作的具體價值。

## 🔁 HW3-4: Rainbow DQN (Random Mode) Analysis

**先分析，再教你怎麼做：**
Rainbow DQN 是 DQN 家族的集大成者，整合了六種經典且互補的改進技術，能在高度隨機的環境（如 `random` 模式：所有物件隨機）中取得最佳表現：
1. **Double DQN**: 解決 Q 值高估問題。
2. **Prioritized Experience Replay (PER)**: 不再是均勻隨機抽樣，而是根據 TD Error 的大小賦予樣本不同的權重。TD Error 越大的樣本代表模型目前預測越不準確，應給予更高機率被抽到來學習。
3. **Dueling Network Architecture**: 分離狀態價值 $V$ 與動作優勢 $A$，增強對環境狀態的理解。
4. **Multi-step Learning (n-step return)**: 不只看下一步的 Reward，而是看未來 n 步的累積 Reward 來更新，有助於加速獎勵信號的傳遞，減少偏差。
5. **Distributional RL (Categorical DQN)**: 神經網路不只預測 Q 值的期望值，而是預測 Q 值的機率分佈，提供更豐富的價值評估。
6. **Noisy Nets**: 在神經網路的權重中加入雜訊來取代傳統的 $\epsilon$-greedy 策略，讓 Agent 的探索（Exploration）行為更有效率且隨環境變化而調整。

**實作建議步驟（教你怎麼做）：**
要將上述結合至我們的 Gridworld 中：
1. 先建立一個基底神經網路，套用 Dueling 架構與 Noisy Linear 層。
2. 替換掉原本簡單的 MSE Loss，改為計算 n-step 分佈的交叉熵損失（Distributional Loss）。
3. 替換標準的 Replay Buffer，使用 Prioritized Experience Replay (通常需實作 SumTree 資料結構以加速抽樣)。
4. （後續我們會在 `hw3_4_rainbow_dqn.py` 中示範一個基礎整合版本）。
