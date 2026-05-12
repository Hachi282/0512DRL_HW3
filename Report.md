# 📘 Homework 3: DQN and its variants - Report

這份報告總結了 HW3 的所有實驗設計、實作細節以及針對進階強化學習變體的分析。為了確保學術嚴謹度，本報告將明確界定各實驗的比較基準與實作範圍。

---

## 📂 作業對照表與實驗宣告

| 作業要求 | 對應檔案 | 環境模式 | 主要實作內容與說明 |
| :--- | :--- | :--- | :--- |
| **HW3-1** | `hw3_1_naive_dqn.py` | Static | 比較 **Naive DQN** (無經驗回放) 與 **Experience Replay DQN**。 |
| **HW3-2** | `hw3_2_enhanced_dqn.py` | Player | 比較 **Double DQN** 與 **Dueling DQN**。 |
| **HW3-3** | `hw3_3_lightning_dqn.py` | Random | 使用 **PyTorch Lightning** 訓練 DQN，加入 Gradient Clipping 與 StepLR。 |
| **HW3-4** | `hw3_4_rainbow_dqn.py` | Random | 加分題：**Rainbow DQN-Lite** (未實作 PER 與 Distributional RL)。 |

> [!IMPORTANT]
> **實驗設計與實作澄清聲明：**
> 1. **關於 HW3-2 的比較設計**：本實驗是將「純 Double DQN」與「單純 Dueling DQN + 標準 Target Network (未開啟 Double 更新)」做平行的效能比較。我們的目的是分別探討「解耦動作選擇」與「分離狀態價值」帶來的個別效益，而非探討兩者疊加（Dueling Double DQN）的組合效應。
> 2. **關於 HW3-3 PyTorch Lightning DataLoader**：由於 DQN 的訓練資料是由 Agent 與環境互動時的 Replay Buffer 動態產生，並不存在標準的固定 Dataset。因此，我們在 Lightning 的實作中使用了一個 Dummy DataLoader（如 `torch.arange(10000)`）來觸發 Lightning 內部的 Training Step 迴圈，而真正的 Batch 資料抽樣則是寫在 `training_step` 中從 Buffer 動態提取。
> 3. **關於 HW3-4 的 Rainbow DQN-Lite**：這是一份部分整合的「縮水版」Rainbow。我們實作了 Multi-step Return、Noisy Nets、Dueling 與 Double 機制；但**尚未實作 Prioritized Experience Replay (PER/SumTree)** 以及 **Distributional RL (Categorical DQN)**（仍使用 MSE Loss 評估期望值而非 KL Divergence 評估機率分佈）。

---

## 🧠 HW3-1: Naive DQN vs. Experience Replay (Static Mode)

**Understanding Report:**
在 HW3-1 中，我們在 `static` 模式的 GridWorld 中測試了 DQN。
- **Naive DQN**: 傳統的 Q-Learning 若直接使用線上學習（每走一步就拿當前的經驗更新神經網路），會因為樣本之間高度的時間相關性而導致訓練不穩定甚至發散（Catastrophic Forgetting）。
- **Experience Replay Buffer**: Agent 會將經驗存入 Buffer 中，更新時隨機抽取一個 Batch。
  1. **打破資料相關性**：隨機抽樣使得訓練數據更接近獨立同分佈（i.i.d）。
  2. **提高數據使用效率**：過去的經驗可以被多次重複學習。
- **評估指標 (Loss & Reward per Episode)**：單看 Loss 震盪下降不代表 Agent 真的學會了。因此我們不僅追蹤 Loss，還追蹤了每個 Episode 的 Total Reward。結果證明 Replay Buffer 不僅能平滑 Loss，也能讓 Agent 在訓練後期穩定達到 Average Reward 近滿分 (10分) 的最佳狀態。

---

## ⚖️ HW3-2: Enhanced DQN Variants (Player Mode)

環境變得更複雜（起點隨機）。我們比較了兩種進階變體：
- **Double DQN (DDQN)**:
  - **改善點**：解決了基本 DQN 容易發生「高估 Q 值（Overestimation bias）」的問題。
  - **原理**：在計算 Target Q 時，使用 Main Network 選擇最佳動作，並使用 Target Network 來評估這個動作的價值（$Q_{target}(s', \arg\max_a Q(s', a))$）。
- **Dueling DQN**:
  - **改善點**：加速訓練，特別是當動作的選擇對於當前狀態影響不大時。
  - **原理**：將輸出的 Q 值拆分為「狀態價值 $V(s)$」與「優勢函數 $A(s,a)$」。最終 $Q(s,a) = V(s) + \left(A(s,a) - \frac{1}{|A|} \sum_{a'} A(s,a')\right)$。這使得網路能更有效地學習哪些狀態本身是安全的。
- **實驗指標**：從輸出的 Episode Reward 曲線可以看到，Dueling DQN 比起 Double DQN 能更快且更穩定地提升 Average Reward，減少陷入陷阱的次數。

---

## 🔁 HW3-3: Enhance DQN for Random Mode (PyTorch Lightning)

在 HW3-3 中，我們將模型轉換為 **PyTorch Lightning** 架構，以應對所有物件（包含玩家、目標、陷阱、牆壁）皆隨機配置的高難度環境。
- **框架轉換的挑戰**：DQN 的訓練資料是動態從環境與 Replay Buffer 收集的，而非固定的資料集。因此我們採用了 Dummy DataLoader 來滿足 Lightning 的框架要求，將核心的 `makeMove` 與 Buffer 抽樣邏輯寫入 `training_step` 中。
- **穩定技術導入**：
  1. **Gradient Clipping (梯度裁剪)**：在全隨機環境中，極易因為極端的狀態分佈產生過大的 TD Error，進而導致梯度爆炸 (Loss NaN)。我們設定了 `gradient_clip_val=1.0` 來將梯度限制在安全範圍內。
  2. **Learning Rate Scheduler**：使用 `StepLR` 每 1000 步衰減學習率，使得模型在訓練後期能夠收斂得更為細緻。
- **實驗指標**：藉由 Lightning 的嚴謹架構與上述穩定技術，即使在隨機環境下，Episode Reward 也能在經歷數千步的探索後，呈現出穩定的上升趨勢。

---

## 🔁 HW3-4: Rainbow DQN 分析與實作建議

Rainbow DQN 是 DQN 家族的集大成者，理論上包含六種改進：
1. **Double DQN**: 解決 Q 值高估問題。
2. **Prioritized Experience Replay (PER)**: 根據 TD Error 給予樣本不同權重，誤差越大的樣本越容易被抽到。
3. **Dueling Network Architecture**: 分離狀態價值與動作優勢。
4. **Multi-step Learning (n-step return)**: 看未來 n 步的累積 Reward 來更新，加速獎勵信號傳遞。
5. **Distributional RL (Categorical DQN)**: 神經網路預測 Q 值的機率分佈，提供更豐富的評估。
6. **Noisy Nets**: 在網路權重中加入雜訊取代 $\epsilon$-greedy，讓模型自行學習探索。

**實作建議（對應至我們的 Rainbow-Lite）：**
在我們的 `hw3_4_rainbow_dqn.py` 中，我們成功將 1, 3, 4, 6 整合進了架構中。我們捨棄了 $\epsilon$-greedy，改用 `NoisyLinear` 層，並實作了 n-step buffer（$n=3$）來累積未來的 Reward。
- **實驗指標**：即使只實作了四項技術，Agent 在全隨機的環境中已經展現了極高的尋路成功率。從 Episode Reward 曲線中可以明顯看出，Rainbow-Lite 架構能以極少的 Epoch 數將 Average Reward 快速推升，展現了強悍的樣本效率 (Sample Efficiency)。
