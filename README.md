# 7×7 五子棋位子牌學習 AI

大學人工智慧課程作業。AI 先手，用「位子牌組」選位置（牌越多越容易被抽中），
每局結束後依勝負增減牌數，經由大量對弈從「一直輸」學到「會贏」。

- 規則：7×7 棋盤，自由五子棋（五子以上連線即勝，無禁手），黑棋先手。
- 主體 AI 完全自行實作（`agents/menace_agent.py`、`agents/pattern_menace_agent.py`）。
- 外部引擎 Rapfi 只作為「強力 AI」與評估對手，不參與獎懲學習。

## 三個版本的學習 AI

| | 修正前：整盤局面位子牌 | 第一次修正：棋形位子牌 | 第二次修正：棋形計數牌（預設） |
|---|---|---|---|
| 檔案 | `agents/menace_agent.py` | `agents/pattern_menace_agent.py` | `agents/gradient_card_agent.py` |
| 位子牌綁定 | 完整的 49 格盤面 | 四個方向左右各 4 格的完整排列 | 四個方向「自己/對手在未被擋的五格內最多幾子」（共 33 種牌） |
| 某位置的牌數 | 該盤面牌組中此位置的牌數 | 四個方向棋形牌數相乘 | 同左 |
| 選擇機率 | P(a\|s) = w(s,a) / Σ w(s,b) | 同左 | 同左 |
| 獎懲 | 勝 +3、敗 −1、和 +1（整數加減） | 同左 | 按比例增減，並讓其他候選位置相對變化；越接近終局調整越大 |
| 訓練對手 | 確定性規則型 | 確定性規則型 | 混合：自我對弈＋多種對手，先後手輪流 |
| 攻防測驗（擋四） | 8.9%（＝亂下） | 6.9% | 75.3% |

攻防測驗是「必須擋四」的局面中，AI 抽中正確位置的機率，3～5 個種子的平均。

### 第四種模式：只靠規則（`agents/rule_only_card_agent.py`）

除了五子棋規則（哪裡能下、誰贏、和局），程式不預先判斷任何東西：沒有「線」「子數」「攻」「守」，
選位置時也不檢查「下這裡會不會連五」。每個空位的牌數 = 10 × e^(10 × V)，V 是「下在這裡之後的盤面」的勝算估值，
由盤面上所有 3×3、2×4、4×2 小區塊的原始內容（自己／對手／空）組成的牌加總而得；終局後依勝負回頭修正（TD(λ)）。
只用自我對弈（90%）與隨機對手（10%）訓練，3 個種子各 30 萬局：攻防測驗能連五 43.9%、擋四 17.6%、擋活三 24.9%
（亂下約 8.5%）；牌值最高的小區塊全是「自己 4 子一排」，是 AI 自己發現的。但對規則型對手勝率仍低，
原因見檢討報告第 10 節。

```bash
python main.py experiment configs/experiments_v4.json --parallel 3
```

### 第五種模式：位子牌＋神經網路（`agents/neural_card_agent.py`，需要 PyTorch）

和第四種模式一樣只靠規則、一樣用「牌數 = 10 × e^(10 × V)」抽牌與 TD(λ) 獎懲，只是 V 改由小型卷積神經網路估計
（輸入只有自己的子、對手的子兩張平面與一張全 1 平面）。3 個種子各 30 萬局：攻防測驗能連五 84.2%、擋四 73.5%、擋活三 65.7%，
對隨機版規則型先手勝率 57.8%（第四種模式 13.2%）；但對棋形計分規則型、Minimax、Rapfi 仍然 0 勝。詳見檢討報告第 11 節。

```bash
pip install -r requirements-nn.txt
```

```bash
python main.py experiment configs/experiments_v5.json --parallel 3
```

沒有安裝 PyTorch 時，其他四種模式照常執行，對弈介面也不會出現這個選項。
為什麼要修正、每次修正的效果，請看 `docs/檢討報告.md` 與 `data/reports/experiment_report.md`。

## 安裝

需要 Python 3.10 以上。

```bash
pip install -r requirements.txt
```

### 外部引擎 Rapfi（選用）

沒有 Rapfi 也能執行所有功能，只是對弈介面少一個選項、實驗會略過 Rapfi 評估並註明原因。

1. 取得 Rapfi 執行檔：<https://github.com/dhbloo/rapfi>（GitHub Releases 下載 Windows 版，或依官方說明自行編譯）。
2. 把 `pbrain-rapfi.exe`（Windows）或 `pbrain-rapfi`（Linux/macOS）放到 `engines/rapfi/`，
   或設定環境變數 `RAPFI_PATH` 指向執行檔。
3. Rapfi 為 GPL-3.0 授權，授權全文在 `engines/rapfi/LICENSE-Rapfi-GPL3.txt`。

本專案在 Linux 上以 Rapfi 原始碼（commit `1387d113`，版本 0.44.01）編譯並測試；
Windows 版執行檔尚未在本專案中實測。

## 使用方式

```bash
python main.py play
```
開啟 Pygame 對弈介面（預設對手：第二次修正的位子牌 AI，你執白）。可切換對手、執黑/執白、顯示 AI 上一手的位子牌機率、開始/停止訓練、保存模型。對局自動記錄在 `data/matches/human_games.jsonl`。

```bash
python main.py play --cli --opponent rule --human black
```
文字模式對弈（不需要 Pygame）。對手可選 `learned`（第二次修正）、`ruleonly`（第四種模式）、`neural`（第五種模式，需 PyTorch）、`improved`（第一次修正）、`baseline`（修正前）、`rule`、`pattern`、`minimax`、`random`、`rapfi`。

```bash
python main.py train --agent card --opponent rule --games 3000 --out data/models/my_model.json --log data/matches/my_train.jsonl
```
訓練一個學習 AI。`--agent card` 是第二次修正（預設）、`pattern` 是第一次修正、`menace` 是修正前、`ruleonly` 是第四種模式、`neural` 是第五種模式；`--color white` 讓 AI 當後手；`--load` 從既有權重繼續訓練。
（混合對手＋自我對弈的訓練請用實驗設定檔，見下方。）

```bash
python main.py evaluate data/models/card_v3_mixed.json --opponents rule pattern random --games 200
```
以凍結權重評估（不改動模型）。

```bash
python main.py experiment configs/experiments.json --parallel 4
```
重跑修正前與第一次修正的實驗（45 組，每組 10000 局；在 4 核心的環境約需 30～60 分鐘）。用 `--only improved_first baseline_first` 只跑指定實驗。既有結果不會被覆蓋，除非加 `--force`。實驗設計寫在 `configs/make_experiments.py`。

```bash
python main.py experiment configs/experiments_v3.json --parallel 4
```
重跑第二次修正的實驗（18 組，設計在 `configs/make_experiments_v3.py`）。

```bash
python main.py report
```
由實驗結果產生圖表與 `data/reports/experiment_report.md`。

```bash
python main.py replay data/matches/improved_first_train.jsonl improved_first-train-001491 --step
```
逐手重播一局。

```bash
python main.py verify data/matches/improved_first_train.jsonl
```
重播檔案中每一局，核對紀錄的勝負。

```bash
python -m pytest
```
執行自動化測試。

## 專案結構

```text
core/            棋盤、五連判定、對弈流程與重播、八種對稱
agents/          學習 AI（五種模式）、混合對手、隨機、規則型、Minimax、Rapfi
training/        獎懲規則、位子牌/對手記憶、訓練迴圈、凍結評估、攻防測驗、實驗執行器
reporting/       棋譜 JSONL 紀錄、報告與圖表產生
ui/              Pygame 介面、文字模式對弈
configs/         實驗設定（make_experiments.py 產生 experiments.json）
data/models/     訓練後的學習資料（JSON、.json.gz；神經網路為 .pt）
data/matches/    每一局棋譜（JSONL；輔助實驗為 .jsonl.gz）
data/experiments/每組實驗的 config / train.csv / eval.csv / summary.json
data/reports/    圖表與 experiment_report.md
docs/            檢討報告、架構圖
tests/           pytest 測試
engines/rapfi/   外部引擎（選用）
```

## 棋譜格式

每局一行 JSON，主要欄位：

| 欄位 | 內容 |
|---|---|
| `game_id` | 例如 `improved_first-train-001491` |
| `ai` / `opponent` | 類型、版本、參數 |
| `seed` | 該局亂數種子（可重現） |
| `ai_color` / `ai_first` | 先後手 |
| `moves` | 每一手：`n` 第幾手、`player`、`row`、`col`、`by`（ai/opponent） |
| `result` / `winner` | `ai_win` / `ai_loss` / `draw` |
| `reward_params` | 本局使用的獎懲參數 |
| `learning` | AI 每一手：局面、選擇、選擇前牌數、牌組總數、增減、選擇後牌數（第一次、第二次修正另含四個方向的棋形與各自牌數；第二次修正另含 θ 調整量 `delta_theta`） |

主要實驗（`baseline_first`、`improved_first`、`baseline_second`、`improved_second`、`card_v3_rule_first`、`card_v3_mixed`）保存完整 `learning` 欄位；
其他實驗只保存落子與結果，以 gzip 壓縮。

## 測試涵蓋

交接文件列出的 12 項必要測試都有對應（`tests/`）：四方向五連、邊界、重複落子、終局後落子、平局、
對稱標準形與座標還原、牌數歸零仍可選、保存/讀取一致、棋譜重播、先後手、評估不改權重；
另有獎懲計算、對手記憶、重現性、按比例增減的方向、自我對弈共用牌組、混合對手、攻防測驗出題、Minimax、Rapfi 完整對局（未安裝 Rapfi 時自動略過）等測試。
