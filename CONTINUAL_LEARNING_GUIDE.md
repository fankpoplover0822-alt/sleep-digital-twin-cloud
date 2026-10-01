# Sleep Digital Twin 持續學習操作說明

本系統採用兩種更新速度：

1. 每次上傳監測或追蹤資料時，立即更新病患 Digital Twin、預測與治療排名。
2. 只有經人工確認的 Arousal 標籤及治療結果，才會進入模型再訓練。

## 正式病患流程

1. 首次上傳 `EDF + Stage.xls/xlsx + Event Grid.xls/xlsx`。
2. 執行完整 Pipeline，輸出四種治療排名。
3. 使用當時的正式模型預測未來60秒內是否出現 apnea。
4. Event Grid 的真實 apnea 事件自動建立標籤並加入學習。
5. 驗證合格後更新下一版 apnea 模型；本次顯示的預測仍保留為更新前模型結果。
6. 同一病患可再補充 PSG Follow-up、臨床報告、回診資料及穿戴裝置資料。

四種治療為：

- CPAP
- APAP
- 手術評估
- 睡眠結構調節藥物／其他治療檢討

## Arousal 回饋資料

特徵檔與標籤檔都必須具有：

```text
patient_id, epoch_index
```

建議另外提供 `study_id`；若缺少，系統會使用本次標籤檔名稱建立。標籤檔必須具有：

```text
true_label
```

匯入：

```powershell
python continual_learning_cli.py import-arousal `
  --features path\to\features.csv `
  --labels path\to\labels.csv `
  --reviewer-id TECH001 `
  --approve
```

## Arousal 模型更新

```powershell
python continual_learning_cli.py status
python continual_learning_cli.py train-arousal
python continual_learning_cli.py evaluate-arousal --version VERSION
python continual_learning_cli.py promote-arousal --version VERSION
```

`--force` 只供研究測試；正式流程應等狀態顯示已達再訓練條件。

## 治療結果資料

最低必要欄位：

```text
patient_id,study_id,treatment,baseline_ahi,follow_up_ahi,follow_up_days,outcome_status
```

只有 `outcome_status=confirmed` 可用於學習。建議增加：

```text
baseline_min_spo2,follow_up_min_spo2,symptom_improved,adherence_rate,adverse_event
```

匯入及訓練：

```powershell
python continual_learning_cli.py import-outcomes `
  --file path\to\outcomes.csv `
  --reviewer-id CLINICIAN001

python continual_learning_cli.py train-treatment
```

治療模型只會在原規則分數上做有限度調整；禁忌與臨床安全規則仍然優先。
