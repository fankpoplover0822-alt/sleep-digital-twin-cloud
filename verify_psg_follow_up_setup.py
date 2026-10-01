from pathlib import Path
import py_compile

ROOT = Path(__file__).resolve().parent
files = [
    ROOT / "app.py",
    ROOT / "clinical_modules" / "psg_follow_up_service.py",
]

for file in files:
    if not file.is_file():
        raise FileNotFoundError(file)
    py_compile.compile(str(file), doraise=True)

app_text = (ROOT / "app.py").read_text(encoding="utf-8")
required = [
    '"PSG_FOLLOW_UP"',
    "prepare_psg_follow_up",
    "finalize_psg_follow_up",
    "rollback_psg_follow_up",
    "儲存並執行最新睡眠分析",
]
missing = [item for item in required if item not in app_text]
if missing:
    raise AssertionError(f"app.py 缺少 PSG Follow-up 整合內容：{missing}")

print("PASS: PSG Follow-up integration is installed.")
