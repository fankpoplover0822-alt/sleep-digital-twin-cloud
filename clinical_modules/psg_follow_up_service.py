from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping


_MANIFEST_FILENAME = "manifest.json"

# 這些名稱通常屬於已累積的臨床資料，不應因 Follow-up PSG 而消失。
# finalize_psg_follow_up() 只會在新版 inference 中缺少檔案時，由 Follow-up 前快照補回；
# 不會覆蓋 Pipeline 新產生或已更新的同名檔案。
_DEFAULT_PROTECTED_CLINICAL_PATTERNS = (
    "clinical_decision",
    "clinical_update",
    "patient_update",
    "pap",
    "cpap",
    "apap",
    "bipap",
    "ent",
    "dise",
    "imaging",
    "image",
    "ct",
    "mri",
    "cephal",
    "preference",
    "comorbid",
    "medication",
    "surgery",
    "oral_appliance",
    "mad",
)

# 用於建立額外、容易瀏覽的治療推薦歷史快照。
_DEFAULT_RECOMMENDATION_PATTERNS = (
    "treatment_suitability",
    "treatment_refinement",
    "digital_twin_report",
    "recommendation",
    "treatment_recommend",
)


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _safe_name(filename: str) -> str:
    name = Path(str(filename)).name.strip()
    name = re.sub(r'[<>:"/\\|?*]', "_", name).rstrip(". ")
    if not name:
        raise ValueError("上傳檔案名稱為空白。")
    return name


def _safe_path_component(value: str, *, field_name: str) -> str:
    text = str(value).strip()
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", text).rstrip(". ")
    if not text or text in {".", ".."}:
        raise ValueError(f"{field_name} 不可為空白或無效路徑名稱。")
    return text


def _normalize_study_date(study_date: str) -> str:
    text = str(study_date).strip()
    if not text:
        raise ValueError("study_date 不可為空白。")

    try:
        return date.fromisoformat(text).isoformat()
    except ValueError as exc:
        raise ValueError(
            f"study_date 必須是 YYYY-MM-DD 格式，目前為：{study_date!r}"
        ) from exc


def _json_safe(value: Any) -> Any:
    """
    將資料轉為嚴格 JSON 可序列化格式。

    主要避免 numpy/pandas NaN、Infinity 或 Path 導致：
    ValueError: Out of range float values are not JSON compliant
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, datetime):
        return value.astimezone().isoformat(timespec="seconds")

    if isinstance(value, date):
        return value.isoformat()

    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]

    # 支援 numpy scalar、pandas scalar 等具有 item() 的物件。
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return _json_safe(item_method())
        except Exception:
            pass

    # 支援 pandas.isna 可辨識但不強制依賴 pandas。
    try:
        if value != value:  # NaN 的通用判斷
            return None
    except Exception:
        pass

    return str(value)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    safe_payload = _json_safe(dict(payload))
    temp_path.write_text(
        json.dumps(
            safe_payload,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    temp_path.replace(path)


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"無法讀取 JSON：{path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"JSON 根節點必須是物件：{path}")
    return data


def _sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        while True:
            chunk = file_obj.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _file_metadata(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path),
        "name": path.name,
        "size_bytes": stat.st_size,
        "sha256": _sha256(path),
        "modified_at": datetime.fromtimestamp(
            stat.st_mtime
        ).astimezone().isoformat(timespec="seconds"),
    }


def _iter_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path


def _build_inventory(root: Path) -> list[dict]:
    inventory: list[dict] = []
    if not root.exists():
        return inventory

    for path in _iter_files(root):
        metadata = _file_metadata(path)
        metadata["relative_path"] = str(path.relative_to(root))
        inventory.append(metadata)
    return inventory


def extract_study_key(filename: str) -> str:
    """
    從 EDF、Stage、Event Grid 檔名擷取共同的睡眠檢查識別碼。

    可處理：
    - Windows 複製檔名：- 複製
    - 英文複製檔名：- Copy
    - 數字副本：(1)、(2)
    """
    stem = Path(str(filename)).stem.strip()

    # 移除 Windows 或英文系統自動加入的副本尾碼。
    stem = re.sub(
        r"(?i)\s*[-_]\s*(?:複製|copy)(?:\s*\(\d+\))?\s*$",
        "",
        stem,
    ).strip()

    # 移除單純的 (1)、(2) 等副本編號。
    stem = re.sub(
        r"\s*\(\d+\)\s*$",
        "",
        stem,
    ).strip()

    # 移除檔案角色名稱。
    stem = re.sub(
        r"(?i)(?:[\s_-]+(?:"
        r"edf|"
        r"stage|"
        r"sleep[\s_-]*stage|"
        r"event[\s_-]*grid"
        r"))$",
        "",
        stem,
    ).strip()

    stem = re.sub(r"\s+", " ", stem)

    if not stem:
        raise ValueError(
            f"無法從檔名取得睡眠檢查識別碼：{filename}"
        )

    return stem


def _split_study_identity(study_key: str) -> tuple[str | None, str | None]:
    """Extract the PSG timestamp and stable patient code from a study key."""
    matched = re.search(
        r"(?i)(\d{8}T\d{6})\s*-\s*([a-z0-9]+)",
        str(study_key),
    )
    if matched is None:
        return None, None
    return matched.group(1), matched.group(2).casefold()


def _validate_extension(
    uploaded_file: Any,
    *,
    label: str,
    allowed_extensions: set[str],
) -> None:
    filename = getattr(uploaded_file, "name", "")
    suffix = Path(str(filename)).suffix.casefold()
    if suffix not in allowed_extensions:
        allowed_text = "、".join(sorted(allowed_extensions))
        raise ValueError(
            f"{label} 檔案格式不正確：{filename}；允許格式為 {allowed_text}。"
        )


def validate_follow_up_files(
    edf_file: Any,
    stage_file: Any,
    event_file: Any,
) -> dict:
    if edf_file is None or stage_file is None or event_file is None:
        raise ValueError("請完整上傳 EDF、Stage Excel、Event Grid Excel。")

    _validate_extension(
        edf_file,
        label="EDF",
        allowed_extensions={".edf"},
    )
    _validate_extension(
        stage_file,
        label="Stage",
        allowed_extensions={".xls", ".xlsx"},
    )
    _validate_extension(
        event_file,
        label="Event Grid",
        allowed_extensions={".xls", ".xlsx"},
    )

    keys = {
        "EDF": extract_study_key(edf_file.name),
        "Stage": extract_study_key(stage_file.name),
        "Event Grid": extract_study_key(event_file.name),
    }

    if not all(keys.values()):
        raise ValueError(f"無法從檔名辨識睡眠檢查識別碼：{keys}")

    unique_keys = {value.casefold() for value in keys.values()}
    if len(unique_keys) != 1:
        identities = {
            label: _split_study_identity(value)
            for label, value in keys.items()
        }
        patient_codes = {
            patient_code
            for _, patient_code in identities.values()
            if patient_code
        }
        timestamps = {
            timestamp
            for timestamp, _ in identities.values()
            if timestamp
        }

        if len(patient_codes) > 1:
            reason = (
                "患者代碼不一致，可能混入其他患者的檔案；"
                "請重新選擇同一位患者的三個檔案。"
            )
        elif len(timestamps) > 1:
            timestamp_groups: dict[str, list[str]] = {}
            for label, (timestamp, _) in identities.items():
                if timestamp:
                    timestamp_groups.setdefault(timestamp, []).append(label)
            majority_timestamp, majority_labels = max(
                timestamp_groups.items(), key=lambda item: len(item[1])
            )
            mismatched_labels = [
                label
                for timestamp, labels in timestamp_groups.items()
                if timestamp != majority_timestamp
                for label in labels
            ]
            if len(majority_labels) >= 2 and mismatched_labels:
                reason = (
                    "患者代碼相同，但檢查時間不一致。"
                    f"EDF／Stage主要檢查時間為 {majority_timestamp}；"
                    f"請更換：{'、'.join(mismatched_labels)}。"
                )
            else:
                reason = (
                    "患者代碼相同，但三個檔案的檢查時間不一致；"
                    "請選擇同一晚PSG產生的配對檔案。"
                )
        else:
            reason = "檔名中的睡眠檢查識別碼不一致。"

        raise ValueError(
            "無法執行：三個檔案不屬於同一次睡眠檢查。\n"
            f"原因：{reason}\n"
            f"EDF：{keys['EDF']}\n"
            f"Stage：{keys['Stage']}\n"
            f"Event Grid：{keys['Event Grid']}\n"
            "為避免把不同晚上的呼吸事件對到錯誤的生理訊號，"
            "系統不會自動合併。"
        )

    return {
        "study_key": next(iter(keys.values())),
        "detected_keys": keys,
    }


def _copy_directory_contents(
    source: Path,
    destination: Path,
    *,
    overwrite: bool = True,
) -> None:
    if not source.exists():
        return

    destination.mkdir(parents=True, exist_ok=True)

    skipped_directory_names = {
        "history",
        "recommendation_history",
        "study_history",
        "__pycache__",
    }

    def ignore_history_directories(
        current_directory: str,
        names: list[str],
    ) -> set[str]:
        del current_directory

        return {
            name
            for name in names
            if name.casefold() in skipped_directory_names
        }

    for item in source.iterdir():
        target = destination / item.name

        if (
            item.is_dir()
            and item.name.casefold() in skipped_directory_names
        ):
            continue

        if item.is_dir():
            shutil.copytree(
                item,
                target,
                dirs_exist_ok=True,
                copy_function=shutil.copy2,
                ignore=ignore_history_directories,
            )
            continue

        if target.exists() and not overwrite:
            continue

        target.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        shutil.copy2(item, target)


def _clear_directory_contents(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)

    for item in folder.iterdir():
        if item.is_dir() and not item.is_symlink():
            shutil.rmtree(item)
        else:
            item.unlink(missing_ok=True)


def _write_uploaded_file(uploaded_file: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    getbuffer = getattr(uploaded_file, "getbuffer", None)
    if callable(getbuffer):
        destination.write_bytes(bytes(getbuffer()))
        return

    getvalue = getattr(uploaded_file, "getvalue", None)
    if callable(getvalue):
        destination.write_bytes(bytes(getvalue()))
        return

    read_method = getattr(uploaded_file, "read", None)
    if callable(read_method):
        current_position = None
        tell_method = getattr(uploaded_file, "tell", None)
        seek_method = getattr(uploaded_file, "seek", None)

        if callable(tell_method):
            try:
                current_position = tell_method()
            except Exception:
                current_position = None

        if callable(seek_method):
            try:
                seek_method(0)
            except Exception:
                pass

        content = read_method()
        if isinstance(content, str):
            content = content.encode("utf-8")
        destination.write_bytes(bytes(content))

        if current_position is not None and callable(seek_method):
            try:
                seek_method(current_position)
            except Exception:
                pass
        return

    raise TypeError(
        f"不支援的上傳檔案物件：{type(uploaded_file).__name__}"
    )


def _path_matches_patterns(
    relative_path: Path,
    patterns: Iterable[str],
) -> bool:
    normalized = str(relative_path).replace("\\", "/").casefold()
    name = relative_path.name.casefold()

    return any(
        pattern.casefold() in normalized
        or pattern.casefold() in name
        for pattern in patterns
    )


def _copy_matching_files(
    source_root: Path,
    destination_root: Path,
    *,
    patterns: Iterable[str],
    overwrite: bool,
) -> list[str]:
    copied: list[str] = []

    if not source_root.exists():
        return copied

    for source_path in _iter_files(source_root):
        relative_path = source_path.relative_to(source_root)

        # recommendation_history / history 本身已經是歷史資料，
        # 不應再複製到新的推薦歷史快照中。
        excluded_directory_names = {
            "history",
            "recommendation_history",
            "study_history",
            "__pycache__",
        }

        relative_parts = {
            part.casefold()
            for part in relative_path.parts[:-1]
        }

        if relative_parts & excluded_directory_names:
            continue

        if not _path_matches_patterns(relative_path, patterns):
            continue


        destination_path = destination_root / relative_path
        if destination_path.exists() and not overwrite:
            continue

        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)
        copied.append(str(relative_path))

    return copied


def _restore_missing_clinical_artifacts(
    before_inference: Path,
    current_inference: Path,
    *,
    protected_patterns: Iterable[str],
) -> list[str]:
    """
    從 Follow-up 前快照補回新版 inference 中缺少的臨床資料。

    重要：
    - 只補「缺少」的檔案。
    - 不覆蓋 Pipeline 新產生的檔案。
    - 因此可保留 PAP / ENT / DISE / Imaging 等累積資料，
      又不會把最新更新退回舊版。
    """
    return _copy_matching_files(
        before_inference,
        current_inference,
        patterns=protected_patterns,
        overwrite=False,
    )


def _snapshot_recommendations(
    source_root: Path,
    destination_root: Path,
    *,
    recommendation_patterns: Iterable[str],
) -> list[str]:
    return _copy_matching_files(
        source_root,
        destination_root,
        patterns=recommendation_patterns,
        overwrite=True,
    )


def _manifest_path_from_manifest(manifest: Mapping[str, Any]) -> Path:
    explicit_path = manifest.get("manifest_path")
    if explicit_path:
        return Path(str(explicit_path))

    history_root = manifest.get("history_root")
    if history_root:
        return Path(str(history_root)) / _MANIFEST_FILENAME

    after_snapshot = manifest.get("after_inference_snapshot")
    if after_snapshot:
        return Path(str(after_snapshot)).parents[1] / _MANIFEST_FILENAME

    raise ValueError("manifest 缺少 manifest_path / history_root。")


def _update_manifest(
    manifest: Mapping[str, Any],
    **updates: Any,
) -> dict:
    updated = {**dict(manifest), **updates}
    path = _manifest_path_from_manifest(updated)
    updated["manifest_path"] = str(path)
    _atomic_write_json(path, updated)
    return updated


def _ensure_patient_exists(
    patient_id: str,
    patient_incoming: Path,
    patient_inference: Path,
) -> None:
    if patient_incoming.exists() or patient_inference.exists():
        return

    raise FileNotFoundError(
        "找不到既有患者資料，Follow-up PSG 不可建立新患者。\n"
        f"patient_id：{patient_id}\n"
        f"incoming：{patient_incoming}\n"
        f"inference：{patient_inference}"
    )


def prepare_psg_follow_up(
    *,
    patient_id: str,
    edf_file: Any,
    stage_file: Any,
    event_file: Any,
    incoming_root: Path,
    inference_root: Path,
    project_root: Path,
    study_date: str,
    study_type: str,
    notes: str = "",
) -> dict:
    """
    準備同一位既有患者的新 PSG Follow-up。

    此函式只負責：
    1. 驗證三個 PSG 輸入檔案。
    2. 完整保存 Follow-up 前的 incoming / inference 快照。
    3. 額外保存 Follow-up 前的治療推薦歷史。
    4. 將 active incoming 切換成這次 PSG 的三個輸入檔案。
    5. 建立可供 finalize / rollback 使用的 manifest。

    此函式不會：
    - 建立新患者。
    - 刪除歷史快照。
    - 主動重建 Clinical Decision Data。
    - 執行原始患者建立 Pipeline。
    """
    safe_patient_id = _safe_path_component(
        patient_id,
        field_name="patient_id",
    )
    normalized_study_date = _normalize_study_date(study_date)
    normalized_study_type = str(study_type).strip() or "Follow-up PSG"
    normalized_notes = str(notes).strip()

    incoming_root = Path(incoming_root)
    inference_root = Path(inference_root)
    project_root = Path(project_root)

    patient_incoming = incoming_root / safe_patient_id
    patient_inference = inference_root / safe_patient_id

    _ensure_patient_exists(
        safe_patient_id,
        patient_incoming,
        patient_inference,
    )

    validation = validate_follow_up_files(
        edf_file,
        stage_file,
        event_file,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    study_id = _safe_path_component(
        f"{normalized_study_date}_{timestamp}",
        field_name="study_id",
    )

    history_root = (
        project_root
        / "data"
        / "study_history"
        / safe_patient_id
        / study_id
    )
    before_incoming = history_root / "before" / "incoming"
    before_inference = history_root / "before" / "inference"
    after_inference = history_root / "after" / "inference"

    recommendation_history_before = (
        history_root / "recommendations" / "before"
    )
    recommendation_history_after = (
        history_root / "recommendations" / "after"
    )
    manifest_path = history_root / _MANIFEST_FILENAME

    if history_root.exists():
        raise FileExistsError(
            f"Follow-up 歷史目錄已存在，為避免覆蓋而停止：{history_root}"
        )

    history_root.mkdir(parents=True, exist_ok=False)

    manifest: dict[str, Any] = {
        "schema_version": 2,
        "operation": "psg_follow_up",
        "status": "preparing",
        "patient_id": safe_patient_id,
        "study_id": study_id,
        "study_key": validation["study_key"],
        "study_date": normalized_study_date,
        "study_type": normalized_study_type,
        "notes": normalized_notes,
        "created_at": _now_iso(),
        "detected_keys": validation["detected_keys"],
        "history_root": str(history_root),
        "manifest_path": str(manifest_path),
        "patient_incoming_path": str(patient_incoming),
        "patient_inference_path": str(patient_inference),
        "before_incoming_snapshot": str(before_incoming),
        "before_inference_snapshot": str(before_inference),
        "after_inference_snapshot": str(after_inference),
        "recommendation_history_before": str(
            recommendation_history_before
        ),
        "recommendation_history_after": str(
            recommendation_history_after
        ),
        "active_input_files": {},
        "previous_recommendation_files": [],
        "restored_clinical_files": [],
        "new_recommendation_files": [],
    }
    _atomic_write_json(manifest_path, manifest)

    try:
        # 先完整備份。任何後續錯誤都可 rollback。
        _copy_directory_contents(
            patient_incoming,
            before_incoming,
            overwrite=True,
        )
        _copy_directory_contents(
            patient_inference,
            before_inference,
            overwrite=True,
        )

        previous_recommendation_files = _snapshot_recommendations(
            patient_inference,
            recommendation_history_before,
            recommendation_patterns=_DEFAULT_RECOMMENDATION_PATTERNS,
        )

        # active incoming 僅放入本次 Follow-up PSG。
        _clear_directory_contents(patient_incoming)

        saved_files = {
            "EDF": patient_incoming / _safe_name(edf_file.name),
            "Stage": patient_incoming / _safe_name(stage_file.name),
            "Event Grid": patient_incoming / _safe_name(event_file.name),
        }

        _write_uploaded_file(edf_file, saved_files["EDF"])
        _write_uploaded_file(stage_file, saved_files["Stage"])
        _write_uploaded_file(event_file, saved_files["Event Grid"])

        active_input_files = {
            key: _file_metadata(path)
            for key, path in saved_files.items()
        }

        manifest = _update_manifest(
            manifest,
            status="prepared",
            prepared_at=_now_iso(),
            active_input_files=active_input_files,
            previous_recommendation_files=previous_recommendation_files,
            before_incoming_inventory=_build_inventory(before_incoming),
            before_inference_inventory=_build_inventory(before_inference),
        )
        return manifest

    except Exception as exc:
        # prepare 階段失敗時，立刻還原 active incoming。
        _clear_directory_contents(patient_incoming)
        _copy_directory_contents(
            before_incoming,
            patient_incoming,
            overwrite=True,
        )

        # inference 理論上尚未執行，但仍保守地還原。
        _clear_directory_contents(patient_inference)
        _copy_directory_contents(
            before_inference,
            patient_inference,
            overwrite=True,
        )

        _update_manifest(
            manifest,
            status="prepare_failed",
            failed_at=_now_iso(),
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        raise


def finalize_psg_follow_up(
    *,
    manifest: dict,
    inference_root: Path,
) -> dict:
    """
    完成 Follow-up PSG。

    執行順序：
    1. 確認 manifest 與患者一致。
    2. 從 Follow-up 前快照補回新版 inference 缺少的累積臨床資料。
    3. 保存 Follow-up 後完整 inference 快照。
    4. 保存 Follow-up 後治療推薦歷史。
    5. 更新 manifest 為 completed。

    注意：
    - 補回臨床資料時不覆蓋新版同名檔案。
    - 舊推薦已保存在 recommendations/before。
    - 新推薦會保存在 recommendations/after。
    """
    if not isinstance(manifest, dict):
        raise TypeError("manifest 必須是 dict。")

    patient_id = _safe_path_component(
        str(manifest.get("patient_id", "")),
        field_name="manifest.patient_id",
    )
    inference_root = Path(inference_root)
    patient_inference = inference_root / patient_id

    expected_inference = manifest.get("patient_inference_path")
    if expected_inference:
        expected_path = Path(str(expected_inference))
        if expected_path.resolve() != patient_inference.resolve():
            raise ValueError(
                "inference_root 與 manifest 記錄不一致，為避免寫入錯誤患者而停止。\n"
                f"manifest：{expected_path}\n"
                f"目前：{patient_inference}"
            )

    if not patient_inference.exists():
        raise FileNotFoundError(
            f"找不到 Follow-up Pipeline 輸出目錄：{patient_inference}"
        )

    before_inference = Path(
        str(manifest["before_inference_snapshot"])
    )
    after_inference = Path(
        str(manifest["after_inference_snapshot"])
    )
    recommendation_history_after = Path(
        str(manifest["recommendation_history_after"])
    )

    try:
        restored_clinical_files = _restore_missing_clinical_artifacts(
            before_inference,
            patient_inference,
            protected_patterns=_DEFAULT_PROTECTED_CLINICAL_PATTERNS,
        )

        # 避免 finalize 重跑時混入舊的 after 快照。
        _clear_directory_contents(after_inference)
        _copy_directory_contents(
            patient_inference,
            after_inference,
            overwrite=True,
        )

        _clear_directory_contents(recommendation_history_after)
        new_recommendation_files = _snapshot_recommendations(
            patient_inference,
            recommendation_history_after,
            recommendation_patterns=_DEFAULT_RECOMMENDATION_PATTERNS,
        )

        completed_manifest = _update_manifest(
            manifest,
            status="completed",
            completed_at=_now_iso(),
            restored_clinical_files=restored_clinical_files,
            new_recommendation_files=new_recommendation_files,
            after_inference_inventory=_build_inventory(after_inference),
        )
        return completed_manifest

    except Exception as exc:
        _update_manifest(
            manifest,
            status="finalize_failed",
            failed_at=_now_iso(),
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        raise


def rollback_psg_follow_up(
    *,
    manifest: dict,
    incoming_root: Path,
    inference_root: Path,
) -> None:
    """
    將 active incoming 與 active inference 還原至 Follow-up 前狀態。

    歷史目錄不會刪除，方便稽核與追蹤失敗原因。
    """
    if not isinstance(manifest, dict):
        raise TypeError("manifest 必須是 dict。")

    patient_id = _safe_path_component(
        str(manifest.get("patient_id", "")),
        field_name="manifest.patient_id",
    )

    incoming_root = Path(incoming_root)
    inference_root = Path(inference_root)

    patient_incoming = incoming_root / patient_id
    patient_inference = inference_root / patient_id

    before_incoming = Path(
        str(manifest["before_incoming_snapshot"])
    )
    before_inference = Path(
        str(manifest["before_inference_snapshot"])
    )

    if not before_incoming.exists() and not before_inference.exists():
        raise FileNotFoundError(
            "找不到 Follow-up 前快照，無法 rollback。\n"
            f"incoming snapshot：{before_incoming}\n"
            f"inference snapshot：{before_inference}"
        )

    try:
        _clear_directory_contents(patient_incoming)
        _copy_directory_contents(
            before_incoming,
            patient_incoming,
            overwrite=True,
        )

        _clear_directory_contents(patient_inference)
        _copy_directory_contents(
            before_inference,
            patient_inference,
            overwrite=True,
        )

        _update_manifest(
            manifest,
            status="rolled_back",
            rolled_back_at=_now_iso(),
            rollback_incoming_inventory=_build_inventory(
                patient_incoming
            ),
            rollback_inference_inventory=_build_inventory(
                patient_inference
            ),
        )

    except Exception as exc:
        _update_manifest(
            manifest,
            status="rollback_failed",
            failed_at=_now_iso(),
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        raise


def load_psg_follow_up_manifest(manifest_path: Path) -> dict:
    """
    讀取單一 Follow-up manifest。
    """
    return _read_json(Path(manifest_path))


def list_psg_follow_ups(
    *,
    patient_id: str,
    project_root: Path,
) -> list[dict]:
    """
    依 created_at 由新到舊列出同一患者的 Follow-up 歷史。
    """
    safe_patient_id = _safe_path_component(
        patient_id,
        field_name="patient_id",
    )
    patient_history_root = (
        Path(project_root)
        / "data"
        / "study_history"
        / safe_patient_id
    )

    if not patient_history_root.exists():
        return []

    manifests: list[dict] = []

    for manifest_path in patient_history_root.glob(
        f"*/{_MANIFEST_FILENAME}"
    ):
        try:
            manifest = _read_json(manifest_path)
            manifest.setdefault(
                "manifest_path",
                str(manifest_path),
            )
            manifests.append(manifest)
        except ValueError:
            # 單筆損壞不應阻止其他歷史顯示。
            continue

    manifests.sort(
        key=lambda item: str(item.get("created_at", "")),
        reverse=True,
    )
    return manifests
