from __future__ import annotations

from pathlib import Path

import yaml


class ChannelMapper:
    def __init__(
        self,
        config_path: str | Path,
    ) -> None:
        self.config_path = Path(config_path)
        self.mapping_config = self._load_config()

    def _load_config(self) -> dict:
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"找不到 Channel 設定檔："
                f"{self.config_path}"
            )

        with self.config_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            config = yaml.safe_load(file)

        if not isinstance(config, dict):
            raise ValueError(
                "channel_map.yaml 格式不正確。"
            )

        return config

    def map_channels(
        self,
        channel_names: list[str],
    ) -> dict[str, str | None]:
        normalized_lookup = {
            channel.strip().lower(): channel
            for channel in channel_names
        }

        result: dict[str, str | None] = {}

        for standard_name, settings in (
            self.mapping_config.items()
        ):
            aliases = settings.get(
                "aliases",
                [],
            )

            matched_channel = None

            # 優先完全相等
            for alias in aliases:
                alias_lower = (
                    str(alias)
                    .strip()
                    .lower()
                )

                if alias_lower in normalized_lookup:
                    matched_channel = (
                        normalized_lookup[
                            alias_lower
                        ]
                    )
                    break

            # 完全相等找不到時，再用包含比對
            if matched_channel is None:
                for alias in aliases:
                    alias_lower = (
                        str(alias)
                        .strip()
                        .lower()
                    )

                    for channel in channel_names:
                        if (
                            alias_lower
                            in channel.lower()
                        ):
                            matched_channel = channel
                            break

                    if matched_channel is not None:
                        break

            result[standard_name] = (
                matched_channel
            )

        return result

    @staticmethod
    def validate_required(
        mapped_channels: dict[
            str,
            str | None
        ],
        required_channels: list[str],
    ) -> list[str]:
        return [
            channel
            for channel in required_channels
            if mapped_channels.get(channel)
            is None
        ]