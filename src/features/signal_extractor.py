from pathlib import Path

import numpy as np


class SignalExtractor:

    def __init__(self, edf_reader):
        self.reader = edf_reader
        self.raw = edf_reader.raw

    def channel_names(self):
        return self.raw.ch_names

    def has_channel(self, keyword):

        keyword = keyword.lower()

        for ch in self.raw.ch_names:

            if keyword in ch.lower():
                return True

        return False

    def get_channel(self, keyword):

        keyword = keyword.lower()

        for ch in self.raw.ch_names:

            if keyword in ch.lower():

                signal = self.raw.get_data(
                    picks=ch
                )[0]

                return ch, signal

        return None, None

    def save_channel(
        self,
        keyword,
        output_folder,
    ):

        name, signal = self.get_channel(keyword)

        if signal is None:

            print(f"{keyword} 不存在")

            return False

        output_folder = Path(output_folder)

        output_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.save(
            output_folder / f"{keyword}.npy",
            signal,
        )

        print(
            f"Saved {keyword} "
            f"({len(signal)} samples)"
        )

        return True