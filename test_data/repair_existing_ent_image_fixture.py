from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    target = (
        Path.home() / "Downloads" / "clinical_test_bank" / "clinical_test_bank"
        / "ENT" / "ent_severe_obstruction_scan.jpg"
    )
    image = Image.new("RGB", (1800, 1250), "white")
    draw = ImageDraw.Draw(image)
    font_path = Path(r"C:\Windows\Fonts\arial.ttf")
    bold_path = Path(r"C:\Windows\Fonts\arialbd.ttf")
    font = ImageFont.truetype(str(font_path), 52)
    bold = ImageFont.truetype(str(bold_path), 68)
    lines = [
        ("ENT EXAMINATION - SYNTHETIC TEST DATA", bold),
        ("nasal_septal_deviation: yes", font),
        ("inferior_turbinate_hypertrophy: yes", font),
        ("tonsil_hypertrophy: 3", font),
        ("soft_palate_abnormality: yes", font),
        ("mallampati: 4", font),
        ("nasal_obstruction: yes", font),
        ("retrognathia: yes", font),
        ("exam_date: 2026-08-01", font),
        ("NOT FOR CLINICAL USE", bold),
    ]
    y = 70
    for text, selected_font in lines:
        draw.text((70, y), text, fill="black", font=selected_font)
        y += 112
    image.save(target, format="JPEG", quality=96, subsampling=0)
    print(target)


if __name__ == "__main__":
    main()
