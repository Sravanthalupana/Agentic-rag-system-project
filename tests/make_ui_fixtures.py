"""Generate small browser-upload fixtures, without committing binary documents."""

import argparse
from pathlib import Path

from tests.fixtures import docx_bytes, pdf_bytes


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/ui-fixtures"))
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "warranty.pdf").write_bytes(pdf_bytes("The warranty lasts 2 years."))
    (args.output / "support.docx").write_bytes(docx_bytes())
    (args.output / "broken.pdf").write_bytes(b"this is not a PDF")
    print(f"Fixtures written to {args.output}")


if __name__ == "__main__":
    main()
