from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "package"

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PACKAGE_ROOT),
    )


from visual_explanation_robustness.utils.config import (
    load_config,
)
from visual_explanation_robustness.utils.paths import (
    CUB_ROOT,
    FULL_RESULT_ROOT,
    IMAGE_ROOT,
    METADATA_PATH,
    PROJECT_ROOT as DETECTED_PROJECT_ROOT,
    SEGMENTATION_ROOT,
    create_required_directories,
)


def main() -> None:
    config = load_config()

    create_required_directories()

    print("Project setup verification")
    print("-" * 60)

    print(
        f"Project root: "
        f"{DETECTED_PROJECT_ROOT}"
    )
    print(
        f"Project root exists: "
        f"{DETECTED_PROJECT_ROOT.exists()}"
    )

    print(
        f"CUB root: "
        f"{CUB_ROOT}"
    )
    print(
        f"CUB root exists: "
        f"{CUB_ROOT.exists()}"
    )

    print(
        f"Image root exists: "
        f"{IMAGE_ROOT.exists()}"
    )

    print(
        f"Segmentation root exists: "
        f"{SEGMENTATION_ROOT.exists()}"
    )

    print(
        f"Metadata exists: "
        f"{METADATA_PATH.exists()}"
    )

    print(
        f"Full result root: "
        f"{FULL_RESULT_ROOT}"
    )

    print(
        "Configured models: "
        f"{config['models']['names']}"
    )

    print(
        "Configured corruptions: "
        f"{config['corruptions']['types']}"
    )

    print(
        "Configured severities: "
        f"{config['corruptions']['severities']}"
    )

    assert DETECTED_PROJECT_ROOT == PROJECT_ROOT
    assert CUB_ROOT.exists()
    assert IMAGE_ROOT.exists()
    assert SEGMENTATION_ROOT.exists()
    assert METADATA_PATH.exists()

    print(
        "\nProject setup verification "
        "completed successfully."
    )


if __name__ == "__main__":
    main()