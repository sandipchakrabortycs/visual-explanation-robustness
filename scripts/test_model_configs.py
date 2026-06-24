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
    get_model_config,
    load_config,
)


def main() -> None:
    config = load_config()

    print("Model configuration test")
    print("-" * 70)

    for model_name in config["models"]["names"]:
        model_config = get_model_config(
            config=config,
            model_name=model_name,
        )

        print(f"\nModel: {model_name}")
        print(
            f"Batch size: "
            f"{model_config['batch_size']}"
        )
        print(
            f"Workers: "
            f"{model_config['number_of_workers']}"
        )
        print(
            f"Learning rate: "
            f"{model_config['learning_rate']}"
        )
        print(
            f"Dropout: "
            f"{model_config['dropout']}"
        )
        print(
            f"Maximum epochs: "
            f"{model_config['maximum_epochs']}"
        )
        print(
            f"Pretrained: "
            f"{model_config['pretrained']}"
        )

    print(
        "\nAll model configurations "
        "loaded successfully."
    )


if __name__ == "__main__":
    main()