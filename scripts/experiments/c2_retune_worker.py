"""Fix the compile-probe input RNG, then delegate to the unchanged v5 worker."""

import os


def main() -> int:
    from kernel_optimizer.gpu import worker_main

    worker_main._ensure_optional_deps()
    from kernelbench.eval import set_seed

    set_seed(int(os.environ["C2_RETUNE_EVALUATION_SEED"]))
    return worker_main.main()


if __name__ == "__main__":
    raise SystemExit(main())
