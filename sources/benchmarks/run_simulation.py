import sys

import parallel
import sequential


def main(argv: list[str]) -> None:
    sequential.main(argv)
    parallel.main(argv)


if __name__ == "__main__":
    main(sys.argv)
