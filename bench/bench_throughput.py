"""Script for benchmarking throughput performance"""
import argparse

def main(args: argparse.Namespace):
    print(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmarking throughput performance",
    )
    args = parser.parse_args()
    main(args)
