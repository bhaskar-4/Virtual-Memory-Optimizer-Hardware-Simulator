import argparse
import csv
import random
from pathlib import Path


DEFAULT_OPERATIONS = 200_000
DEFAULT_MATRIX_SIZE = 256
MEMORY_BASE_HEX = 0x7FFF00000000


def generate_memory_traces(
    output_filename: str | Path = "memory_trace.txt",
    num_operations: int = DEFAULT_OPERATIONS,
    matrix_size: int = DEFAULT_MATRIX_SIZE,
    pattern: str = "strided-scan",
    seed: int = 42,
) -> int:
    if num_operations <= 0:
        raise ValueError("num_operations must be positive")
    if matrix_size <= 0:
        raise ValueError("matrix_size must be positive")
    if pattern not in {"strided-scan", "tensor-local"}:
        raise ValueError("pattern must be 'strided-scan' or 'tensor-local'")

    rng = random.Random(seed)
    input_base = MEMORY_BASE_HEX
    weight_base = input_base + matrix_size * matrix_size * 8
    output_base = weight_base + matrix_size * matrix_size * 8
    operation_count = 0

    with Path(output_filename).open("w", newline="", encoding="utf-8") as trace_file:
        writer = csv.writer(trace_file)

        def record(operation: str, address: int) -> None:
            writer.writerow((operation_count, operation, hex(address)))

        while operation_count < num_operations:
            for row in range(matrix_size):
                if operation_count >= num_operations:
                    break
                row_index = (row * 7) % matrix_size if pattern == "strided-scan" else row

                for column in range(matrix_size):
                    if operation_count >= num_operations:
                        break
                    column_index = matrix_size - 1 - column if pattern == "strided-scan" else column

                    for inner in range(matrix_size):
                        if operation_count >= num_operations:
                            break

                        input_offset = (row_index * matrix_size + inner) * 8
                        record("R", input_base + input_offset)
                        operation_count += 1

                        if operation_count < num_operations:
                            weight_offset = (inner * matrix_size + column_index) * 8
                            record("R", weight_base + weight_offset)
                            operation_count += 1

                        if pattern == "tensor-local" and operation_count < num_operations and rng.random() < 0.02:
                            noise_address = MEMORY_BASE_HEX - rng.randint(1_000, 5_000)
                            record("W", noise_address)
                            operation_count += 1

                    if pattern == "tensor-local" and operation_count < num_operations:
                        output_offset = (row * matrix_size + column) * 8
                        record("W", output_base + output_offset)
                        operation_count += 1

    return operation_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a tensor-style memory trace.")
    parser.add_argument("--output", default="memory_trace.txt")
    parser.add_argument("--operations", type=int, default=DEFAULT_OPERATIONS)
    parser.add_argument("--matrix-size", type=int, default=DEFAULT_MATRIX_SIZE)
    parser.add_argument("--pattern", choices=("strided-scan", "tensor-local"), default="strided-scan")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    count = generate_memory_traces(args.output, args.operations, args.matrix_size, args.pattern, args.seed)
    print(f"Generated {count:,} {args.pattern} memory references in {args.output}")


if __name__ == "__main__":
    main()