"""Mini NPU 시뮬레이터: python3 main.py로 실행합니다."""
import argparse
import math
from pathlib import Path

from analysis import analyze_file, print_performance
from matrix import Matrix
from npu import benchmark, decide, mac


def read_matrix(name):
    print(f"\n{name}: 3줄을 입력하세요. 한 줄에는 공백으로 나눈 숫자 3개가 필요합니다.")
    matrix = Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    for row in range(3):
        while True:
            raw = input(f"{name} {row + 1}/3행: ").strip().split()
            try:
                if len(raw) != 3:
                    raise ValueError("숫자 3개가 필요합니다.")
                values = [float(value) for value in raw]
                if not all(math.isfinite(value) for value in values):
                    raise ValueError("NaN 또는 무한대는 사용할 수 없습니다.")
                for column, value in enumerate(values):
                    matrix.set(row, column, value)
                break
            except ValueError:
                print("입력 형식 오류: 각 줄에 유한한 숫자 3개를 공백으로 구분해 입력하세요. 같은 행을 다시 입력합니다.")
    print(f"{name} 저장 확인:")
    for row in matrix.values:
        print(" ".join(f"{value:g}" for value in row))
    return matrix


def interactive(repeats=100):
    kernel_a = read_matrix("필터 A")
    kernel_b = read_matrix("필터 B")
    pattern = read_matrix("패턴")
    score_a = mac(pattern, kernel_a)
    score_b = mac(pattern, kernel_b)
    prediction = decide(score_a, score_b, "A", "B")
    print(f"\nA 점수: {score_a:.17g}\nB 점수: {score_b:.17g}")
    print("판정: " + ("판정 불가 (UNDECIDED, |A-B| < 1e-9)" if prediction == "UNDECIDED" else prediction))
    rows = [{"size": 3, "mean_ms": benchmark(pattern, kernel_a, repeats), "operations": 9}]
    print_performance(rows, repeats)


def main():
    parser = argparse.ArgumentParser(description="반복문으로 계산하는 Mini NPU Simulator")
    parser.add_argument("--data", type=Path, help="메뉴 없이 지정한 JSON 파일 분석")
    parser.add_argument("--repeats", type=int, default=100, help="성능 측정 반복 수, 최소 10 (기본 100)")
    args = parser.parse_args()
    if args.repeats < 10:
        parser.error("--repeats는 10 이상이어야 합니다.")
    print("=== 강준권의 Mini NPU Simulator ===")
    try:
        if args.data is not None:
            return 0 if analyze_file(args.data, args.repeats) is not None else 1
        while True:
            print("\n1. 사용자 입력 (3x3)\n2. data.json 분석\n0. 종료")
            choice = input("선택: ").strip()
            if choice == "0":
                return 0
            if choice == "1":
                interactive(args.repeats)
                return 0
            if choice == "2":
                path = Path(__file__).resolve().parent / "data.json"
                return 0 if analyze_file(path, args.repeats) is not None else 1
            print("0, 1, 2 중에서 선택해 주세요.")
    except (KeyboardInterrupt, EOFError):
        print("\n입력이 중단되어 종료합니다. 공식 데이터 파일은 수정하지 않았습니다.")
        return 0
    except ValueError as error:
        print(f"계산할 수 없습니다: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
