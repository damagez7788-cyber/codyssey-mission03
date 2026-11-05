"""같은 위치의 값을 곱하고 누적하는 MAC 연산과 판정입니다."""
import math
import time

EPSILON = 1e-9


def mac(pattern, kernel):
    if pattern.size != kernel.size:
        raise ValueError("패턴과 필터의 크기가 다릅니다.")
    total = 0.0
    for row in range(pattern.size):
        for column in range(pattern.size):
            total += pattern.get(row, column) * kernel.get(row, column)
    if not math.isfinite(total):
        raise ValueError("MAC 결과가 숫자 표현 범위를 넘었습니다.")
    return total


def decide(score_a, score_b, label_a="Cross", label_b="X", epsilon=EPSILON):
    if not all(math.isfinite(value) for value in (score_a, score_b, epsilon)) or epsilon <= 0:
        raise ValueError("점수와 허용오차는 유한해야 하며 허용오차는 양수여야 합니다.")
    if abs(score_a - score_b) < epsilon:
        return "UNDECIDED"
    return label_a if score_a > score_b else label_b


def normalize_label(value):
    if not isinstance(value, str):
        raise ValueError("라벨은 글자여야 합니다.")
    label = value.strip().casefold()
    if label in ("+", "cross"):
        return "Cross"
    if label == "x":
        return "X"
    raise ValueError(f"알 수 없는 라벨: {value!r}")


def benchmark(pattern, kernel, repeats=100):
    if type(repeats) is not int or repeats < 10:
        raise ValueError("시간 측정은 10회 이상 반복해야 합니다.")
    mac(pattern, kernel)  # 최초 호출 준비 비용을 측정값에서 분리합니다.
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        mac(pattern, kernel)
        elapsed = time.perf_counter_ns() - start
        samples.append(elapsed / 1_000_000)
    return sum(samples) / repeats
