"""공식 JSON의 키·라벨·크기를 확인하고 케이스별로 분석합니다."""
import json
from pathlib import Path
import re

from matrix import Matrix
from npu import benchmark, decide, mac, normalize_label


CROSS_3 = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
X_3 = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]


def load_filters(data):
    if not isinstance(data, dict) or not isinstance(data.get("filters"), dict):
        raise ValueError("filters 객체가 없습니다.")
    filters, errors = {}, {}
    for size in (5, 13, 25):
        key = f"size_{size}"
        try:
            raw = data["filters"][key]
            if not isinstance(raw, dict):
                raise ValueError("필터 묶음은 객체여야 합니다.")
            normalized = {}
            for label, values in raw.items():
                standard = normalize_label(label)
                if standard in normalized:
                    raise ValueError("중복된 표준 필터 라벨입니다.")
                normalized[standard] = Matrix(values, size)
            if set(normalized) != {"Cross", "X"}:
                raise ValueError("Cross와 X 두 필터가 필요합니다.")
            filters[size] = normalized
            print(f"{key} 필터 로드 완료 (Cross, X)")
        except (ValueError, KeyError, TypeError) as error:
            errors[size] = str(error)
            print(f"{key} 필터 오류: {error}")
    return filters, errors


def analyze_case(key, item, filters, errors):
    result = {"id": key, "passed": False}
    try:
        match = re.fullmatch(r"size_(\d+)_(\d+)", key)
        if not match:
            raise ValueError("패턴 키는 size_N_idx 형식이어야 합니다.")
        size = int(match.group(1))
        if size not in filters:
            raise ValueError("대응 필터 없음/손상: " + errors.get(size, f"size_{size}"))
        if not isinstance(item, dict):
            raise ValueError("패턴 항목은 객체여야 합니다.")
        pattern = Matrix(item["input"], size)
        expected = normalize_label(item["expected"])
        cross_score = mac(pattern, filters[size]["Cross"])
        x_score = mac(pattern, filters[size]["X"])
        prediction = decide(cross_score, x_score)
        passed = prediction == expected
        reason = "일치" if passed else ("동점(UNDECIDED): 허용오차 내 점수" if prediction == "UNDECIDED" else "계산된 판정과 expected가 다름")
        result.update(cross_score=cross_score, x_score=x_score, prediction=prediction,
                      expected=expected, passed=passed, reason=reason, size=size)
    except (ValueError, TypeError, KeyError) as error:
        result["reason"] = "데이터/스키마 오류: " + str(error)
    return result


def performance_rows(filters, repeats):
    cross = Matrix(CROSS_3)
    pairs = [(3, cross, cross)]
    for size in (5, 13, 25):
        if size in filters:
            pairs.append((size, filters[size]["Cross"], filters[size]["Cross"]))
    rows = []
    for size, pattern, kernel in pairs:
        try:
            rows.append({"size": size, "mean_ms": benchmark(pattern, kernel, repeats), "operations": size * size})
        except ValueError as error:
            print(f"size_{size} 성능 측정 제외: {error}")
    return rows


def print_performance(rows, repeats):
    print(f"\n성능 분석: MAC 함수 한 번, 평균/{repeats}회 (입출력 제외)")
    print("크기       평균 시간(ms)       MAC 횟수(N²)")
    for row in rows:
        print(f"{row['size']:2}x{row['size']:<2}      {row['mean_ms']:.6f}              {row['operations']}")


def analyze_file(path, repeats=100):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("patterns"), dict):
            raise ValueError("patterns 객체가 없습니다.")
        print("\n[1] 필터 로드")
        filters, errors = load_filters(data)
    except (OSError, UnicodeError, ValueError) as error:
        print(f"데이터 파일을 읽을 수 없습니다: {error}")
        return None
    print("\n[2] 패턴 분석 (라벨 정규화 적용)")
    results = []
    for key, item in data["patterns"].items():
        result = analyze_case(key, item, filters, errors)
        results.append(result)
        print(f"\n--- {key} ---")
        if "prediction" in result:
            print(f"Cross 점수: {result['cross_score']:.17g}")
            print(f"X 점수: {result['x_score']:.17g}")
            print(f"판정: {result['prediction']} | expected: {result['expected']}")
        print(("PASS" if result["passed"] else "FAIL") + " | " + result["reason"])
    rows = performance_rows(filters, repeats)
    print_performance(rows, repeats)
    failed = [result for result in results if not result["passed"]]
    print(f"\n총 테스트: {len(results)}개 / 통과: {len(results) - len(failed)}개 / 실패: {len(failed)}개")
    for result in failed:
        print(f"- {result['id']}: {result['reason']}")
    return {"total": len(results), "passed": len(results) - len(failed), "failed": failed,
            "results": results, "performance": rows, "repeats": repeats}
