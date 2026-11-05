"""정사각 행렬의 크기와 숫자를 검증하고 행·열 위치로 값을 다룹니다."""
import math


class Matrix:
    def __init__(self, values, expected_size=None):
        if not isinstance(values, list) or not values:
            raise ValueError("행렬은 비어 있지 않은 행 목록이어야 합니다.")
        size = len(values)
        if expected_size is not None and size != expected_size:
            raise ValueError(f"행 수 불일치: 기대 {expected_size}, 실제 {size}")
        converted = []
        for row_number, row in enumerate(values, start=1):
            if not isinstance(row, list) or len(row) != size:
                raise ValueError(f"{row_number}행 열 수 불일치: 각 행은 {size}개여야 합니다.")
            result = []
            for value in row:
                if type(value) not in (int, float):
                    raise ValueError("행렬에는 숫자만 넣을 수 있습니다.")
                try:
                    number = float(value)
                except OverflowError:
                    raise ValueError("표현 가능한 범위를 넘은 숫자입니다.") from None
                if not math.isfinite(number):
                    raise ValueError("NaN 또는 무한대는 사용할 수 없습니다.")
                result.append(number)
            converted.append(result)
        self.size = size
        self.values = converted

    def get(self, row, column):
        return self.values[row][column]

    def set(self, row, column, value):
        if type(value) not in (int, float):
            raise ValueError("숫자만 저장할 수 있습니다.")
        try:
            number = float(value)
        except OverflowError:
            raise ValueError("표현 가능한 범위를 넘은 숫자입니다.") from None
        if not math.isfinite(number):
            raise ValueError("유한한 숫자만 저장할 수 있습니다.")
        self.values[row][column] = number
