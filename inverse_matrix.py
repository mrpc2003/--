import csv
from pathlib import Path
from typing import List, Optional, Tuple

Matrix = List[List[float]]


# -----------------------------
# 행렬 관련 유틸리티 함수
# -----------------------------

def is_square(matrix: Matrix) -> bool:
    return len(matrix) > 0 and all(len(row) == len(matrix) for row in matrix)


def zeros(n: int, m: int) -> Matrix:
    return [[0.0 for _ in range(m)] for _ in range(n)]


def identity_matrix(n: int) -> Matrix:
    I = zeros(n, n)
    for i in range(n):
        I[i][i] = 1.0
    return I


def copy_matrix(A: Matrix) -> Matrix:
    return [row[:] for row in A]


def multiply_matrices(A: Matrix, B: Matrix) -> Matrix:
    n, p = len(A), len(A[0]) if A else 0
    p2, m = len(B), len(B[0]) if B else 0
    if p != p2:
        raise ValueError("Incompatible shapes for multiplication")
    C = zeros(n, m)
    for i in range(n):
        for k in range(p):
            aik = A[i][k]
            if aik == 0:
                continue
            for j in range(m):
                C[i][j] += aik * B[k][j]
    return C


def matrix_equal(A: Matrix, B: Matrix, tol: float = 1e-10) -> bool:
    if A is None or B is None:
        return False
    if len(A) != len(B) or (A and len(A[0]) != len(B[0])):
        return False
    n, m = len(A), len(A[0]) if A else 0
    for i in range(n):
        for j in range(m):
            if abs(A[i][j] - B[i][j]) > tol:
                return False
    return True


def print_matrix(matrix: Matrix, title: str = "Matrix", digits: int = 4) -> None:
    """
    행렬을 보기 좋게(박스 문자를 사용해) 출력한다.
    - 각 열은 최대 너비에 맞춰 정렬한다.
    - 매우 작은 절댓값은 0으로 표시하여 -0.0000 출력 방지.
    """
    print(f"\n=== {title} ===")
    if not matrix:
        print("<empty>")
        print("-" * 40)
        return

    n = len(matrix)
    m = len(matrix[0]) if matrix else 0
    fmt = f"{{:.{digits}f}}"
    zero_threshold = 10 ** (-(digits + 1))  # -0.0000 방지용 임계값

    # 문자열로 포맷팅 + 열 너비 계산
    str_rows = []
    col_widths = [0] * m
    for i in range(n):
        row_strs = []
        for j in range(m):
            x = matrix[i][j]
            if abs(x) < zero_threshold:
                x = 0.0
            s = fmt.format(x)
            # "-0.0000" 같은 출력 방지
            if s.startswith("-0.") and float(s) == 0.0:
                s = "0." + s.split(".", 1)[1]
            row_strs.append(s)
            col_widths[j] = max(col_widths[j], len(s))
        str_rows.append(row_strs)

    # 박스 드로잉 문자
    for i, row in enumerate(str_rows):
        if n == 1:
            lbr, rbr = "⎡", "⎤"
        elif i == 0:
            lbr, rbr = "⎡", "⎤"
        elif i == n - 1:
            lbr, rbr = "⎣", "⎦"
        else:
            lbr, rbr = "⎢", "⎥"
        body = " ".join(s.rjust(col_widths[j]) for j, s in enumerate(row))
        print(f"{lbr} {body} {rbr}")
    print("-" * 40)


def input_matrix() -> Optional[Matrix]:
    """
    사용자에게 n×n 정방행렬 입력을 요청하고 검증한다.
    - n은 양의 정수여야 하며, 잘못 입력하면 재입력 요청
    - 각 행은 공백으로 구분된 n개의 실수로 구성
    - 성공 시 float 기반 2차원 리스트 반환
    - EOF 입력 시 None 반환으로 종료 처리
    """
    try:
        while True:
            raw_n = input("정방행렬의 크기 n을 입력하세요 (예: 3): ").strip()
            if not raw_n:
                continue
            try:
                n = int(raw_n)
                if n <= 0:
                    print("n은 양의 정수여야 합니다. 다시 입력하세요.")
                    continue
                break
            except ValueError:
                print("정수를 입력하세요. 예: 3")

        print("행렬의 각 행을 공백으로 구분하여 입력하세요.")
        print("예: 1 2 3")
        mat: Matrix = []
        for i in range(n):
            while True:
                raw = input(f"행 {i+1}/{n}: ").strip()
                parts = raw.split()
                if len(parts) != n:
                    print(f"원소 개수가 {n}개가 아닙니다. 다시 입력하세요.")
                    continue
                try:
                    row = [float(x) for x in parts]
                except ValueError:
                    print("숫자만 입력하세요. 예: 1 2.5 -3")
                    continue
                mat.append(row)
                break
        return mat
    except EOFError:
        return None


# -----------------------------
# 파일 입출력 유틸리티
# -----------------------------

def _validate_rectangular(data: Matrix) -> Matrix:
    """
    읽어들인 데이터가 직사각형이면서 정방 행렬인지 검증한다.
    """
    if not data:
        raise ValueError("빈 행렬은 허용되지 않습니다.")
    row_len = len(data[0])
    for row in data:
        if len(row) != row_len:
            raise ValueError("행마다 열의 개수가 다릅니다.")
    if row_len != len(data):
        raise ValueError("정방 행렬이 아닙니다. n×n 형태여야 합니다.")
    return data


def read_matrix_from_csv(path: str) -> Matrix:
    """
    CSV 파일에서 행렬을 읽어 float 행렬로 반환한다.
    """
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        data: Matrix = []
        for row in reader:
            if not row:
                continue
            try:
                data.append([float(x) for x in row])
            except ValueError as exc:
                raise ValueError(f"CSV에 숫자가 아닌 값이 포함되어 있습니다: {row}") from exc
    return _validate_rectangular(data)


def read_matrix_from_text(path: str, delimiter: Optional[str] = None) -> Matrix:
    """
    공백 또는 지정한 구분자로 이루어진 텍스트 파일에서 행렬을 읽는다.
    """
    data: Matrix = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split(delimiter) if delimiter else stripped.split()
            try:
                data.append([float(x) for x in parts])
            except ValueError as exc:
                raise ValueError(f"텍스트 파일에 숫자가 아닌 값이 있습니다: {parts}") from exc
    return _validate_rectangular(data)


def load_matrix_from_file(path: str) -> Matrix:
    """
    파일 확장자를 기준으로 CSV/text 판별 후 행렬을 읽는다.
    """
    ext = Path(path).suffix.lower()
    if ext == ".csv":
        return read_matrix_from_csv(path)
    return read_matrix_from_text(path)


def write_matrix_to_csv(matrix: Matrix, path: str) -> None:
    """
    주어진 행렬을 CSV 파일로 저장한다.
    """
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for row in matrix:
            writer.writerow(row)


def write_matrices_with_labels(path: str, matrices: List[Tuple[str, Optional[Matrix]]]) -> None:
    """
    (제목, 행렬) 목록을 CSV에 순서대로 기록한다. 행렬이 None이면 해당 섹션에 메시지를 남긴다.
    """
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for label, matrix in matrices:
            writer.writerow([label])
            if matrix is None:
                writer.writerow(["결과 없음"])
            else:
                for row in matrix:
                    writer.writerow(row)
            writer.writerow([])


# -----------------------------
# 시각화 유틸리티
# -----------------------------


def _resolve_font_family(preferred: Optional[List[str]] = None) -> str:
    """
    시스템에 설치된 폰트 중 사용 가능한 한글 폰트를 우선순위대로 선택한다.
    """
    if preferred is None:
        preferred = [
            "NanumGothic",
            "AppleGothic",
            "Malgun Gothic",
            "Noto Sans CJK KR",
            "Noto Sans KR",
            "Arial Unicode MS",
            "UnDotum",
            "UnBatang",
            "DejaVu Sans",
        ]
    try:
        from matplotlib import font_manager
    except ImportError:
        return "DejaVu Sans"

    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in available:
            return name
    return "DejaVu Sans"


def _sanitize_label(label: str) -> str:
    """
    파일 이름에 사용할 수 있도록 간단히 치환한다.
    """
    safe = label.replace(" ", "_").replace("/", "_")
    safe = safe.replace("(", "").replace(")", "")
    safe = safe.replace("-", "_").replace("×", "x")
    return safe


def save_heatmap(matrix: Matrix, title: str, filename: str) -> None:
    """
    matplotlib을 이용해 행렬 히트맵 이미지를 저장한다.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib 모듈을 사용할 수 없어 시각화를 진행할 수 없습니다.") from exc

    font_family = _resolve_font_family()
    plt.rcParams["font.family"] = font_family
    plt.rcParams["font.sans-serif"] = [font_family, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots()
    cax = ax.imshow(matrix, cmap="viridis", aspect="auto")
    ax.set_title(title)
    fig.colorbar(cax, ax=ax, shrink=0.8)
    ax.set_xlabel("열")
    ax.set_ylabel("행")
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def prompt_visualize_results(result: dict) -> None:
    """
    행렬 히트맵 저장 여부를 사용자에게 묻는다.
    """
    try:
        choice = input("행렬 히트맵 이미지를 저장하시겠습니까? (y/n): ").strip().lower()
    except EOFError:
        choice = "n"
    if choice != "y":
        return
    try:
        base = input("저장할 파일 이름 접두사를 입력하세요 (예: output/matrix): ").strip()
    except EOFError:
        print("입력이 취소되었습니다. 시각화를 종료합니다.")
        return
    if not base:
        base = "matrix"

    matrices = [
        ("원본 행렬", result.get("matrix")),
        ("역행렬 (행렬식 기반)", result.get("inverse_determinant")),
        ("역행렬 (가우스-조던)", result.get("inverse_gauss_jordan")),
    ]
    saved_any = False
    for label, matrix in matrices:
        if matrix is None:
            continue
        filename = f"{base}_{_sanitize_label(label)}.png"
        try:
            save_heatmap(matrix, label, filename)
            print(f"히트맵을 저장했습니다: {filename}")
            saved_any = True
        except RuntimeError as exc:
            print(exc)
            return
        except OSError as exc:
            print(f"이미지를 저장하는 중 오류가 발생했습니다: {exc}")
            return
    if not saved_any:
        print("저장할 행렬이 없어 시각화를 건너뜁니다.")


# -----------------------------
# 행렬식·여인수를 이용한 역행렬 계산
# -----------------------------

def determinant(A: Matrix, eps: float = 1e-12) -> float:
    """
    부분 피벗팅이 포함된 가우스 소거법으로 행렬식을 계산한다.
    특이 행렬(또는 거의 특이한 행렬)은 0.0을 반환한다.
    """
    if not is_square(A):
        raise ValueError("determinant: 정방행렬만 가능합니다.")
    n = len(A)
    if n == 0:
        return 1.0
    if n == 1:
        return float(A[0][0])

    M = copy_matrix(A)
    sign = 1.0
    for col in range(n):
        # 부분 피벗팅
        pivot_row = col
        max_abs = abs(M[pivot_row][col])
        for r in range(col + 1, n):
            v = abs(M[r][col])
            if v > max_abs:
                max_abs = v
                pivot_row = r
        if max_abs < eps:
            return 0.0
        if pivot_row != col:
            M[col], M[pivot_row] = M[pivot_row], M[col]
            sign *= -1.0

        pivot = M[col][col]
        for r in range(col + 1, n):
            factor = M[r][col] / pivot
            # 피벗 아래 원소를 제거
            for c in range(col + 1, n):
                M[r][c] -= factor * M[col][c]
            M[r][col] = 0.0

    det = sign
    for i in range(n):
        det *= M[i][i]
    return float(det)


def minor_matrix(A: Matrix, row: int, col: int) -> Matrix:
    return [
        [A[i][j] for j in range(len(A)) if j != col]
        for i in range(len(A)) if i != row
    ]


def cofactor_matrix(A: Matrix) -> Matrix:
    if not is_square(A):
        raise ValueError("cofactor_matrix: 정방행렬만 가능합니다.")
    n = len(A)
    if n == 1:
        return [[1.0]]
    C = zeros(n, n)
    for i in range(n):
        for j in range(n):
            Mij = minor_matrix(A, i, j)
            C[i][j] = ((-1.0) ** (i + j)) * determinant(Mij)
    return C


def adjugate_matrix(A: Matrix) -> Matrix:
    C = cofactor_matrix(A)
    n = len(C)
    # 여인수 행렬 전치
    Adj = zeros(n, n)
    for i in range(n):
        for j in range(n):
            Adj[j][i] = C[i][j]
    return Adj


def inverse_determinant(A: Matrix, eps: float = 1e-12) -> Optional[Matrix]:
    if not is_square(A):
        raise ValueError("inverse_determinant: 정방행렬만 가능합니다.")
    detA = determinant(A, eps=eps)
    if abs(detA) < eps:
        print("행렬식이 0에 가깝습니다. 역행렬이 존재하지 않습니다.")
        return None
    Adj = adjugate_matrix(A)
    n = len(A)
    Inv = zeros(n, n)
    inv_det = 1.0 / detA
    for i in range(n):
        for j in range(n):
            Inv[i][j] = Adj[i][j] * inv_det
    return Inv


# -----------------------------
# 부분 피벗팅을 포함한 가우스-조던 역행렬 계산
# -----------------------------

def inverse_gauss_jordan(A: Matrix, eps: float = 1e-12) -> Matrix:
    if not is_square(A):
        raise ValueError("inverse_gauss_jordan: 정방행렬만 가능합니다.")
    n = len(A)
    L = copy_matrix(A)
    R = identity_matrix(n)

    for col in range(n):
        # 부분 피벗팅
        pivot_row = col
        max_abs = abs(L[pivot_row][col])
        for r in range(col + 1, n):
            v = abs(L[r][col])
            if v > max_abs:
                max_abs = v
                pivot_row = r
        if max_abs < eps:
            raise ValueError("역행렬이 존재하지 않습니다 (singular matrix).")
        if pivot_row != col:
            L[col], L[pivot_row] = L[pivot_row], L[col]
            R[col], R[pivot_row] = R[pivot_row], R[col]

        # 피벗 행을 1로 정규화
        pivot = L[col][col]
        inv_pivot = 1.0 / pivot
        for j in range(n):
            L[col][j] *= inv_pivot
            R[col][j] *= inv_pivot

        # 다른 행의 피벗 열 제거
        for r in range(n):
            if r == col:
                continue
            factor = L[r][col]
            if factor == 0:
                continue
            for j in range(n):
                L[r][j] -= factor * L[col][j]
                R[r][j] -= factor * R[col][j]

    return R


# -----------------------------
# 결과 비교 및 역행렬 검증
# -----------------------------

def compare_matrices(A: Optional[Matrix], B: Optional[Matrix], tol: float = 1e-10) -> bool:
    return matrix_equal(A, B, tol=tol)


def verify_inverse(A: Matrix, Ainv: Optional[Matrix], tol: float = 1e-10) -> bool:
    if Ainv is None:
        return False
    n = len(A)
    I = identity_matrix(n)
    prod = multiply_matrices(A, Ainv)
    print_matrix(prod, "A × A⁻¹")
    return matrix_equal(prod, I, tol)


# -----------------------------
# 테스트 케이스 및 결과 처리
# -----------------------------

def run_test_case(
    A: Matrix,
    title: str,
    tol: float = 1e-10,
    *,
    return_data: bool = False,
) -> Optional[dict]:
    print_matrix(A, f"테스트 케이스: {title}")
    detA = determinant(A)
    print(f"det(A) = {detA:.6f}")

    inv1 = inverse_determinant(A)
    if inv1 is not None:
        print_matrix(inv1, "역행렬 (행렬식 기반)")
    else:
        print("역행렬 (행렬식 기반): 존재하지 않음")

    gauss_error: Optional[str] = None
    try:
        inv2 = inverse_gauss_jordan(A)
        print_matrix(inv2, "역행렬 (가우스-조던)")
    except ValueError as e:
        inv2 = None
        gauss_error = str(e)
        print(f"역행렬 (가우스-조던): {e}")

    if inv1 is not None and inv2 is not None:
        same = compare_matrices(inv1, inv2, tol)
        print("비교 결과:", "두 방법의 결과가 일치합니다." if same else "결과에 차이가 있습니다.")
        ok = verify_inverse(A, inv2, tol)
        print("검증:", "A×A⁻¹ = I (허용오차 내)" if ok else "A×A⁻¹ ≠ I")
    elif inv1 is None and inv2 is None:
        print("두 방법 모두 역행렬이 존재하지 않음을 보고했습니다.")
    else:
        print("한 방법만 역행렬을 계산했습니다. (수치/방법 차이)")
    print()

    if return_data:
        return {
            "matrix": A,
            "determinant": detA,
            "inverse_determinant": inv1,
            "inverse_gauss_jordan": inv2,
            "gauss_error": gauss_error,
        }
    return None

def prompt_save_results(result: dict) -> None:
    """
    계산된 행렬들을 CSV로 저장할지 사용자에게 묻는다.
    """
    try:
        choice = input("결과를 CSV 파일로 저장하시겠습니까? (y/n): ").strip().lower()
    except EOFError:
        choice = "n"
    if choice != "y":
        return
    try:
        path = input("저장할 CSV 파일 경로를 입력하세요: ").strip()
    except EOFError:
        print("입력이 취소되었습니다. 저장을 종료합니다.")
        return
    if not path:
        print("경로가 비어 있습니다. 저장을 취소합니다.")
        return
    try:
        write_matrices_with_labels(
            path,
            [
                ("원본 행렬", result["matrix"]),
                ("역행렬 (행렬식 기반)", result.get("inverse_determinant")),
                ("역행렬 (가우스-조던)", result.get("inverse_gauss_jordan")),
            ],
        )
        print(f"CSV로 저장했습니다: {path}")
    except OSError as exc:
        print(f"파일 저장 중 오류가 발생했습니다: {exc}")


def handle_matrix_workflow(A: Matrix, title: str, tol: float = 1e-10) -> None:
    """
    행렬을 분석하고 결과 저장 여부를 처리한다.
    """
    result = run_test_case(A, title, tol=tol, return_data=True)
    if result is None:
        return
    prompt_save_results(result)
    prompt_visualize_results(result)


# -----------------------------
# 메인 실행부
# -----------------------------

def main() -> None:
    print("역행렬 계산기 (행렬식 기반 vs. 가우스-조던)")
    print("-" * 50)
    # 시작 시 즉시 직접 입력 단계 실행
    A0 = input_matrix()
    if A0 is not None:
        handle_matrix_workflow(A0, "사용자 입력 행렬")

    while True:
        try:
            action = input(
                "추가 작업을 선택하세요 (1: 직접 입력, 2: 파일에서 읽기, 3: 종료): "
            ).strip()
        except EOFError:
            break
        if action == "1":
            A = input_matrix()
            if A is None:
                print("입력이 종료되었습니다.")
                continue
            handle_matrix_workflow(A, "사용자 입력 행렬")
        elif action == "2":
            path = input("읽을 파일 경로를 입력하세요: ").strip()
            if not path:
                print("경로가 비어 있습니다. 다시 시도하세요.")
                continue
            try:
                A = load_matrix_from_file(path)
            except (OSError, ValueError) as exc:
                print(f"파일을 읽는 중 오류가 발생했습니다: {exc}")
                continue
            handle_matrix_workflow(A, f"파일 입력 ({path})")
        elif action == "3":
            break
        else:
            print("알 수 없는 입력입니다. 1, 2, 3 중에서 선택하세요.")


if __name__ == "__main__":
    main() 
