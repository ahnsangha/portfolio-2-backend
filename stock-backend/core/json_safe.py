import math
from starlette.responses import JSONResponse

def sanitize(obj):
    """dict/list/tuple/numpy/pandas 재귀 순회하며 NaN/Inf -> None."""
    try:
        import numpy as np
        import pandas as pd
    except Exception:
        np = None; pd = None

    # 스칼라
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if np is not None and isinstance(obj, (np.floating,)):
        f = float(obj);  return f if math.isfinite(f) else None

    # 컨테이너
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(sanitize(v) for v in obj)

    # numpy
    if np is not None and isinstance(obj, np.ndarray):
        return sanitize(obj.tolist())

    # pandas
    if np is not None and pd is not None:
        try:
            if isinstance(obj, pd.DataFrame):
                df = obj.replace([np.inf, -np.inf], np.nan).where(pd.notnull(df), None)
                return sanitize(df.to_dict(orient="records"))
            if isinstance(obj, pd.Series):
                s = obj.replace([np.inf, -np.inf], np.nan)
                return sanitize([None if pd.isna(v) else v for v in s.tolist()])
        except Exception:
            return None

    return obj

class SafeJSONResponse(JSONResponse):
    def render(self, content) -> bytes:
        # 응답 직전에 무조건 정화
        return super().render(sanitize(content))