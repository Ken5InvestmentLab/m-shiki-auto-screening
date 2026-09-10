import independent_daily_lab_v2 as v2

# 全探索とは別に、ユーザー提供Pineのデフォルト設定を
# 日足へ忠実移植したExact 1本だけを高速で全銘柄検証する。
_exact = next(p for p in v2.PINE_PROFILES if p["id"] == "exact")
v2.PINE_PROFILES[:] = [_exact]

if __name__ == "__main__":
    v2.main()
