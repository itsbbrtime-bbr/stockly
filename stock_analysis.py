################################################################################################
# MARKET + TECHNICAL + FUNDAMENTAL INTEGRATED ANALYSIS ENGINE
#
# 핵심 철학
#
# 1. 시장의 방향을 먼저 판단한다.
# 2. 개별 종목/ETF의 장기 추세를 판단한다.
# 3. 시장 대비 상대강도(Relative Strength)를 확인한다.
# 4. 좋은 자산이라도 진입 타이밍을 따로 판단한다.
# 5. RSI 70, 볼린저 상단 돌파 등을 자동 매도 신호로 사용하지 않는다.
# 6. 여러 조건 중 하나만 만족해도 BUY/SELL이 발생하는 OR 구조를 사용하지 않는다.
# 7. Score 기반으로 STRONG BUY / BUY / HOLD / WAIT / REDUCE / RISK OFF를 판단한다.
# 8. ETF와 개별 기업의 펀더멘털 분석을 분리한다.
# 9. Fear & Greed는 매수/매도 버튼이 아닌 시장 심리 보조지표로 사용한다.
# 10. 가격 성과와 전략 성과를 혼동하지 않는다.
################################################################################################


# ==============================================================================================
# 1. IMPORT
# ==============================================================================================

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rc

from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo

import yfinance as yf
import requests
import re


# ==============================================================================================
# 2. FONT SETTING
# ==============================================================================================

try:
    rc("font", family="AppleGothic")
except Exception:
    pass

matplotlib.rcParams["axes.unicode_minus"] = False


# ==============================================================================================
# 3. CONFIGURATION
# ==============================================================================================

NY_TZ = ZoneInfo("America/New_York")
TODAY = datetime.now(NY_TZ).date()

# GitHub Actions 환경변수 우선, 로컬 실행 시 기본값 사용
ticker_symbol = os.getenv("TICKER", "SOXX")
# ticker_symbol = "QLD"
# ticker_symbol = "TQQQ"  
# ticker_symbol = "SMH"
# ticker_symbol = "SOXX"
# ticker_symbol = "SOXQ"
# ticker_symbol = "DRAM"

# 개별주
# ticker_symbol = "MSFT"
# ticker_symbol = "AMZN"
# ticker_symbol = "AAPL"
# ticker_symbol = "PL"
# ticker_symbol = "VRT"
# ticker_symbol = "ETN"
# ticker_symbol = "CEG"
# ticker_symbol = "GEV"
# ticker_symbol = "PWR"

# 시장 벤치마크
MARKET_BENCHMARK = os.getenv("BENCHMARK", "VOO")

# 상대강도 비교용
RS_BENCHMARK = MARKET_BENCHMARK

# 기술적 분석 기간
# MA200을 안정적으로 계산하고 장기 추세/성과를 보기 위해 3년 사용
ANALYSIS_YEARS = 3

START_DATE = TODAY - timedelta(days=365 * ANALYSIS_YEARS)
SAVE_CHARTS = os.getenv("SAVE_CHARTS", "true").lower() == "true"
CHART_DIR = os.getenv("CHART_DIR", "charts")
os.makedirs(CHART_DIR, exist_ok=True) if SAVE_CHARTS else None

# 환율
FX_TICKER = "USDKRW=X"

# Dollar Index
DXY_TICKER = "DX-Y.NYB"

# ETF 후보
ETF_SYMBOLS = {"QQQ", "QLD", "TQQQ", "SMH", "SOXX", "SOXQ", "DRAM", "SPY", "VOO", "VTI", "XLK", "XLE", "XLF", "XLV", "XLI", "IBB", "ARKK"}

# region Entire File
# ==============================================================================================
# 4. HELPER FUNCTIONS
# ==============================================================================================

def safe_float(value):
    """
    숫자를 안전하게 float로 변환.
    None / NaN / Inf는 None 반환.
    """
    try:
        if value is None:
            return None

        value = float(value)

        if np.isnan(value) or np.isinf(value):
            return None

        return value

    except Exception:
        return None


def latest(series):
    """
    Series의 마지막 값을 안전하게 반환.
    """
    if series is None or len(series) == 0:
        return None

    try:
        value = series.iloc[-1]
        return safe_float(value)

    except Exception:
        return None


def previous(series, periods=1):
    """
    현재 시점 이전 값을 반환.
    """
    try:
        return safe_float(series.iloc[-1 - periods])
    except Exception:
        return None


def pct_change_safe(current, past):
    """
    안전한 수익률 계산.
    """
    current = safe_float(current)
    past = safe_float(past)

    if current is None or past is None or past == 0:
        return None

    return current / past - 1


def fmt_num(x, decimals=2):
    """
    숫자 출력 포맷.
    """
    x = safe_float(x)

    if x is None:
        return "N/A"

    return f"{x:,.{decimals}f}"


def fmt_pct(x, decimals=2):
    """
    소수 비율을 %로 출력.
    """
    x = safe_float(x)

    if x is None:
        return "N/A"

    return f"{x * 100:.{decimals}f}%"


def print_section(title):
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)


def save_or_close_chart(filename):
    """GitHub Actions에서는 PNG로 저장하고 로컬에서도 화면을 띄우지 않는다."""
    if SAVE_CHARTS:
        plt.savefig(os.path.join(CHART_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close()


def is_etf(symbol, info=None):
    """
    ETF 여부 판단.

    1차: 사전 등록 ETF 목록
    2차: Yahoo quoteType
    """

    if symbol.upper() in ETF_SYMBOLS:
        return True

    try:
        quote_type = str(info.get("quoteType", "")).upper()

        if quote_type in ["ETF", "MUTUALFUND"]:
            return True

    except Exception:
        pass

    return False


# ==============================================================================================
# 5. DATA DOWNLOAD
# ==============================================================================================

def download_price_data(symbol, start_date, end_date):
    """
    yfinance 가격 데이터 다운로드.
    """

    df = yf.download(
        symbol,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False
    )

    if df is None or df.empty:
        raise ValueError(f"{symbol} 가격 데이터를 가져올 수 없습니다.")

    # MultiIndex 대응
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{symbol} 필수 가격 데이터 누락: {sorted(missing)}")

    df = df.copy()

    return df


# ==============================================================================================
# 6. FIND FINANCIAL ROW
# ==============================================================================================

def find_row_label(df, candidates):
    """
    yfinance 재무제표의 다양한 항목명을 유연하게 탐색.
    """

    if df is None or df.empty:
        return None

    labels = list(map(str, df.index))

    # 1. 정확 일치
    for candidate in candidates:
        for label in labels:

            if candidate.lower() == label.lower():
                return label

    # 2. 부분 문자열
    for candidate in candidates:
        for label in labels:

            if candidate.lower() in label.lower():
                return label

    # 3. 모든 토큰 포함
    for candidate in candidates:

        tokens = candidate.lower().split()

        for label in labels:

            lower_label = label.lower()

            if all(token in lower_label for token in tokens):
                return label

    return None


def series_from_df(df, candidates):
    """
    재무제표에서 시계열 Series 반환.
    최신 데이터가 앞에 있는 경우 날짜 기준 정렬.
    """

    label = find_row_label(df, candidates)

    if label is None:
        return None

    try:

        series = df.loc[label].astype(float)

        # 날짜 기준 오래된 순 -> 최신 순 정렬
        series = series.sort_index()

        return series.dropna()

    except Exception:
        return None


def latest_value_from_df(df, candidates):

    series = series_from_df(df, candidates)

    if series is None or series.empty:
        return None

    return safe_float(series.iloc[-1])


def calculate_cagr(series):
    """
    재무 데이터 시계열 CAGR 계산.

    음수/0 시작값에서는 CAGR의 해석이 어려우므로 None.
    """

    if series is None:
        return None

    series = series.dropna()

    if len(series) < 2:
        return None

    start_value = safe_float(series.iloc[0])
    end_value = safe_float(series.iloc[-1])

    periods = len(series) - 1

    if (
        start_value is None
        or end_value is None
        or start_value <= 0
        or end_value <= 0
        or periods <= 0
    ):
        return None

    return (end_value / start_value) ** (1 / periods) - 1


def calculate_growth(series):
    """
    최근 1기간 성장률.
    """

    if series is None:
        return None

    series = series.dropna()

    if len(series) < 2:
        return None

    current = safe_float(series.iloc[-1])
    previous_value = safe_float(series.iloc[-2])

    return pct_change_safe(current, previous_value)


# ==============================================================================================
# 7. FUNDAMENTAL ANALYSIS
# ==============================================================================================

def calculate_fundamentals(symbol):

    ticker = yf.Ticker(symbol)

    try:
        info = ticker.info
    except Exception:
        info = {}

    asset_is_etf = is_etf(symbol, info)

    result = {
        "symbol": symbol,
        "is_etf": asset_is_etf,
        "info": info
    }

    # ------------------------------------------------------------------------------------------
    # ETF
    # ------------------------------------------------------------------------------------------

    if asset_is_etf:

        result["analysis_type"] = "ETF"

        result["name"] = info.get("longName") or info.get("shortName") or symbol

        result["category"] = info.get("category", "N/A")

        result["expense_ratio"] = safe_float(info.get("annualReportExpenseRatio"))

        result["total_assets"] = safe_float(info.get("totalAssets"))

        result["yield"] = safe_float(
            info.get("yield")
            or info.get("trailingAnnualDividendYield")
        )

        return result

    # ------------------------------------------------------------------------------------------
    # INDIVIDUAL STOCK
    # ------------------------------------------------------------------------------------------

    try:
        financials = ticker.financials
    except Exception:
        financials = pd.DataFrame()
    try:
        balance_sheet = ticker.balance_sheet
    except Exception:
        balance_sheet = pd.DataFrame()
    try:
        cashflow = ticker.cashflow
    except Exception:
        cashflow = pd.DataFrame()

    result["analysis_type"] = "STOCK"

    result["name"] = info.get("longName") or symbol

    result["sector"] = info.get("sector", "N/A")
    result["industry"] = info.get("industry", "N/A")

    result["trailing_pe"] = safe_float(info.get("trailingPE"))
    result["forward_pe"] = safe_float(info.get("forwardPE"))
    result["price_to_book"] = safe_float(info.get("priceToBook"))

    result["market_cap"] = safe_float(info.get("marketCap"))

    result["return_on_equity_info"] = safe_float(
        info.get("returnOnEquity")
    )

    # 후보 라벨
    revenue_candidates = [
        "Total Revenue",
        "Revenue",
        "Net Sales",
        "Sales"
    ]

    net_income_candidates = [
        "Net Income",
        "Net Income Common Stockholders",
        "Net Income Applicable To Common Shares"
    ]

    operating_income_candidates = [
        "Operating Income",
        "Operating Income or Loss",
        "Income From Operations"
    ]

    pretax_income_candidates = [
        "Pretax Income",
        "Income Before Tax"
    ]

    tax_candidates = [
        "Tax Provision",
        "Tax Effect Of Unusual Items"
    ]

    total_assets_candidates = [
        "Total Assets"
    ]

    total_liabilities_candidates = [
        "Total Liabilities",
        "Total Liab",
        "Total Liabilities Net Minority Interest"
    ]

    total_equity_candidates = [
        "Stockholders Equity",
        "Total Stockholder Equity",
        "Total Stockholders' Equity",
        "Total Shareholder Equity",
        "Total Equity Gross Minority Interest",
        "Total Equity"
    ]

    current_assets_candidates = [
        "Current Assets",
        "Total Current Assets"
    ]

    current_liabilities_candidates = [
        "Current Liabilities",
        "Total Current Liabilities"
    ]

    cash_candidates = [
        "Cash And Cash Equivalents",
        "Cash Cash Equivalents And Short Term Investments",
        "Cash"
    ]

    operating_cf_candidates = [
        "Operating Cash Flow",
        "Total Cash From Operating Activities",
        "Net Cash Provided by Operating Activities"
    ]

    capex_candidates = [
        "Capital Expenditure",
        "Capital Expenditures",
        "Purchase Of PPE"
    ]

    fcf_candidates = [
        "Free Cash Flow"
    ]

    rd_candidates = [
        "Research And Development",
        "Research Development",
        "Research & Development",
        "R&D"
    ]

    diluted_eps_candidates = [
        "Diluted EPS",
        "Diluted Average Shares"
    ]

    diluted_shares_candidates = [
        "Diluted Average Shares",
        "Diluted Average Shares Outstanding",
        "Basic Average Shares"
    ]

    # 시계열

    revenue_series = series_from_df(
        financials,
        revenue_candidates
    )

    net_income_series = series_from_df(
        financials,
        net_income_candidates
    )

    operating_income_series = series_from_df(
        financials,
        operating_income_candidates
    )

    operating_cf_series = series_from_df(
        cashflow,
        operating_cf_candidates
    )

    capex_series = series_from_df(
        cashflow,
        capex_candidates
    )

    fcf_series = series_from_df(
        cashflow,
        fcf_candidates
    )

    rd_series = series_from_df(
        financials,
        rd_candidates
    )

    diluted_eps_series = series_from_df(
        financials,
        diluted_eps_candidates
    )

    diluted_shares_series = series_from_df(
        financials,
        diluted_shares_candidates
    )

    # FCF 직접 데이터가 없으면 OCF + CapEx
    if fcf_series is None:

        if (
            operating_cf_series is not None
            and capex_series is not None
        ):

            try:
                fcf_series = operating_cf_series + capex_series
            except Exception:
                fcf_series = None

    # 최신 값

    revenue = latest(revenue_series)
    net_income = latest(net_income_series)
    operating_income = latest(operating_income_series)

    operating_cf = latest(operating_cf_series)
    capex = latest(capex_series)
    fcf = latest(fcf_series)

    rd = latest(rd_series)

    total_assets = latest_value_from_df(
        balance_sheet,
        total_assets_candidates
    )

    total_liabilities = latest_value_from_df(
        balance_sheet,
        total_liabilities_candidates
    )

    total_equity = latest_value_from_df(
        balance_sheet,
        total_equity_candidates
    )

    current_assets = latest_value_from_df(
        balance_sheet,
        current_assets_candidates
    )

    current_liabilities = latest_value_from_df(
        balance_sheet,
        current_liabilities_candidates
    )

    cash = latest_value_from_df(
        balance_sheet,
        cash_candidates
    )

    # 기본 수익성

    roe = None

    if (
        net_income is not None
        and total_equity is not None
        and total_equity != 0
    ):
        roe = net_income / total_equity

    net_margin = None

    if revenue is not None and revenue != 0 and net_income is not None:
        net_margin = net_income / revenue

    operating_margin = None

    if (
        revenue is not None
        and revenue != 0
        and operating_income is not None
    ):
        operating_margin = operating_income / revenue

    # 재무 안정성

    debt_ratio = None

    if (
        total_liabilities is not None
        and total_assets is not None
        and total_assets != 0
    ):
        debt_ratio = total_liabilities / total_assets

    current_ratio = None

    if (
        current_assets is not None
        and current_liabilities is not None
        and current_liabilities != 0
    ):
        current_ratio = current_assets / current_liabilities

    cash_ratio = None

    if (
        cash is not None
        and current_liabilities is not None
        and current_liabilities != 0
    ):
        cash_ratio = cash / current_liabilities

    # R&D는 GOOD/BAD 기준이 아니라 추세 참고용

    rd_ratio = None

    if (
        rd is not None
        and revenue is not None
        and revenue != 0
    ):
        rd_ratio = rd / revenue

    # 성장성

    revenue_cagr = calculate_cagr(revenue_series)

    revenue_growth = calculate_growth(revenue_series)

    net_income_growth = calculate_growth(
        net_income_series
    )

    fcf_growth = calculate_growth(
        fcf_series
    )

    rd_growth = calculate_growth(
        rd_series
    )

    # EPS

    eps_cagr = calculate_cagr(
        diluted_eps_series
    )

    eps_growth = calculate_growth(
        diluted_eps_series
    )

    # EPS가 없으면 순이익 성장률을 EPS로 대체하지 않는다.
    # 두 데이터는 별개로 유지한다.

    # 주식 수 변화

    share_count_change = calculate_growth(
        diluted_shares_series
    )

    # ROIC 근사 계산
    #
    # NOPAT = Operating Income × (1 - Tax Rate)
    # Invested Capital ≈ Equity + Interest-bearing Debt - Cash
    #
    # yfinance에서 debt 항목이 누락될 수 있으므로
    # 데이터가 충분할 때만 계산.

    long_term_debt = latest_value_from_df(
        balance_sheet,
        [
            "Long Term Debt",
            "Long Term Debt And Capital Lease Obligation"
        ]
    )

    current_debt = latest_value_from_df(
        balance_sheet,
        [
            "Current Debt",
            "Current Debt And Capital Lease Obligation"
        ]
    )

    pretax_income = latest_value_from_df(
        financials,
        pretax_income_candidates
    )

    tax_provision = latest_value_from_df(
        financials,
        tax_candidates
    )

    effective_tax_rate = None

    if (
        pretax_income is not None
        and pretax_income > 0
        and tax_provision is not None
    ):

        effective_tax_rate = abs(tax_provision / pretax_income)

    roic = None

    if (
        operating_income is not None
        and total_equity is not None
    ):

        total_debt = (
            (long_term_debt or 0)
            +
            (current_debt or 0)
        )

        invested_capital = (
            total_equity
            +
            total_debt
            -
            (cash or 0)
        )

        if invested_capital > 0:

            tax_rate = (
                effective_tax_rate
                if effective_tax_rate is not None
                else 0.21
            )

            nopat = operating_income * (1 - tax_rate)

            roic = nopat / invested_capital

    # FCF Margin

    fcf_margin = None

    if (
        fcf is not None
        and revenue is not None
        and revenue != 0
    ):

        fcf_margin = fcf / revenue

    result.update({

        "revenue": revenue,
        "net_income": net_income,
        "operating_income": operating_income,

        "operating_cf": operating_cf,
        "capex": capex,
        "fcf": fcf,

        "roe": roe,
        "roic": roic,

        "net_margin": net_margin,
        "operating_margin": operating_margin,
        "fcf_margin": fcf_margin,

        "debt_ratio": debt_ratio,
        "current_ratio": current_ratio,
        "cash_ratio": cash_ratio,

        "rd_ratio": rd_ratio,
        "rd_growth": rd_growth,

        "revenue_cagr": revenue_cagr,
        "revenue_growth": revenue_growth,

        "net_income_growth": net_income_growth,

        "eps_growth": eps_growth,
        "eps_cagr": eps_cagr,

        "share_count_change": share_count_change,

        "fcf_growth": fcf_growth
    })

    return result


# ==============================================================================================
# 8. FUNDAMENTAL SCORE
# ==============================================================================================

def calculate_fundamental_score(fundamental):

    if fundamental["is_etf"]:

        return {
            "score": None,
            "max_score": None,
            "grade": "ETF - 펀더멘털 점수 미적용",
            "reasons": [
                "ETF는 개별 기업과 동일한 PER/ROE/FCF 기준으로 평가하지 않습니다.",
                "ETF 평가는 기술적 추세, 상대강도, 변동성, 구성 종목 분석이 중심입니다."
            ]
        }

    score = 0
    reasons = []

    # ------------------------------------------------------------------------------------------
    # 성장
    # ------------------------------------------------------------------------------------------

    revenue_growth = fundamental.get("revenue_growth")

    if revenue_growth is not None:

        if revenue_growth > 0.15:
            score += 2
            reasons.append("매출 성장 강함")

        elif revenue_growth > 0:
            score += 1
            reasons.append("매출 성장 유지")

        else:
            score -= 1
            reasons.append("매출 성장 둔화/감소")

    eps_growth = fundamental.get("eps_growth")

    if eps_growth is not None:

        if eps_growth > 0.15:
            score += 2
            reasons.append("EPS 성장 강함")

        elif eps_growth > 0:
            score += 1
            reasons.append("EPS 성장 유지")

        else:
            score -= 1
            reasons.append("EPS 감소")

    # ------------------------------------------------------------------------------------------
    # 수익성
    # ------------------------------------------------------------------------------------------

    operating_margin = fundamental.get(
        "operating_margin"
    )

    if operating_margin is not None:

        if operating_margin > 0.20:
            score += 2
            reasons.append("높은 영업이익률")

        elif operating_margin > 0.10:
            score += 1
            reasons.append("양호한 영업이익률")

    fcf = fundamental.get("fcf")

    if fcf is not None:

        if fcf > 0:
            score += 1
            reasons.append("FCF 플러스")

        else:
            score -= 1
            reasons.append("FCF 마이너스")

    # ------------------------------------------------------------------------------------------
    # 자본 효율성
    # ------------------------------------------------------------------------------------------

    roic = fundamental.get("roic")

    if roic is not None:

        if roic > 0.20:
            score += 2
            reasons.append("ROIC 매우 우수")

        elif roic > 0.10:
            score += 1
            reasons.append("ROIC 양호")

    # ------------------------------------------------------------------------------------------
    # 재무 안정성
    # ------------------------------------------------------------------------------------------

    debt_ratio = fundamental.get(
        "debt_ratio"
    )

    if debt_ratio is not None:

        if debt_ratio > 0.80:
            score -= 1
            reasons.append("부채 수준 주의")

    # ------------------------------------------------------------------------------------------
    # 주식 수 변화
    # ------------------------------------------------------------------------------------------

    share_change = fundamental.get(
        "share_count_change"
    )

    if share_change is not None:

        if share_change < -0.02:
            score += 1
            reasons.append("주식 수 감소")

        elif share_change > 0.05:
            score -= 1
            reasons.append("주식 수 증가/희석 가능성")

    # ------------------------------------------------------------------------------------------
    # 밸류에이션
    #
    # 절대 PER 25 / 35 같은 기준은 사용하지 않는다.
    # 산업별 비교가 필요하기 때문에 참고 데이터로만 사용.
    # ------------------------------------------------------------------------------------------

    trailing_pe = fundamental.get("trailing_pe")

    if trailing_pe is not None:

        reasons.append(
            f"Trailing PER 참고값: {trailing_pe:.2f}"
        )

    if score >= 6:
        grade = "STRONG"

    elif score >= 3:
        grade = "GOOD"

    elif score >= 0:
        grade = "NEUTRAL"

    else:
        grade = "WEAK"

    return {

        "score": score,
        "max_score": 10,
        "grade": grade,
        "reasons": reasons
    }


# ==============================================================================================
# 9. TECHNICAL INDICATORS
# ==============================================================================================

def calculate_indicators(df):

    df = df.copy()

    # ------------------------------------------------------------------------------------------
    # Moving Average
    # ------------------------------------------------------------------------------------------

    df["MA20"] = df["Close"].rolling(20).mean()

    df["MA50"] = df["Close"].rolling(50).mean()

    df["MA100"] = df["Close"].rolling(100).mean()

    df["MA200"] = df["Close"].rolling(200).mean()

    # MA200 기울기
    df["MA200_Slope"] = df["MA200"].pct_change(20)
    df["Return_5D"] = df["Close"].pct_change(5)

    # ------------------------------------------------------------------------------------------
    # Bollinger Band
    # ------------------------------------------------------------------------------------------

    df["BB_Mid"] = df["MA20"]

    df["BB_Std"] = df["Close"].rolling(20).std()

    df["BB_Upper"] = (
        df["BB_Mid"]
        +
        2 * df["BB_Std"]
    )

    df["BB_Lower"] = (
        df["BB_Mid"]
        -
        2 * df["BB_Std"]
    )

    # Bollinger Width
    df["BB_Width"] = (
        (df["BB_Upper"] - df["BB_Lower"])
        /
        df["BB_Mid"]
    )

    df["BB_Width_MA50"] = (
        df["BB_Width"]
        .rolling(50)
        .mean()
    )

    # Bollinger %B: 하단 0 / 상단 1을 기준으로 현재 위치 판단
    df["BB_PctB"] = (df["Close"] - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"]).replace(0, np.nan)

    # 변동성 수축
    df["BB_Squeeze"] = (
        df["BB_Width"]
        <
        df["BB_Width_MA50"] * 0.75
    )

    # ------------------------------------------------------------------------------------------
    # RSI - Wilder Smoothing
    # ------------------------------------------------------------------------------------------

    delta = df["Close"].diff()

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["RSI"] = (
        100
        -
        (100 / (1 + rs))
    )

    # ------------------------------------------------------------------------------------------
    # MACD
    # ------------------------------------------------------------------------------------------

    df["EMA12"] = (
        df["Close"]
        .ewm(span=12, adjust=False)
        .mean()
    )

    df["EMA26"] = (
        df["Close"]
        .ewm(span=26, adjust=False)
        .mean()
    )

    df["MACD"] = (
        df["EMA12"]
        -
        df["EMA26"]
    )

    df["MACD_Signal"] = (
        df["MACD"]
        .ewm(span=9, adjust=False)
        .mean()
    )

    df["MACD_Hist"] = (
        df["MACD"]
        -
        df["MACD_Signal"]
    )

    df["MACD_Hist_Rising"] = (
        df["MACD_Hist"]
        >
        df["MACD_Hist"].shift(1)
    )

    # ------------------------------------------------------------------------------------------
    # Volume
    # ------------------------------------------------------------------------------------------

    df["Volume_MA20"] = (
        df["Volume"]
        .rolling(20)
        .mean()
    )

    df["Volume_Ratio"] = (
        df["Volume"]
        /
        df["Volume_MA20"]
    )

    # ------------------------------------------------------------------------------------------
    # OBV
    # ------------------------------------------------------------------------------------------

    direction = np.sign(
        df["Close"].diff()
    ).fillna(0)

    df["OBV"] = (
        direction
        *
        df["Volume"]
    ).cumsum()

    df["OBV_MA20"] = (
        df["OBV"]
        .rolling(20)
        .mean()
    )

    df["OBV_Strength"] = (
        df["OBV"]
        >
        df["OBV_MA20"]
    )

    # ------------------------------------------------------------------------------------------
    # ATR
    # ------------------------------------------------------------------------------------------

    high_low = (
        df["High"]
        -
        df["Low"]
    )

    high_close = (
        df["High"]
        -
        df["Close"].shift()
    ).abs()

    low_close = (
        df["Low"]
        -
        df["Close"].shift()
    ).abs()

    true_range = pd.concat(
        [
            high_low,
            high_close,
            low_close
        ],
        axis=1
    ).max(axis=1)

    df["ATR"] = (
        true_range
        .ewm(alpha=1 / 14, adjust=False)
        .mean()
    )

    df["ATR_Pct"] = (
        df["ATR"]
        /
        df["Close"]
    )

    # ------------------------------------------------------------------------------------------
    # CCI
    # 정확한 Rolling Mean Deviation 방식
    # ------------------------------------------------------------------------------------------

    typical_price = (
        df["High"]
        +
        df["Low"]
        +
        df["Close"]
    ) / 3

    tp_sma = (
        typical_price
        .rolling(20)
        .mean()
    )

    mean_deviation = (
        typical_price
        .rolling(20)
        .apply(
            lambda x: np.mean(
                np.abs(
                    x - np.mean(x)
                )
            ),
            raw=True
        )
    )

    df["CCI"] = (
        (typical_price - tp_sma)
        /
        (0.015 * mean_deviation)
    )

    # ------------------------------------------------------------------------------------------
    # Stochastic
    # 보조 확인용으로만 사용
    # ------------------------------------------------------------------------------------------

    lowest_low = (
        df["Low"]
        .rolling(14)
        .min()
    )

    highest_high = (
        df["High"]
        .rolling(14)
        .max()
    )

    denominator = (
        highest_high
        -
        lowest_low
    )

    df["Stoch_K"] = (
        (
            df["Close"]
            -
            lowest_low
        )
        /
        denominator.replace(0, np.nan)
        * 100
    )

    df["Stoch_D"] = (
        df["Stoch_K"]
        .rolling(3)
        .mean()
    )

    return df


# ==============================================================================================
# 10. RELATIVE STRENGTH
# ==============================================================================================

def calculate_relative_strength(
    asset_df,
    benchmark_df
):

    df = asset_df.copy()

    asset = asset_df["Close"]

    benchmark = benchmark_df["Close"]

    # 공통 날짜 정렬
    aligned = pd.concat(
        [
            asset.rename("Asset"),
            benchmark.rename("Benchmark")
        ],
        axis=1
    ).dropna()

    # 정규화 가격
    asset_normalized = (
        aligned["Asset"]
        /
        aligned["Asset"].iloc[0]
    )

    benchmark_normalized = (
        aligned["Benchmark"]
        /
        aligned["Benchmark"].iloc[0]
    )

    relative_strength = (
        asset_normalized
        /
        benchmark_normalized
    )

    rs_df = pd.DataFrame(
        index=aligned.index
    )

    rs_df["Relative_Strength"] = (
        relative_strength
    )

    rs_df["RS_MA20"] = (
        rs_df["Relative_Strength"]
        .rolling(20)
        .mean()
    )

    rs_df["RS_MA60"] = (
        rs_df["Relative_Strength"]
        .rolling(60)
        .mean()
    )

    rs_df["RS_Rising_20"] = (
        rs_df["Relative_Strength"]
        >
        rs_df["Relative_Strength"].shift(20)
    )

    rs_df["RS_Above_MA20"] = (
        rs_df["Relative_Strength"]
        >
        rs_df["RS_MA20"]
    )

    # 원본 DF에 결합

    df = df.join(rs_df)

    return df


# ==============================================================================================
# 11. MARKET REGIME
# ==============================================================================================

def determine_market_regime(market_df):

    last = market_df.iloc[-1]

    close = safe_float(last["Close"])

    ma50 = safe_float(last["MA50"])
    ma200 = safe_float(last["MA200"])

    ma200_slope = safe_float(
        last["MA200_Slope"]
    )

    if (
        close is None
        or ma50 is None
        or ma200 is None
    ):
        return {
            "regime": "UNKNOWN",
            "score": 0,
            "reason": "시장 데이터 부족"
        }

    # Strong Bull

    if (
        close > ma200
        and ma50 > ma200
        and ma200_slope is not None
        and ma200_slope > 0
    ):

        return {
            "regime": "STRONG_BULL",
            "score": 2,
            "reason": "가격 > MA200, MA50 > MA200, MA200 상승"
        }

    # Bull

    if (
        close > ma200
        and ma50 > ma200
    ):

        return {
            "regime": "BULL",
            "score": 1,
            "reason": "장기 상승 추세 유지"
        }

    # Bear

    if (
        close < ma200
        and ma50 < ma200
    ):

        return {
            "regime": "BEAR",
            "score": -2,
            "reason": "가격 < MA200, MA50 < MA200"
        }

    # Neutral

    return {
        "regime": "NEUTRAL",
        "score": 0,
        "reason": "추세 전환/횡보 구간"
    }


# ==============================================================================================
# 12. ASSET TREND SCORE
# ==============================================================================================

def calculate_technical_score(
    df,
    market_regime
):

    last = df.iloc[-1]

    score = 0

    reasons = []

    close = safe_float(last["Close"])

    ma20 = safe_float(last["MA20"])
    ma50 = safe_float(last["MA50"])
    ma200 = safe_float(last["MA200"])

    ma200_slope = safe_float(
        last["MA200_Slope"]
    )

    rsi = safe_float(last["RSI"])

    volume_ratio = safe_float(
        last["Volume_Ratio"]
    )

    atr_pct = safe_float(
        last["ATR_Pct"]
    )

    # ------------------------------------------------------------------------------------------
    # TREND
    # ------------------------------------------------------------------------------------------

    if close is not None and ma200 is not None:

        if close > ma200:

            score += 2

            reasons.append(
                "가격이 MA200 위"
            )

        else:

            score -= 2

            reasons.append(
                "가격이 MA200 아래"
            )

    if ma50 is not None and ma200 is not None:

        if ma50 > ma200:

            score += 2

            reasons.append(
                "MA50 > MA200"
            )

        else:

            score -= 1

            reasons.append(
                "MA50 < MA200"
            )

    if (
        ma200_slope is not None
    ):

        if ma200_slope > 0:

            score += 2

            reasons.append(
                "MA200 상승"
            )

        else:

            score -= 1

            reasons.append(
                "MA200 하락"
            )

    if (
        ma20 is not None
        and ma50 is not None
    ):

        if ma20 > ma50:

            score += 1

            reasons.append(
                "MA20 > MA50"
            )

    # ------------------------------------------------------------------------------------------
    # RELATIVE STRENGTH
    # ------------------------------------------------------------------------------------------

    rs_rising = bool(
        last.get(
            "RS_Rising_20",
            False
        )
    )

    rs_above_ma = bool(
        last.get(
            "RS_Above_MA20",
            False
        )
    )

    if rs_rising:

        score += 2

        reasons.append(
            "시장 대비 상대강도 상승"
        )

    else:

        score -= 1

        reasons.append(
            "시장 대비 상대강도 약화"
        )

    if rs_above_ma:

        score += 1

        reasons.append(
            "상대강도 MA20 위"
        )

    # ------------------------------------------------------------------------------------------
    # MOMENTUM
    # ------------------------------------------------------------------------------------------

    if rsi is not None:

        # 강한 추세에서 RSI 70 이상은 매도 신호가 아님

        if 50 <= rsi <= 75:

            score += 1

            reasons.append(
                "건강한 상승 모멘텀 RSI"
            )

        elif 40 <= rsi < 50:

            score += 2

            reasons.append(
                "상승 추세 내 조정 가능성"
            )

        elif rsi > 80:

            reasons.append(
                "RSI 극단적 과열 - 신규 매수 속도 조절"
            )

        elif rsi < 30:

            reasons.append(
                "RSI 과매도 - 추세 확인 필요"
            )

    macd = safe_float(last["MACD"])

    macd_signal = safe_float(
        last["MACD_Signal"]
    )

    macd_hist_rising = bool(
        last.get(
            "MACD_Hist_Rising",
            False
        )
    )

    if (
        macd is not None
        and macd_signal is not None
    ):

        if macd > macd_signal:

            score += 1

            reasons.append(
                "MACD 상승 우위"
            )

    if macd_hist_rising:

        score += 1

        reasons.append(
            "MACD 히스토그램 개선"
        )

    # ------------------------------------------------------------------------------------------
    # ENTRY - Pullback
    # ------------------------------------------------------------------------------------------

    near_ma20 = False

    near_ma50 = False

    if close is not None and ma20 is not None:

        near_ma20 = (
            abs(close - ma20)
            /
            ma20
            <= 0.03
        )

    if close is not None and ma50 is not None:

        near_ma50 = (
            abs(close - ma50)
            /
            ma50
            <= 0.05
        )

    strong_trend = (
        close is not None
        and ma20 is not None
        and ma50 is not None
        and ma200 is not None
        and close > ma200
        and ma50 > ma200
    )

    if strong_trend:

        if near_ma20 or near_ma50:

            score += 2

            reasons.append(
                "상승 추세 내 이동평균선 조정 구간"
            )

    # ------------------------------------------------------------------------------------------
    # BREAKOUT + VOLUME
    # ------------------------------------------------------------------------------------------

    recent_high = (
        df["High"]
        .rolling(20)
        .max()
        .shift(1)
    )

    breakout = False

    try:

        breakout = (
            close
            >
            recent_high.iloc[-1]
        )

    except Exception:
        pass

    if (
        breakout
        and volume_ratio is not None
        and volume_ratio >= 1.2
    ):

        score += 2

        reasons.append(
            "거래량 확인된 20일 돌파"
        )

    # ------------------------------------------------------------------------------------------
    # BOLLINGER SQUEEZE
    # ------------------------------------------------------------------------------------------

    if bool(last.get("BB_Squeeze", False)):

        reasons.append(
            "볼린저 밴드 수축 - 변동성 확대 가능성"
        )

    # ------------------------------------------------------------------------------------------
    # OBV
    # ------------------------------------------------------------------------------------------

    if bool(last.get("OBV_Strength", False)):

        score += 1

        reasons.append(
            "OBV가 평균 위 - 거래량 흐름 양호"
        )

    # ------------------------------------------------------------------------------------------
    # VOLATILITY RISK
    # ------------------------------------------------------------------------------------------

    if atr_pct is not None:

        if atr_pct > 0.08:

            score -= 2

            reasons.append(
                "ATR 변동성 매우 높음"
            )

        elif atr_pct > 0.05:

            score -= 1

            reasons.append(
                "ATR 변동성 높음"
            )

    # ------------------------------------------------------------------------------------------
    # MARKET REGIME
    # ------------------------------------------------------------------------------------------

    if market_regime["regime"] == "STRONG_BULL":

        score += 1

        reasons.append(
            "시장 환경 강한 상승 추세"
        )

    elif market_regime["regime"] == "BEAR":

        score -= 2

        reasons.append(
            "시장 환경 하락 추세"
        )

    return {

        "score": score,

        "reasons": reasons,

        "strong_trend": strong_trend,

        "near_ma20": near_ma20,

        "near_ma50": near_ma50,

        "breakout": breakout,

        "rsi": rsi,

        "atr_pct": atr_pct,

        "volume_ratio": volume_ratio
    }



# ==============================================================================================
# 12B. LONG-TERM INVESTING DECISION ENGINE
# ==============================================================================================

def calculate_trend_score(df, market_regime):
    last = df.iloc[-1]
    close = safe_float(last.get("Close")); ma200 = safe_float(last.get("MA200")); ma50 = safe_float(last.get("MA50")); slope = safe_float(last.get("MA200_Slope")); rs = safe_float(last.get("Relative_Strength")); rs_ma = safe_float(last.get("RS_MA20"))
    score, reasons = 0, []
    if close is not None and ma200 is not None and close > ma200: score += 1; reasons.append("가격이 MA200 위")
    if ma50 is not None and ma200 is not None and ma50 > ma200: score += 1; reasons.append("MA50 > MA200")
    if slope is not None and slope > 0: score += 1; reasons.append("MA200 상승")
    if rs is not None and rs_ma is not None and rs > rs_ma: score += 1; reasons.append("시장 대비 상대강도 우위")
    status = "STRONG" if score == 4 else "HEALTHY" if score == 3 else "NEUTRAL" if score == 2 else "WEAK"
    return {"score":score,"status":status,"reasons":reasons}

def calculate_current_drawdown(df):
    close=df["Close"].dropna(); peak=close.cummax().iloc[-1] if len(close) else None
    return None if peak in (None,0) else safe_float(close.iloc[-1]/peak-1)

def calculate_accumulation_score(df, fng_value):
    last=df.iloc[-1]; score=0; reasons=[]; dd=calculate_current_drawdown(df)
    rsi=safe_float(last.get("RSI")); bb=safe_float(last.get("BB_PctB")); close=safe_float(last.get("Close")); ma200=safe_float(last.get("MA200"))
    dist=None if close is None or ma200 in (None,0) else close/ma200-1
    if dd is not None:
        if dd <= -0.30: score+=5; reasons.append("고점 대비 -30% 이상 조정")
        elif dd <= -0.20: score+=4; reasons.append("고점 대비 -20% 이상 조정")
        elif dd <= -0.15: score+=3; reasons.append("고점 대비 -15% 이상 조정")
        elif dd <= -0.10: score+=2; reasons.append("고점 대비 -10% 이상 조정")
        elif dd <= -0.05: score+=1; reasons.append("고점 대비 -5% 이상 조정")
    if fng_value is not None:
        if fng_value < 15: score+=4; reasons.append("Fear & Greed 극단적 공포")
        elif fng_value < 25: score+=3; reasons.append("Fear & Greed 강한 공포")
        elif fng_value < 35: score+=2; reasons.append("Fear & Greed 공포")
        elif fng_value < 45: score+=1; reasons.append("Fear & Greed 약한 공포")
        elif fng_value >= 75: score-=2; reasons.append("Fear & Greed 극단적 탐욕 - 신규 매수 축소")
        elif fng_value >= 60: score-=1; reasons.append("Fear & Greed 탐욕 - 신규 매수 속도 조절")
    if rsi is not None:
        if rsi < 25: score+=3; reasons.append("RSI 극단적 과매도")
        elif rsi < 35: score+=2; reasons.append("RSI 과매도")
        elif rsi < 45: score+=1; reasons.append("RSI 조정 구간")
        elif rsi > 75: score-=2; reasons.append("RSI 극단적 과열")
        elif rsi > 65: score-=1; reasons.append("RSI 과열권")
    if bb is not None:
        if bb < 0: score+=2; reasons.append("볼린저 하단 이탈")
        elif bb < 0.2: score+=1; reasons.append("볼린저 하단권")
        elif bb > 1: score-=2; reasons.append("볼린저 상단 돌파 - 신규 매수 축소")
        elif bb > 0.8: score-=1; reasons.append("볼린저 상단권")
    if dist is not None:
        if dist < -0.25: score+=2; reasons.append("MA200 대비 큰 할인")
        elif dist < -0.10: score+=1; reasons.append("MA200 아래 조정")
        elif dist > 0.35: score-=2; reasons.append("MA200 대비 과도한 이격")
        elif dist > 0.20: score-=1; reasons.append("MA200 대비 높은 이격")
    return {"score":score,"reasons":reasons,"drawdown":dd,"rsi":rsi,"bb_pctb":bb,"ma200_distance":dist}

def calculate_risk_check(df, trend, fundamental_score, is_etf):
    last=df.iloc[-1]; close=safe_float(last.get("Close")); ma200=safe_float(last.get("MA200")); slope=safe_float(last.get("MA200_Slope")); vr=safe_float(last.get("Volume_Ratio")); ret5=safe_float(last.get("Return_5D")); dd=calculate_current_drawdown(df)
    flags=[]
    if dd is not None and dd <= -0.20 and vr is not None and vr >= 2.5: flags.append("큰 하락과 거래량 급증 - 투매/구조적 악화 확인 필요")
    if close is not None and ma200 is not None and close < ma200 and slope is not None and slope < 0: flags.append("가격과 MA200 장기 추세 모두 약화")
    fundamental_value = None
    if isinstance(fundamental_score, dict):
        fundamental_value = safe_float(fundamental_score.get("score"))
    else:
        fundamental_value = safe_float(fundamental_score)
    if (not is_etf) and fundamental_value is not None and fundamental_value <= 0:
        flags.append("펀더멘털 점수 약화 - 투자 논리 재점검")
    if len(flags)>=2: status="THESIS REVIEW"
    elif flags: status="CAUTION"
    else: status="NORMAL"
    return {"status":status,"flags":flags,"volume_ratio":vr,"return_5d":ret5}

def determine_long_term_decision(trend, accumulation, risk):
    reasons=[]
    if risk["status"] == "THESIS REVIEW": return {"signal":"THESIS REVIEW","reasons":risk["flags"]}
    score=accumulation["score"]
    if risk["status"] == "CAUTION": score=min(score,3); reasons.extend(risk["flags"])
    if score >= 10: signal="AGGRESSIVE ACCUMULATE"
    elif score >= 7: signal="STRONG ACCUMULATE"
    elif score >= 4: signal="ACCUMULATE"
    elif score >= 1: signal="NORMAL DCA"
    elif score >= -1: signal="HOLD / LIGHT DCA"
    else: signal="HOLD / WAIT"
    if trend["status"] == "WEAK" and signal in ("AGGRESSIVE ACCUMULATE","STRONG ACCUMULATE"): signal="ACCUMULATE / CAUTION"; reasons.append("장기 추세 약화로 공격적 매수 제한")
    reasons.extend(accumulation["reasons"])
    if not reasons: reasons.append("공포/할인 신호가 부족하므로 기본 적립 또는 보유")
    return {"signal":signal,"reasons":reasons}

# ==============================================================================================
# 14. FEAR & GREED
# ==============================================================================================

def fetch_fng_timeseries(start_date):

    url = (
        "https://production.dataviz.cnn.io/"
        "index/fearandgreed/"
        f"graphdata/{start_date}"
    )

    headers = {

        "User-Agent":
        (
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "Chrome/120 Safari/537.36"
        )
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=10
    )

    response.raise_for_status()

    json_data = response.json()

    data = (
        json_data
        ["fear_and_greed_historical"]
        ["data"]
    )

    df = pd.DataFrame(data)

    df["date"] = pd.to_datetime(
        df["x"] // 1000,
        unit="s"
    )

    df["fng"] = df["y"]

    df = (
        df[
            ["date", "fng"]
        ]
        .set_index("date")
    )

    return df


def interpret_fng(fng_df):

    if fng_df is None or fng_df.empty:

        return {
            "value": None,
            "zone": "N/A",
            "interpretation": "Fear & Greed 데이터 없음"
        }

    value = safe_float(
        fng_df["fng"].iloc[-1]
    )

    if value is None:

        return {
            "value": None,
            "zone": "N/A",
            "interpretation": "Fear & Greed 데이터 없음"
        }

    if value <= 20:

        return {
            "value": value,
            "zone": "EXTREME FEAR",
            "interpretation":
            (
                "시장 심리가 극단적으로 위축. "
                "자동 매수 신호가 아니라 장기 추세와 가격 구조 확인 후 "
                "분할매수 기회를 탐색."
            )
        }

    elif value <= 40:

        return {
            "value": value,
            "zone": "FEAR",
            "interpretation":
            (
                "시장 위험회피 심리 우세. "
                "좋은 자산의 조정 구간인지 확인."
            )
        }

    elif value < 60:

        return {
            "value": value,
            "zone": "NEUTRAL",
            "interpretation":
            "시장 심리 중립."
        }

    elif value < 80:

        return {
            "value": value,
            "zone": "GREED",
            "interpretation":
            (
                "시장 낙관 심리 우세. "
                "기존 포지션 자동 매도가 아니라 "
                "신규 진입 기대수익률을 점검."
            )
        }

    else:

        return {
            "value": value,
            "zone": "EXTREME GREED",
            "interpretation":
            (
                "시장 심리가 과열된 상태. "
                "자동 매도 신호가 아니며 "
                "신규 매수 속도와 포지션 리스크를 조절."
            )
        }


# ==============================================================================================
# 15. PERFORMANCE METRICS
# ==============================================================================================

def calculate_price_performance(df):

    close = df["Close"]

    cumulative_max = close.cummax()

    drawdown = (
        close
        /
        cumulative_max
        -
        1
    )

    max_drawdown = drawdown.min()

    start_price = safe_float(
        close.iloc[0]
    )

    end_price = safe_float(
        close.iloc[-1]
    )

    total_return = None

    if (
        start_price is not None
        and end_price is not None
        and start_price > 0
    ):

        total_return = (
            end_price
            /
            start_price
            -
            1
        )

    years = (
        df.index[-1]
        -
        df.index[0]
    ).days / 365.25

    annualized_return = None

    if (
        total_return is not None
        and years > 0
        and start_price > 0
    ):

        annualized_return = (
            end_price
            /
            start_price
        ) ** (
            1 / years
        ) - 1

    calmar_ratio = None

    if (
        annualized_return is not None
        and max_drawdown is not None
        and max_drawdown < 0
    ):

        calmar_ratio = (
            annualized_return
            /
            abs(max_drawdown)
        )

    return {

        "start_price": start_price,

        "end_price": end_price,

        "total_return": total_return,

        "annualized_price_return":
        annualized_return,

        "max_drawdown":
        max_drawdown,

        "calmar_ratio":
        calmar_ratio
    }


# ==============================================================================================
# 16. MAIN ANALYSIS
# ==============================================================================================

print_section(
    "1. DATA DOWNLOAD"
)

asset_df = download_price_data(
    ticker_symbol,
    START_DATE,
    TODAY
)

benchmark_df = download_price_data(
    MARKET_BENCHMARK,
    START_DATE,
    TODAY
)

fx_df = download_price_data(
    FX_TICKER,
    START_DATE,
    TODAY
)

dxy_df = download_price_data(
    DXY_TICKER,
    START_DATE,
    TODAY
)

print(
    f"Asset: {ticker_symbol}"
)

print(
    f"Market Benchmark: {MARKET_BENCHMARK}"
)


# ==============================================================================================
# 17. INDICATORS
# ==============================================================================================

print_section(
    "2. TECHNICAL INDICATORS"
)

asset_df = calculate_indicators(
    asset_df
)

benchmark_df = calculate_indicators(
    benchmark_df
)

asset_df = calculate_relative_strength(
    asset_df,
    benchmark_df
)

# 현재 기술 지표를 분석 단계에서 즉시 출력
last = asset_df.iloc[-1]
current_price = safe_float(last["Close"])
current_rsi = safe_float(last["RSI"])
current_atr_pct = safe_float(last["ATR_Pct"])
current_volume_ratio = safe_float(last["Volume_Ratio"])
current_rs = safe_float(last["Relative_Strength"])
current_drawdown = calculate_current_drawdown(asset_df)
current_bb_pct_b = safe_float(last.get("BB_PctB"))
current_ma200_distance = None
if safe_float(last.get("MA200")) not in (None, 0) and current_price is not None:
    current_ma200_distance = current_price / safe_float(last["MA200"]) - 1

print(f"현재 가격                : {fmt_num(current_price)}")
print(f"MA20                     : {fmt_num(last.get('MA20'))}")
print(f"MA50                     : {fmt_num(last.get('MA50'))}")
print(f"MA100                    : {fmt_num(last.get('MA100'))}")
print(f"MA200                    : {fmt_num(last.get('MA200'))}")
print(f"MA200 Slope              : {fmt_pct(last.get('MA200_Slope'))}")
print(f"RSI                      : {fmt_num(current_rsi)}")
print(f"ATR %                    : {fmt_pct(current_atr_pct)}")
print(f"Volume Ratio             : {fmt_num(current_volume_ratio)}")
print(f"Relative Strength        : {fmt_num(current_rs, 4)}")
print(f"Current Drawdown         : {fmt_pct(current_drawdown)}")
print(f"Bollinger %B             : {fmt_num(current_bb_pct_b)}")
print(f"MA200 Distance           : {fmt_pct(current_ma200_distance)}")

# ==============================================================================================
# 18. MARKET REGIME
# ==============================================================================================

market_regime = determine_market_regime(
    benchmark_df
)


# ==============================================================================================
# 19. FUNDAMENTAL
# ==============================================================================================

print_section(
    "3. FUNDAMENTAL ANALYSIS"
)

fundamental = calculate_fundamentals(
    ticker_symbol
)

fundamental_score = (
    calculate_fundamental_score(
        fundamental
    )
)

# 펀더멘털 원본 분석 결과를 분석 단계에서 즉시 출력
print(f"Analysis Type            : {fundamental.get('analysis_type', 'N/A')}")

if fundamental.get("is_etf", False):
    print(f"ETF Name                 : {fundamental.get('name', 'N/A')}")
    print(f"Category                 : {fundamental.get('category', 'N/A')}")
    print(f"Expense Ratio            : {fmt_pct(fundamental.get('expense_ratio'))}")
    print(f"Total Assets             : {fmt_num(fundamental.get('total_assets'))}")
    if fundamental_score is not None:
        print(f"Fundamental Score        : {fundamental_score.get('score', 'N/A')}")
else:
    print(f"Company Name             : {fundamental.get('name', 'N/A')}")
    print(f"Sector                   : {fundamental.get('sector', 'N/A')}")
    print(f"Industry                 : {fundamental.get('industry', 'N/A')}")
    print(f"Market Cap               : {fmt_num(fundamental.get('market_cap'))}")
    print(f"Trailing P/E             : {fmt_num(fundamental.get('trailing_pe'))}")
    print(f"Forward P/E              : {fmt_num(fundamental.get('forward_pe'))}")
    print(f"Revenue Growth           : {fmt_pct(fundamental.get('revenue_growth'))}")
    print(f"Profit Margin            : {fmt_pct(fundamental.get('profit_margin'))}")
    if fundamental_score is not None:
        print(f"Fundamental Score        : {fundamental_score.get('score', 'N/A')}")


# ==============================================================================================
# 20. TREND SCORE
# ==============================================================================================

trend = calculate_trend_score(
    asset_df,
    market_regime
)

# Fear & Greed 이후 Accumulation Score와 최종 결정을 계산한다.
accumulation = None
risk_check = None
final_decision = None


# ==============================================================================================
# 22. FEAR & GREED
# ==============================================================================================

try:

    fng_df = fetch_fng_timeseries(
        START_DATE.strftime("%Y-%m-%d")
    )

    fng_result = interpret_fng(
        fng_df
    )

except Exception as e:

    fng_df = None

    fng_result = {

        "value": None,

        "zone": "N/A",

        "interpretation":
        f"Fear & Greed 데이터 로딩 실패: {e}"
    }


# ==============================================================================================
# 22B. ACCUMULATION / RISK / FINAL DECISION
# ==============================================================================================

accumulation = calculate_accumulation_score(asset_df, fng_result.get("value"))
risk_check = calculate_risk_check(asset_df, trend, fundamental_score, fundamental["is_etf"])
final_decision = determine_long_term_decision(trend, accumulation, risk_check)

# ==============================================================================================
# 23. PERFORMANCE
# ==============================================================================================

performance = calculate_price_performance(
    asset_df
)


# ==============================================================================================
# 24. CURRENT VALUES
# ==============================================================================================

last = asset_df.iloc[-1]

current_price = safe_float(
    last["Close"]
)

current_rsi = safe_float(
    last["RSI"]
)

current_atr_pct = safe_float(
    last["ATR_Pct"]
)

current_volume_ratio = safe_float(
    last["Volume_Ratio"]
)

current_rs = safe_float(
    last["Relative_Strength"]
)

current_fx = latest(fx_df["Close"]) if not fx_df.empty else None
current_dxy = latest(dxy_df["Close"]) if not dxy_df.empty else None


# ==============================================================================================
# 25. FINAL REPORT
# ==============================================================================================

print_section(
    f"FINAL INVESTMENT ANALYSIS : {ticker_symbol}"
)

print(
    f"분석 날짜                : {TODAY}"
)

print(
    f"현재 가격                : {fmt_num(current_price)}"
)

print()

print(
    "[ MARKET REGIME ]"
)

print(
    f"시장 상태                : {market_regime['regime']}"
)

print(
    f"시장 판단                : {market_regime['reason']}"
)

print()

print(
    "[ FINAL DECISION ]"
)

print(
    f"최종 시그널              : {final_decision['signal']}"
)

print(
    f"Accumulation Score       : {accumulation['score']}"
)

for reason in final_decision["reasons"]:

    print(
        f"  - {reason}"
    )

print()

print("[ TREND / ACCUMULATION / RISK ]")
print(f"Trend Score               : {trend['score']} ({trend['status']})")
print(f"Accumulation Score        : {accumulation['score']}")
print(f"Current Drawdown          : {fmt_pct(accumulation['drawdown'])}")
print(f"Risk Check                : {risk_check['status']}")
for flag in risk_check['flags']:
    print(f"  - {flag}")

print()

print(
    "[ TECHNICAL ]"
)

print(
    f"MA20                     : {fmt_num(last['MA20'])}"
)

print(
    f"MA50                     : {fmt_num(last['MA50'])}"
)

print(
    f"MA100                    : {fmt_num(last['MA100'])}"
)

print(
    f"MA200                    : {fmt_num(last['MA200'])}"
)

print(
    f"MA200 Slope              : {fmt_pct(last['MA200_Slope'])}"
)

print(
    f"RSI                      : {fmt_num(current_rsi)}"
)

print(
    f"ATR %                    : {fmt_pct(current_atr_pct)}"
)

print(
    f"Volume Ratio             : {fmt_num(current_volume_ratio)}"
)

print(
    f"Relative Strength        : {fmt_num(current_rs, 4)}"
)

print()

print(
    "[ TREND SCORE REASONS ]"
)

for reason in trend["reasons"]:

    print(
        f"  - {reason}"
    )

print()

print(
    "[ FUNDAMENTAL ]"
)

print(
    f"Analysis Type            : {fundamental['analysis_type']}"
)

if fundamental["is_etf"]:

    print(
        f"ETF Name                 : {fundamental.get('name')}"
    )

    print(
        f"Category                 : {fundamental.get('category')}"
    )

    print(
        f"Expense Ratio            : {fmt_pct(fundamental.get('expense_ratio'))}"
    )

    print(
        f"Total Assets             : {fmt_num(fundamental.get('total_assets'))}"
    )

else:

    print(
        f"Company                  : {fundamental.get('name')}"
    )

    print(
        f"Sector                   : {fundamental.get('sector')}"
    )

    print(
        f"Industry                 : {fundamental.get('industry')}"
    )

    print(
        f"Trailing PER             : {fmt_num(fundamental.get('trailing_pe'))}"
    )

    print(
        f"Forward PER              : {fmt_num(fundamental.get('forward_pe'))}"
    )

    print(
        f"Revenue Growth           : {fmt_pct(fundamental.get('revenue_growth'))}"
    )

    print(
        f"Revenue CAGR             : {fmt_pct(fundamental.get('revenue_cagr'))}"
    )

    print(
        f"EPS Growth               : {fmt_pct(fundamental.get('eps_growth'))}"
    )

    print(
        f"EPS CAGR                 : {fmt_pct(fundamental.get('eps_cagr'))}"
    )

    print(
        f"Operating Margin         : {fmt_pct(fundamental.get('operating_margin'))}"
    )

    print(
        f"FCF Margin               : {fmt_pct(fundamental.get('fcf_margin'))}"
    )

    print(
        f"ROE                      : {fmt_pct(fundamental.get('roe'))}"
    )

    print(
        f"ROIC                     : {fmt_pct(fundamental.get('roic'))}"
    )

    print(
        f"Debt Ratio               : {fmt_pct(fundamental.get('debt_ratio'))}"
    )

    print(
        f"Share Count Change       : {fmt_pct(fundamental.get('share_count_change'))}"
    )

    print()

    print(
        f"Fundamental Score        : "
        f"{fundamental_score['score']}"
    )

    print(
        f"Fundamental Grade        : "
        f"{fundamental_score['grade']}"
    )

    for reason in fundamental_score["reasons"]:

        print(
            f"  - {reason}"
        )


print()

print(
    "[ MARKET SENTIMENT ]"
)

print(
    f"Fear & Greed             : {fmt_num(fng_result['value'])}"
)

print(
    f"Fear & Greed Zone        : {fng_result['zone']}"
)

print(
    f"Interpretation           : "
    f"{fng_result['interpretation']}"
)


print()

print(
    "[ MACRO ]"
)

print(
    f"USD/KRW                  : {fmt_num(current_fx)}"
)

print(
    f"Dollar Index             : {fmt_num(current_dxy)}"
)


print()

print(
    "[ PRICE PERFORMANCE ]"
)

print(
    f"Analysis Period          : "
    f"{asset_df.index[0].date()} ~ "
    f"{asset_df.index[-1].date()}"
)

print(
    f"Total Return             : "
    f"{fmt_pct(performance['total_return'])}"
)

print(
    f"Annualized Price Return  : "
    f"{fmt_pct(performance['annualized_price_return'])}"
)

print(
    f"Maximum Drawdown         : "
    f"{fmt_pct(performance['max_drawdown'])}"
)

print(
    f"Calmar Ratio             : "
    f"{fmt_num(performance['calmar_ratio'])}"
)


# ==============================================================================================
# 26. CHART 1 - PRICE / MA / BOLLINGER
# ==============================================================================================

plt.figure(figsize=(14, 7))

plt.plot(
    asset_df.index,
    asset_df["Close"],
    label="Close",
    linewidth=2
)

plt.plot(
    asset_df.index,
    asset_df["MA20"],
    label="MA20",
    linestyle="--"
)

plt.plot(
    asset_df.index,
    asset_df["MA50"],
    label="MA50",
    linestyle="--"
)

plt.plot(
    asset_df.index,
    asset_df["MA100"],
    label="MA100",
    linestyle="--"
)

plt.plot(
    asset_df.index,
    asset_df["MA200"],
    label="MA200",
    linewidth=2
)

plt.plot(
    asset_df.index,
    asset_df["BB_Upper"],
    label="BB Upper",
    linestyle=":"
)

plt.plot(
    asset_df.index,
    asset_df["BB_Lower"],
    label="BB Lower",
    linestyle=":"
)

plt.title(
    f"{ticker_symbol} Price / Trend"
)

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

save_or_close_chart("price_trend.png")
print("차트 해석: Close와 MA200의 관계를 가장 먼저 봅니다. 가격이 MA200 위이고 MA200이 상승하면 장기 상승추세로 봅니다. MA20/50은 조정 후 재진입 위치를 보는 보조선입니다.")


# ==============================================================================================
# 27. CHART 2 - RSI
# ==============================================================================================

plt.figure(figsize=(14, 5))
plt.plot(asset_df.index, asset_df["RSI"], label="RSI")
plt.axhline(70, color="red", linestyle="--", label="Extreme Heat")
plt.axhline(50, color="gray", linestyle="--")
plt.axhline(30, color="blue", linestyle="--", label="Oversold")

plt.title(
    f"{ticker_symbol} RSI"
)

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

save_or_close_chart("rsi.png")
print("차트 해석: RSI 70 이상을 자동 매도로 사용하지 않습니다. 상승추세에서는 50~70도 정상적인 강세일 수 있고, 30 이하에서는 추세가 유지되는지 확인한 뒤 분할매수 후보로 봅니다.")


# ==============================================================================================
# 28. CHART 3 - MACD
# ==============================================================================================

plt.figure(figsize=(14, 5))

plt.plot(
    asset_df.index,
    asset_df["MACD"],
    label="MACD"
)

plt.plot(
    asset_df.index,
    asset_df["MACD_Signal"],
    label="Signal"
)

plt.bar(
    asset_df.index,
    asset_df["MACD_Hist"],
    alpha=0.5,
    label="Histogram"
)

plt.axhline(
    0,
    linewidth=1
)

plt.title(
    f"{ticker_symbol} MACD"
)

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

save_or_close_chart("macd.png")
print("차트 해석: MACD는 추세의 방향과 변화 속도를 봅니다. 골든/데드크로스 하나만으로 매수·매도하지 않고 히스토그램 개선 여부를 보조적으로 확인합니다.")


# ==============================================================================================
# 29. CHART 4 - RELATIVE STRENGTH
# ==============================================================================================

plt.figure(figsize=(14, 5))

plt.plot(
    asset_df.index,
    asset_df["Relative_Strength"],
    label=f"RS vs {RS_BENCHMARK}"
)

plt.plot(
    asset_df.index,
    asset_df["RS_MA20"],
    label="RS MA20",
    linestyle="--"
)

plt.plot(
    asset_df.index,
    asset_df["RS_MA60"],
    label="RS MA60",
    linestyle="--"
)

plt.title(
    f"{ticker_symbol} Relative Strength vs {RS_BENCHMARK}"
)

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

save_or_close_chart("relative_strength.png")
print("차트 해석: 1.0 자체보다 RS의 방향과 RS MA20/60 위·아래를 봅니다. 벤치마크보다 지속적으로 강하면 장기 보유 자산으로서 우위가 있다는 의미입니다.")


# ==============================================================================================
# 30. CHART 5 - VOLUME RATIO
# ==============================================================================================

plt.figure(figsize=(14, 5))

plt.plot(
    asset_df.index,
    asset_df["Volume_Ratio"],
    label="Volume Ratio"
)

plt.axhline(1.0, color="gray", linestyle="--", label="Average Volume")
plt.axhline(1.2, color="red", linestyle="--", label="Volume Confirmation")

plt.title(
    f"{ticker_symbol} Volume Ratio"
)

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

save_or_close_chart("volume_ratio.png")
print("차트 해석: 1.0은 20일 평균 거래량입니다. 1.2 이상은 가격 돌파가 거래량으로 확인되는지 보는 기준이며, 거래량 하나만으로 매수하지 않습니다.")


# ==============================================================================================
# 31. CHART 6 - ATR %
# ==============================================================================================

plt.figure(figsize=(14, 5))
plt.plot(asset_df.index, asset_df["ATR_Pct"] * 100, label="ATR %")
plt.title(f"{ticker_symbol} Volatility (ATR %)")
plt.ylabel("%")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
save_or_close_chart("atr.png")
print("차트 해석: ATR%는 가격 변동성입니다. 높아질수록 매수 타이밍의 변동폭이 커졌다는 의미이지 매도 신호 자체는 아닙니다.")

# ==============================================================================================
# 32. CHART 7 - FEAR & GREED
# ==============================================================================================

if fng_df is not None and not fng_df.empty:
    plt.figure(figsize=(14, 5))
    plt.plot(fng_df.index, fng_df["fng"], label="Fear & Greed")
    plt.axhline(25, color="blue", linestyle="--", label="Extreme Fear")
    plt.axhline(50, color="gray", linestyle="--")
    plt.axhline(75, color="red", linestyle="--", label="Extreme Greed")
    plt.title("CNN Fear & Greed Index")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    save_or_close_chart("fear_greed.png")
print("차트 해석: 25 이하 공포는 장기 투자자에게 관심 구간입니다. 75 이상 탐욕은 신규매수 속도를 늦추는 참고자료이지 기존 ETF 자동매도 신호가 아닙니다.")
# ==============================================================================================
# 33. CHART 8 - USD/KRW
# ==============================================================================================

plt.figure(figsize=(14, 5))
plt.plot(fx_df.index, fx_df["Close"], label="USD/KRW")
plt.title("USD/KRW")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
save_or_close_chart("usdkrw.png")
print("차트 해석: 원화 투자자의 실제 체감 수익률과 달러자산 가치에 영향을 주는 환율입니다. ETF 매매 신호의 핵심이 아니라 거시환경 참고자료입니다.")

# ==============================================================================================
# 34. CHART 9 - DXY
# ==============================================================================================

plt.figure(figsize=(14, 5))
plt.plot(dxy_df.index, dxy_df["Close"], label="Dollar Index")
plt.title("Dollar Index")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
save_or_close_chart("dxy.png")
print("차트 해석: 달러 강세/약세 환경을 확인합니다. 금리·유동성·위험선호와 함께 해석하며 단독 매매신호로 사용하지 않습니다.")

# ==============================================================================================
# 35. FINAL SUMMARY
# ==============================================================================================

print_section(
    "FINAL SUMMARY"
)

print(
    f"{ticker_symbol} : "
    f"{final_decision['signal']}"
)

print(
    f"Trend Score     : "
    f"{trend['score']} / 4 ({trend['status']})"
)

print(
    f"Accumulation    : "
    f"{accumulation['score']}"
)

print(
    f"Risk Check      : "
    f"{risk_check['status']}"
)

print(
    f"Market Regime   : "
    f"{market_regime['regime']}"
)

if not fundamental["is_etf"]:

    print(
        f"Fundamental    : "
        f"{fundamental_score['grade']}"
    )

print()

print("중요 원칙:")
print("1. RSI 70 이상 = 자동 매도가 아님.")
print("2. Bollinger Upper 돌파 = 자동 매도가 아님.")
print("3. Fear & Greed Extreme Greed = 자동 매도가 아님.")
print("4. RSI 30 이하 = 자동 매수가 아님.")
print("5. 가장 중요한 판단 순서:")
print("   Market Regime")
print("   -> Long-term Trend")
print("   -> Relative Strength")
print("   -> Momentum")
print("   -> Entry Timing")
print("   -> Volatility Risk")
print("   -> Fundamental")
print("   -> Final Decision")
# endregion

# ==============================================================================================
# 36. GITHUB ACTIONS METADATA
# ==============================================================================================
with open("analysis_metadata.env", "w", encoding="utf-8") as f:
    f.write(f"ANALYSIS_DATE={TODAY}\n")
    f.write(f"FINAL_SIGNAL={final_decision['signal']}\n")
    f.write(f"TICKER={ticker_symbol}\n")
