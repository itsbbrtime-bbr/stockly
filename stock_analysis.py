import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo
import yfinance as yf
import requests
import re

################################################################################################
# GitHub Actions / headless environment 설정
# GitHub Ubuntu에는 AppleGothic이 없으므로 서버 기본 폰트를 사용합니다.
# 차트는 기본적으로 저장하지 않으며(SAVE_CHARTS=false), 필요할 때만 PNG로 저장합니다.
SAVE_CHARTS = os.getenv("SAVE_CHARTS", "false").lower() == "true"
CHART_DIR = os.getenv("CHART_DIR", "charts")
if SAVE_CHARTS:
    os.makedirs(CHART_DIR, exist_ok=True)

matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["font.family"] = "DejaVu Sans"

def finish_plot(filename):
    """GitHub Actions에서 finish_plot('fundamental.png') 대신 저장/종료."""
    if SAVE_CHARTS:
        plt.savefig(os.path.join(CHART_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close()

def normalize_yf_columns(df):
    """yfinance 버전에 따라 발생하는 MultiIndex 컬럼을 일반 컬럼으로 정리."""
    if df is None or df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        # (Price, Ticker) 형태가 일반적
        if len(df.columns.levels) == 2:
            df.columns = [
                col[0] if str(col[0]) in {"Open", "High", "Low", "Close", "Adj Close", "Volume"} else str(col[-1])
                for col in df.columns
            ]
    return df

def download_yf(ticker, start_date, end_date):
    """GitHub Actions에서 안정적으로 yfinance 데이터를 가져오는 래퍼."""
    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    df = normalize_yf_columns(df)
    if df is None or df.empty:
        raise RuntimeError(f"{ticker}: Yahoo Finance 데이터가 비어 있습니다.")
    return df

def latest_close(df):
    """Close 마지막 값을 float으로 안전하게 반환."""
    if df is None or df.empty or "Close" not in df.columns:
        return None
    value = df["Close"].iloc[-1]
    if isinstance(value, pd.Series):
        value = value.iloc[0]
    try:
        return float(value)
    except Exception:
        return None

# 미국 동부시간 기준으로 분석 날짜를 결정.
# yfinance의 end는 exclusive이므로 오늘 날짜 + 1일을 end로 사용합니다.
ny_today = datetime.now(ZoneInfo("America/New_York")).date()
today = ny_today + timedelta(days=1)

################################################################################################
# 티커(symbol) 입력
# 환경변수 TICKER가 있으면 그것을 우선 사용합니다.
ticker_symbol = os.getenv("TICKER", "SOXX")

# 비교 벤치마크
benchmark_symbol = os.getenv("BENCHMARK", "VOO")

################################################################################################

# region Entire File
################################################################################################
# DAILY INVESTMENT ANALYSIS
# GitHub Actions compatible / long-term ETF quantitative decision system
################################################################################################
print("#" * 130)
print(f"DAILY INVESTMENT ANALYSIS | {ticker_symbol} | {ny_today}")
print("#" * 130)
################################################################################################
# Fundamental Analysis Code
# Ticker 객체 생성
ticker = yf.Ticker(ticker_symbol)

# yfinance .info는 네트워크 상황에 따라 실패할 수 있으므로 자동화 환경에서 예외 처리
try:
    info = ticker.info or {}
except Exception as e:
    print(f"[WARNING] yfinance info 조회 실패: {e}")
    info = {}

per = info.get("trailingPE")            # PER : Price / Earnings (주가수익비율)
pbr = info.get("priceToBook")           # PBR : Price / Book (주가순자산비율)
peg = info.get("pegRatio")              # PEG : PER / EPS 성장률 (analyst 기반일 수 있음)
roe_info = info.get("returnOnEquity")   # ROE (info에 있으면 가져옴, 없으면 재무제표로 계산)
sector = info.get("sector", "N/A")
industry = info.get("industry", "N/A")
summary = info.get("longBusinessSummary", "N/A")
#sentences = summary.split(".")

# 재무제표 데이터 불러오기, yfinance에서 연간 재무제표(최근 연도부터 컬럼 정렬)
try:
    financials = ticker.financials
except Exception:
    financials = pd.DataFrame()

try:
    balancesheet = ticker.balance_sheet
except Exception:
    balancesheet = pd.DataFrame()

try:
    cashflow = ticker.cashflow
except Exception:
    cashflow = pd.DataFrame()

# 손익/대차/현금흐름이 비어있을 수 있으므로 항상 체크
latest_column = financials.columns[0] if (not financials.empty) else None

# 헬퍼 함수: 여러 후보 라벨 중 실제 데이터프레임에 존재하는 라벨을 찾아 반환, yfinance는 항목명 표기가 일관되지 않을 때가 많으므로, 후보 문자열 배열을 넣어 유연히 찾음
def find_row_label(df, candidates):
    """
    df: pandas DataFrame (ex. financials, balance_sheet, cashflow)
    candidates: ['Total Revenue', 'Revenue', ...] 등 후보 라벨 리스트
    반환: df에 실제로 존재하는 라벨 문자열 (존재하지 않으면 None)
    """
    if df is None or df.empty:
        return None

    labels = list(map(str, df.index))
    # 1) 정확 일치(대소문자 무시)
    for cand in candidates:
        for label in labels:
            if cand.lower() == label.lower():
                return label

    # 2) 후보가 레이블의 서브스트링으로 포함되는 경우
    for cand in candidates:
        for label in labels:
            if cand.lower() in label.lower():
                return label

    # 3) 레이블에 후보의 모든 토큰이 포함되는 경우
    for cand in candidates:
        cand_tokens = cand.lower().split()
        for label in labels:
            low_label = label.lower()
            if all(tok in low_label for tok in cand_tokens):
                return label

    return None

def safe_value_from_df(df, candidates):
    """
    df에서 candidates 중 하나를 찾아 최신(0번째) 값을 반환.
    값이 없거나 df가 비어있으면 None 반환.
    """
    label = find_row_label(df, candidates)
    if label is None:
        return None
    try:
        series = df.loc[label]
        # yfinance에서는 컬럼 0이 최신 연도(보통)임
        return series.iloc[0]
    except Exception:
        return None

# 주요 항목(후보 라벨 지정)
revenue_candidates = ["Total Revenue", "Revenue", "Net Sales", "Sales"]
net_income_candidates = ["Net Income", "Net Income Common Stockholders", "Net Income Applicable To Common Shares"]
operating_income_candidates = ["Operating Income", "Operating Income or Loss", "Income From Operations"]
total_assets_candidates = ["Total Assets"]
total_liabilities_candidates = ["Total Liab", "Total Liabilities"]
total_equity_candidates = ["Total Stockholder Equity", "Total Stockholders' Equity", "Total Shareholder Equity", "Total equity"]
current_assets_candidates = ["Total Current Assets", "Current Assets"]
current_liab_candidates = ["Total Current Liabilities", "Current Liabilities"]
operating_cf_candidates = ["Total Cash From Operating Activities", "Net Cash Provided by Operating Activities", "Operating Cash Flow"]
capex_candidates = ["Capital Expenditures", "CapEx", "Purchase of property, plant and equipment", "Payments for property, plant and equipment"]
cash_candidates = ["Cash And Cash Equivalents", "Cash", "Cash and short term investments", "Cash And Short Term Investments"]
rd_candidates = ["Research Development", "R&D", "Research and development", "Research & Development"]

# 연속(시계열) 데이터 시리즈(매출, 순이익 등) 가져오기 (추세/성장률 계산용)
def series_from_df(df, candidates):
    """
    df의 라벨을 찾아 series(기간별 값, 최신순) 반환.
    (존재하지 않으면 None 반환)
    """
    label = find_row_label(df, candidates)
    if label is None:
        return None
    try:
        s = df.loc[label].astype('float64')
        # index는 연도(또는 기간)들을 나타내므로 그대로 반환
        return s
    except Exception:
        return None

revenue_series = series_from_df(financials, revenue_candidates)        # 매출(기간별)
net_income_series = series_from_df(financials, net_income_candidates)  # 순이익(기간별)
operating_cf_series = series_from_df(cashflow, operating_cf_candidates)
capex_series = series_from_df(cashflow, capex_candidates)
fcf_series = None

# 만약 현금흐름표에 "Free Cash Flow"가 바로 있으면 사용, 없으면 operating_cf + capex 로 계산(일반적)
fcf_label = find_row_label(cashflow, ["Free Cash Flow", "Free cash flow"])
if fcf_label:
    try:
        fcf_series = cashflow.loc[fcf_label].astype('float64')
    except Exception:
        fcf_series = None
else:
    # 운영현금흐름 + 자본적지출(CapEx: 일반적으로 음수) -> FCF
    if (operating_cf_series is not None) and (capex_series is not None):
        # series끼리 더하면 기간이 겹치는 부분 계산됨
        try:
            fcf_series = operating_cf_series + capex_series
        except Exception:
            fcf_series = None

# 최신 연도(가용 데이터 기준) 값들 추출
def latest_value(series):
    """series가 있으면 최신(0번째) 값(스칼라)을 반환, 없으면 None"""
    if series is None or series.empty:
        return None
    try:
        return series.iloc[0]
    except Exception:
        return None

revenue_latest = latest_value(revenue_series)
net_income_latest = latest_value(net_income_series)
operating_cf_latest = latest_value(operating_cf_series)
capex_latest = latest_value(capex_series)
fcf_latest = latest_value(fcf_series)

total_assets_latest = safe_value_from_df(balancesheet, total_assets_candidates)
total_liabilities_latest = safe_value_from_df(balancesheet, total_liabilities_candidates)
total_equity_latest = safe_value_from_df(balancesheet, total_equity_candidates)
current_assets_latest = safe_value_from_df(balancesheet, current_assets_candidates)
current_liab_latest = safe_value_from_df(balancesheet, current_liab_candidates)
cash_latest = safe_value_from_df(balancesheet, cash_candidates)
rd_latest = safe_value_from_df(financials, rd_candidates)  # R&D는 보통 손익계산서에 존재

# 지표 계산 (공식과 의미는 주석으로 설명)
# ROE (Return on Equity, 자기자본이익률)
# ROE = Net Income / Total Shareholder Equity
# => 주주의 자본 한 단위(예: 1원)을 이용해 얼마의 순이익을 냈는지 측정
roe_calc = None
if (net_income_latest is not None) and (total_equity_latest is not None) and (total_equity_latest != 0):
    roe_calc = net_income_latest / total_equity_latest  # 소수(예: 0.15 -> 15%)

# Net Margin (순이익률) = Net Income / Revenue
net_margin = None
if (net_income_latest is not None) and (revenue_latest is not None) and (revenue_latest != 0):
    net_margin = net_income_latest / revenue_latest

# Operating Margin (영업이익률) = Operating Income / Revenue (영업이익 항목이 있다면)
operating_income_latest = safe_value_from_df(financials, operating_income_candidates)
operating_margin = None
if (operating_income_latest is not None) and (revenue_latest is not None) and (revenue_latest != 0):
    operating_margin = operating_income_latest / revenue_latest

# Debt Ratio (부채비율) = Total Liabilities / Total Assets
debt_ratio = None
if (total_liabilities_latest is not None) and (total_assets_latest is not None) and (total_assets_latest != 0):
    debt_ratio = total_liabilities_latest / total_assets_latest

# Current Ratio (유동비율) = Current Assets / Current Liabilities
current_ratio = None
if (current_assets_latest is not None) and (current_liab_latest is not None) and (current_liab_latest != 0):
    current_ratio = current_assets_latest / current_liab_latest

# Cash Ratio (현금비율) = Cash / Current Liabilities  (현금으로 단기부채를 얼마나 커버하는지)
cash_ratio = None
if (cash_latest is not None) and (current_liab_latest is not None) and (current_liab_latest != 0):
    cash_ratio = cash_latest / current_liab_latest

# R&D 비중 = R&D / Revenue (연구개발 투자 비중 — 성장성 판단에 도움)
rd_ratio = None
if (rd_latest is not None) and (revenue_latest is not None) and (revenue_latest != 0):
    rd_ratio = rd_latest / revenue_latest

# FCF (Free Cash Flow) 이미 위에서 series/ latest 구했음 (운영현금흐름 + CapEx)
# FCF가 None이면 계산 불가 (데이터 누락 가능성)

# Revenue CAGR (연평균성장률) — 사용 가능한 기간 전체를 이용한 CAGR 계산 (연간 재무제표 기준)
revenue_cagr = None
if revenue_series is not None and revenue_series.shape[0] >= 2:
    try:
        latest_rev = revenue_series.iloc[0]
        oldest_rev = revenue_series.iloc[-1]
        periods = revenue_series.shape[0] - 1  # ex: 3년치면 기간 차이는 2
        if oldest_rev > 0 and periods > 0:
            revenue_cagr = (latest_rev / oldest_rev) ** (1.0 / periods) - 1.0
    except Exception:
        revenue_cagr = None

# EPS 성장률 대체(근사) — EPS가 직접 없으면 Net Income 성장률을 EPS 성장률의 Proxy로 사용
eps_growth_approx = None
if net_income_series is not None and net_income_series.shape[0] >= 2:
    try:
        latest_net = net_income_series.iloc[0]
        oldest_net = net_income_series.iloc[-1]
        periods = net_income_series.shape[0] - 1
        if oldest_net > 0 and periods > 0:
            eps_growth_approx = (latest_net / oldest_net) ** (1.0 / periods) - 1.0
    except Exception:
        eps_growth_approx = None

# PEG (근사) : 만약 info에 peg가 없으면 PER / (EPS 성장률(%)) 로 근사 가능 (EPS 성장률을 %로 사용)
peg_approx = peg
if peg_approx is None and per is not None and eps_growth_approx is not None:
    try:
        eps_growth_pct = eps_growth_approx * 100.0
        if eps_growth_pct > 0:
            peg_approx = per / eps_growth_pct  # 예: PER=20, 성장률=20% -> PEG=1
    except Exception:
        peg_approx = None

# 출력(정리된 숫자와 설명)
def fmt_num(x):
    """숫자형 데이터를 1,000 단위 쉼표 표시"""
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "N/A"
    if abs(x) >= 1:
        return f"{x:,.0f}"
    else:
        return f"{x:.4f}"
    
def fmt_num_unit(x):
    """현금 단위 항목 표시 (1,000 단위 쉼표)"""
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "N/A"
    return f"${x:,.0f}"

def fmt_pct(x):
    """비율 데이터 % 표시"""
    if x is None:
        return "N/A"
    try:
        return f"{x * 100:.2f}%"
    except Exception:
        return "N/A"

# 현금 단위 항목
money_keys = [
    "매출", "순이익", "영업현금흐름", "CapEx", "FCF"
]

# 퍼센트로 표시할 항목
percent_keys = [
    "ROE", "순이익률", "영업이익률", "부채비율", 
    "유동비율", "현금비율", "R&D 비중", "CAGR", "성장률"
]

# 항목 이름과 값
output_items = [
    ("PER (Price to Earnings Ratio)", per),                         #주가수익비율, Price/Earnings
    ("PBR (Price to Book Ratio)", pbr),                             #주가순자산비율, Price/Book
    ("PEG (Price/Earnings to Growth)", peg),                        #정보/계산된 값, 낮을수록 성장 대비 저평가
    ("PEG Approximation", peg_approx),                              #근사 PEG
    ("ROE (Return on Equity)", roe_info),                           #ROE (info)
    ("Revenue (Latest)", revenue_latest),                           #매출
    ("Net Income (Latest)", net_income_latest),                     #순이익
    ("Operating Cash Flow (Latest)", operating_cf_latest),          #영업현금흐름
    ("CapEx (Capital Expenditures)", capex_latest),                 #CapEx
    ("FCF (Free Cash Flow)", fcf_latest),                           #FCF(Operating CF + CapEx)
    ("Revenue CAGR (Available Period)", revenue_cagr),              #매출 CAGR 
    ("EPS Growth Approx (Based on Net Income)", eps_growth_approx), #EPS 성장률 근사
    ("ROE Calc (Net Income / Total Equity)", roe_calc),             #ROE
    ("Net Margin (Net Income Margin)", net_margin),                 #순이익률
    ("Operating Margin", operating_margin),                         #영업이익률
    ("Debt Ratio (Total Liabilities / Total Assets)", debt_ratio),  #부채비율
    ("Current Ratio", current_ratio),                               #유동비율
    ("Cash Ratio", cash_ratio),                                     #현금비율
    ("R&D Ratio (R&D / Revenue)", rd_ratio)                         #R&D 비중
]

# 가장 긴 항목명 찾기
max_key_len = max(len(item[0]) for item in output_items)

# 기업 정보 총정리
print("#"*130)
print(f"📊 Ticker: {ticker_symbol}")
print("---- 기업 정보 ----")
print(f"산업 (Sector)   : {sector}")
print(f"업종 (Industry) : {industry}")
#print(f"사업 개요 (Summary):\n{summary}")
texts = re.split(r'(?<=[.!?])\s+', summary)
for sentence in texts:
    sentence = sentence.strip()  # 공백 제거
    if sentence:  # 빈 문장이 아닐 때만 출력
        print(sentence)    # 다시 . 붙여서 출력
print("#"*130)
print(f"📊 Ticker: {ticker_symbol}")
print("---- 주요 재무 지표 ----")

for key, val in output_items:
    if any(mk in key for mk in money_keys):
        val_str = fmt_num_unit(val)
    elif any(pk in key for pk in percent_keys):
        val_str = fmt_pct(val)
    else:
        val_str = fmt_num(val)
    print(f"{key.ljust(max_key_len)} : {val_str}")
print("#"*130)


# DataFrame로 보기 쉽게 정리
summary = {
    "ticker": ticker_symbol,
    "revenue_latest": revenue_latest,
    "net_income_latest": net_income_latest,
    "fcf_latest": fcf_latest,
    "revenue_cagr": revenue_cagr,
    "eps_growth_approx": eps_growth_approx,
    "roe_calc": roe_calc,
    "net_margin": net_margin,
    "operating_margin": operating_margin,
    "debt_ratio": debt_ratio,
    "current_ratio": current_ratio,
    "cash_ratio": cash_ratio,
    "rd_ratio": rd_ratio,
    "peg_approx": peg_approx,
    "PER": per,
    "PBR": pbr
}
summary_df = pd.DataFrame([summary])
# 보기 편한 컬럼 순서
cols = ["ticker","PER","PBR","peg_approx","revenue_latest","net_income_latest","fcf_latest",
        "revenue_cagr","eps_growth_approx","roe_calc","net_margin","operating_margin",
        "debt_ratio","current_ratio","cash_ratio","rd_ratio"]
summary_df = summary_df[cols]
pd.set_option('display.float_format', lambda x: f'{x:,.2f}')
#print("\n== Summary DataFrame ==")
#print(summary_df.T)

# 투자 추천 기준 설정
recommendation = "보류"  # 기본값
reason_list = []

# PER 기준 (낮을수록 저평가, 15~25 범위는 적정)
if per is not None:
    if per < 25:
        reason_list.append(f"[GOOD] PER이 낮아 저평가 판단 ({per:.2f})")
    elif per > 35:
        reason_list.append(f"[BAD] PER이 높아 고평가 우려 ({per:.2f})")

# PEG 기준 (성장 대비 저평가)
if peg_approx is not None:
    if peg_approx < 1:
        reason_list.append(f"[GOOD] PEG가 1 미만, 성장 대비 저평가 ({peg_approx:.2f})")
    elif peg_approx > 2:
        reason_list.append(f"[BAD] PEG가 2 이상, 성장 대비 고평가 ({peg_approx:.2f})")

# ROE 기준 (높을수록 좋음, 15% 이상 양호)
if roe_calc is not None:
    if roe_calc > 0.15:
        reason_list.append(f"[GOOD] ROE가 높음 ({roe_calc*100:.1f}%)")
    elif roe_calc < 0.05:
        reason_list.append(f"[BAD] ROE가 낮음 ({roe_calc*100:.1f}%)")

# Net Margin 기준 (순이익률 10% 이상 양호)
if net_margin is not None:
    if net_margin > 0.10:
        reason_list.append(f"[GOOD] 순이익률 양호 ({net_margin*100:.1f}%)")
    elif net_margin < 0.05:
        reason_list.append(f"[BAD] 순이익률 낮음 ({net_margin*100:.1f}%)")

# 부채비율 기준 (50% 이하 안정적)
if debt_ratio is not None:
    if debt_ratio < 0.5:
        reason_list.append(f"[GOOD] 부채비율 안정적 ({debt_ratio*100:.1f}%)")
    elif debt_ratio > 1.0:
        reason_list.append(f"[BAD] 부채비율 높음 ({debt_ratio*100:.1f}%)")

# FCF (Free Cash Flow) 기준
if fcf_latest is not None:
    if fcf_latest > 0:
        reason_list.append(f"[GOOD] FCF 플러스 ({fcf_latest:,.0f})")
    else:
        reason_list.append(f"[BAD] FCF 마이너스 ({fcf_latest:,.0f})")

# R&D 비중 (10% 이상이면 성장 투자 적극)
if rd_ratio is not None:
    if rd_ratio > 0.10:
        reason_list.append(f"[GOOD] R&D 비중 높음 ({rd_ratio*100:.1f}%)")
    elif rd_ratio < 0.10:
        reason_list.append(f"[BAD] R&D 비중 낮음 ({rd_ratio*100:.1f}%)")

# 최종 추천 판단
# 간단히 플러스 요인 > 마이너스 요인이 많으면 추천
pos_score = sum("양호" in r or "저평가" in r or "플러스" in r or "높음" in r for r in reason_list)
neg_score = sum("고평가" in r or "낮음" in r or "마이너스" in r for r in reason_list)

if pos_score >= 3 and neg_score <= 1:
    recommendation = "추천"
elif neg_score >= 3:
    recommendation = "비추천"
else:
    recommendation = "보류"

# Fundamental 기준 투자 여부 결정
print(f"📌 {ticker_symbol} 투자 추천: {recommendation}")
print("🔹 판단 이유:")
for r in reason_list:
    print(f"  - {r}")
print("#"*130)

# 1. 최근 3년 데이터를 추출하는 헬퍼 함수
def get_3y_data(df, candidates):
    # candidates 리스트 중 데이터프레임 인덱스에 존재하는 첫 번째 항목 찾기
    target_key = next((c for c in candidates if c in df.index), None)
    if target_key:
        data = df.loc[target_key].sort_index(ascending=True).tail(3)
        # 모든 데이터가 0이거나 결측치인 경우 체크
        if data.sum() == 0 or data.isna().all():
            return None
        return data
    return None

# 2. 각 카테고리별 데이터 준비
years = [d.strftime('%Y') for d in financials.columns[::-1][-3:]] # 최근 3년 연도 라벨

# 그룹 A: 수익성 (Revenue, Net Income, R&D)
rev_3y = get_3y_data(financials, ['Total Revenue'])
net_3y = get_3y_data(financials, ['Net Income'])
rd_3y = get_3y_data(financials, ['Research And Development', 'Research & Development'])

# 그룹 B: 현금흐름 (OCF, CapEx, FCF)
ocf_3y = get_3y_data(cashflow, ['Operating Cash Flow'])
capex_3y = get_3y_data(cashflow, ['Capital Expenditure'])
if capex_3y is not None: capex_3y = capex_3y.abs()
fcf_3y = get_3y_data(cashflow, ['Free Cash Flow'])

# 그룹 C: 재무구조 (Assets, Liabilities, Equity)
asset_3y = get_3y_data(balancesheet, ['Total Assets'])
liab_3y = get_3y_data(balancesheet, ['Total Liabilities Net Minor Interest', 'Total Liabilities', 'Total Liab', "Total Current Liabilities", "Current Liabilities"])
equity_3y = get_3y_data(balancesheet, ['Stockholders Equity', 'Total Equity Gross Minority Interest'])

# 3. 차트 그리기 (1행 3열 구성)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# [Plot 1] 수익성
if rev_3y is not None: axes[0].bar(years, rev_3y, label='Revenue', color='skyblue', alpha=0.6)
if net_3y is not None: axes[0].plot(years, net_3y, label='Net Income', marker='o', color='red')
if rd_3y is not None: axes[0].plot(years, rd_3y, label='R&D', marker='s', color='green', linestyle='--')
axes[0].set_title('Profitability & R&D')
if axes[0].get_legend_handles_labels()[0]: axes[0].legend() # 데이터가 있을 때만 범례 표시

# [Plot 2] 현금흐름
if ocf_3y is not None: axes[1].bar(years, ocf_3y, label='Op Cash Flow', color='lightgreen', alpha=0.6)
if capex_3y is not None: axes[1].bar(years, capex_3y, label='CapEx', color='orange', alpha=0.5)
if fcf_3y is not None: axes[1].plot(years, fcf_3y, label='Free Cash Flow', marker='d', color='blue')
axes[1].set_title('Cash Flow & Investment')
if axes[1].get_legend_handles_labels()[0]: axes[1].legend()

# [Plot 3] 재무구조
if equity_3y is not None and liab_3y is not None:
    axes[2].bar(years, equity_3y, label='Equity', color='navy', alpha=0.7)
    axes[2].bar(years, liab_3y, bottom=equity_3y, label='Liabilities', color='crimson', alpha=0.7)
elif equity_3y is not None: # 부채 데이터가 없는 경우 자본만 표시
    axes[2].bar(years, equity_3y, label='Equity', color='navy', alpha=0.7)
axes[2].set_title('Capital Structure')
if axes[2].get_legend_handles_labels()[0]: axes[2].legend()

# 데이터가 아예 없는 그래프에 안내 메시지 추가
for ax in axes:
    if not ax.get_legend_handles_labels()[0]:
        ax.text(0.5, 0.5, 'No Data Available (e.g. ETF)', transform=ax.transAxes, 
                ha='center', va='center', fontsize=12, color='gray')

plt.tight_layout()
finish_plot('price_signals.png')

def print_fundamental_guide():
    guide = [
        ("Revenue (매출액)", "기업이 상품이나 서비스를 팔아 벌어들인 총액입니다. '외형 성장'의 핵심 지표입니다."),
        ("Net Income (당기순이익)", "매출에서 모든 비용과 세금을 빼고 남은 최종 이익입니다. 주주에게 돌아가는 몫입니다."),
        ("R&D (연구개발비)", "미래 경쟁력을 위해 기술 개발에 투자하는 비용입니다. 성장주 분석에 중요합니다."),
        ("Op Cash Flow (영업현금흐름)", "장부상이 아닌, 영업활동을 통해 실제로 회사 금고에 들어온 현금입니다."),
        ("CapEx (설비투자)", "미래를 위해 공장, 기계, 토지 등에 재투자한 비용입니다."),
        ("Free Cash Flow (잉여현금흐름)", "영업으로 번 돈에서 투자를 뺀 '진짜 남는 돈'입니다. 배당이나 자사주 매입의 원천이 됩니다."),
        ("Equity (자본)", "회사의 주인인 주주들의 몫입니다. 자산에서 부채를 뺀 순자산입니다."),
        ("Liabilities (부채)", "회사가 갚아야 할 빚입니다. 자본 대비 너무 많으면 재무 리스크가 커집니다."),
        ("Total Assets (자산)", "자본과 부채를 합친 것으로, 현재 회사가 운용 중인 전체 자원 규모입니다.")
    ]

    print("#"*130)
    print("주요 펀더멘털 지표 용어 설명")
    print("#"*130)
    for title, desc in guide:
        print(f"{title:<20} : {desc}")
    print("#"*130)

# 함수 호출
print_fundamental_guide()

################################################################################################


################################################################################################
# Techincal Analysis Code
# 데이터 전처리
start = today - timedelta(days=420)

# 종목 입력
stockName = ticker_symbol
data = download_yf(stockName, start, today)

# 벤치마크: 시장 대비 상대강도 계산용
benchmark_data = download_yf(benchmark_symbol, start, today)

# 환율 / 달러인덱스
stockDOL = "USDKRW=X"
try:
    dataDER = download_yf(stockDOL, start, today)
    ClosedDER = latest_close(dataDER)
except Exception as e:
    print(f"[WARNING] USD/KRW 데이터 조회 실패: {e}")
    dataDER = pd.DataFrame()
    ClosedDER = None

dollarIndex = "DX-Y.NYB"
try:
    dataDI = download_yf(dollarIndex, start, today)
    ClosedDI = latest_close(dataDI)
except Exception as e:
    print(f"[WARNING] Dollar Index 데이터 조회 실패: {e}")
    dataDI = pd.DataFrame()
    ClosedDI = None

# 데이터 다운로드
session = requests.Session()  # 세션 생성
#data = yf.download(stockName, start='2024-01-01', session=session)
#data = yf.download(stockName, start='2025-01-01')

# 데이터 정리
data['High_Value'] = data['High']
data['Low_Value'] = data['Low']
data['Close_Value'] = data['Close']
data['Volume_Value'] = data['Volume']
################################################################################################

################################################################################################
# 지표 계산

# 이동평균선 계산
data['MA05'] = data['Close'].rolling(window=5).mean()
data['MA20'] = data['Close'].rolling(window=20).mean()
data['MA50'] = data['Close'].rolling(window=50).mean()
data['MA100'] = data['Close'].rolling(window=100).mean()
data['MA120'] = data['Close'].rolling(window=120).mean()
data['MA200'] = data['Close'].rolling(window=200).mean()

# 시장 대비 상대강도
benchmark_close = benchmark_data['Close'].reindex(data.index).ffill()
data['Relative_Strength'] = data['Close'] / benchmark_close
data['RS_MA20'] = data['Relative_Strength'].rolling(20).mean()

# 볼린저 밴드 계산 (20일 기준) : 가격이 평균에서 얼마나 벗어났는지
# 밴드 상단 돌파 → 과매수 가능성, 하단 돌파 → 과매도 가능성
# 밴드가 좁아지면 곧 변동성 증가 가능성 (즉, 변동 전 조짐)
data['BB_Mid'] = data['MA20']
data['BB_Std'] = data['Close'].rolling(window=20).std()
data['BB_Upper'] = data['MA20'] + (2 * data['BB_Std'])
data['BB_Lower'] = data['MA20'] - (2 * data['BB_Std'])

# RSI 계산 (14일 기준) : 최근 상승 vs 하락 강도를 비교해 과열/침체를 판단
delta = data['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
data['RSI'] = 100 - (100 / (1 + rs))

# MACD 계산 : 두 이동평균선 간 **거리 차이(추세 강도)**를 분석
# MACD선 = 단기 EMA - 장기 EMA (보통 12일 - 26일)
# Signal선 = MACD선의 9일 EMA
# MACD선이 Signal선을 위로 돌파 → 매수 시그널
# MACD선이 Signal선을 아래로 돌파 → 매도 시그널
data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
data['MACD'] = data['EMA12'] - data['EMA26']
data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

# OBV (On Balance Volume) : 가격이 상승하면 거래량을 더하고, 하락하면 빼는 방식. 세력의 매집 여부를 판단 가능.
# OBV > OBV_Mean 위로 돌파 → 매수 시그널
# OBV > OBV_Mean 아래로 돌파 → 매도 시그널
# OBV 상승, 가격은 횡보 → 세력매집, 선행매수 기회
# OBV 하락, 가격은 상승 → 이탈조짐, 추세약화 경고
price_diff = data['Close_Value'].diff()
direction = np.where(price_diff > 0, 1, np.where(price_diff < 0, -1, 0))
data['OBV'] = (data['Volume_Value'] * direction).cumsum()
data['OBV_Mean'] = data['OBV'].rolling(window=20).mean()
data['OBV_Signal'] = 0

data['OBV_Prev'] = data['OBV'].shift(1)
data['OBV_Mean_Prev'] = data['OBV_Mean'].shift(1)
data.loc[(data['OBV_Prev'] < data['OBV_Mean_Prev']) & (data['OBV'] > data['OBV_Mean']), 'OBV_Signal'] = 1  # 매수 시그널
data.loc[(data['OBV_Prev'] > data['OBV_Mean_Prev']) & (data['OBV'] < data['OBV_Mean']), 'OBV_Signal'] = -1 # 매도 시그널

# CCI (Commodity Channel Index) : 가격이 평균과 얼마나 다른지 측정 → +100 이상이면 과매수, -100 이하면 과매도 신호
tp = (data['High_Value'] + data['Low_Value'] + data['Close_Value']) / 3
smaTp = tp.rolling(window=20).mean()
mad = (tp - smaTp).abs().rolling(window=20).mean()
data['CCI'] = (tp - smaTp) / (0.015 * mad)

# Stochastic Oscillator : 종가가 최근 N일 동안의 최고/최저 범위에서 어디에 위치하는지 확인, %K > %D 교차 시 매수 신호, 반대면 매도 신호
lowMin = data['Low_Value'].rolling(window=14).min()
highMax = data['High_Value'].rolling(window=14).max()
data['%K'] = (data['Close_Value'] - lowMin) / (highMax - lowMin) * 100
data['%D'] = data['%K'].rolling(window=3).mean()

# 최대 손실, 수익률 분석
# MDD (Maximum Drawdown, 최대 낙폭) : 투자 중 최악의 손실이 얼마나 컸는지 측정.
# CAGR (연평균 수익률) : 연도별 평균 수익률을 계산해 장기 투자 성과 평가!
# Calmar Ratio 계산 (CAGR을 MDD의 절대값으로 나눔), 위험 대비 수익 효율성을 판단.
cumulativeMax = data['Close_Value'].cummax()
drawdown = (data['Close_Value'] - cumulativeMax) / cumulativeMax
maxDrawdown = drawdown.min()

startPrice = data['Close_Value'].iloc[0]
endPrice = data['Close_Value'].iloc[-1]
numYears = (data.index[-1] - data.index[0]).days / 365.25
cagr = ((endPrice / startPrice) ** (1 / numYears)) -1

# CAGR이 양수일 때만 Calmar Ratio를 계산하고, 음수면 0 또는 None 처리
# 0.5 미만: 수익 대비 위험이 큽니다. (비효율적)
# 1.0 이상: 상당히 우수한 전략입니다. (MDD만큼은 매년 벌어준다는 뜻)
# 2.0 이상: 매우 뛰어난 전략이며, 리스크 관리가 탁월한 상태입니다.
# 5.0 이상: 현실에서 지속하기 어려운 수준의 엄청난 성과입니다.
if cagr > 0 and maxDrawdown != 0:
    calmarRatio = cagr / abs(maxDrawdown)
else:
    calmarRatio = 0  # 혹은 float('nan')
################################################################################################

################################################################################################
# 매수/매도 조건 정의 및 데이터 저장

# 매수 신호 조건 정의
buyRsi = data['RSI'] < 30                                                                                # RSI가 30 이하
buyGoldenCross = (data['MA20'] > data['MA50']) & (data['MA20'].shift(1) <= data['MA50'].shift(1))        # 골든 크로스
buyMacd = (data['MACD'] > data['MACD_Signal']) & (data['MACD'].shift(1) <= data['MACD_Signal'].shift(1)) # MACD 골든 크로스
buyBollinger = data['Close_Value'] < data['BB_Lower']                                                    # 볼린저 밴드 하단 터치
buyCci = data['CCI'] < -100
buyStoch = (data['%K'] > data['%D']) & (data['%K'].shift(1) <= data['%D'].shift(1)) & (data['%K'] < 20)
buyObv = data['OBV_Signal'] == 1

# 매수 신호 및 날짜/가격 저장
data['Buy_Signal'] = buyRsi | buyGoldenCross | buyMacd | buyBollinger | buyCci | buyStoch | buyObv
buyDates = data[data['Buy_Signal']].index
buyPrices = data.loc[buyDates, 'Close']

# 매도 신호 조건 정의
sellRsi = data['RSI'] > 70                                                                                # RSI가 70 이상
sellDeadCross = (data['MA20'] < data['MA50']) & (data['MA20'].shift(1) >= data['MA50'].shift(1))          # 데드 크로스
sellMacd = (data['MACD'] < data['MACD_Signal']) & (data['MACD'].shift(1) >= data['MACD_Signal'].shift(1)) # MACD 데드 크로스
sellBollinger = data['Close_Value'] > data['BB_Upper']                                                    # 볼린저 밴드 상단 터치
sellCci = data['CCI'] > 100
sellStoch = (data['%K'] < data['%D']) & (data['%K'].shift(1) >= data['%D'].shift(1)) & (data['%K'] > 80)
sellObv = data['OBV_Signal'] == -1

# 매도 신호 및 날짜/가격 저장
data['Sell_Signal'] = sellRsi | sellDeadCross | sellMacd | sellBollinger  | sellCci | sellStoch | sellObv
sellDates = data[data['Sell_Signal']].index

sellPrices = data.loc[sellDates, 'Close']
################################################################################################

################################################################################################
# 차트 그리기
plt.figure(figsize=(12,6))
plt.plot(data.index, data['Close'], label=f'{stockName} Close Price', color='blue', linewidth=2)
plt.plot(data.index, data['MA05'], label='5-day MA', color='red', linestyle='--')
plt.plot(data.index, data['MA20'], label='20-day MA', color='purple', linestyle='--')
plt.plot(data.index, data['MA50'], label='50-day MA', color='black', linestyle='--')
plt.plot(data.index, data['MA100'], label='100-day MA', color='brown', linestyle='--')
plt.plot(data.index, data['MA200'], label='200-day MA', color='darkgreen', linestyle='-')
plt.plot(data.index, data['MA120'], label='120-day MA', color='green', linestyle='--')
plt.plot(data.index, data['BB_Upper'], label='Bollinger Upper', color='gray', linestyle='dotted')
plt.plot(data.index, data['BB_Lower'], label='Bollinger Lower', color='gray', linestyle='dotted')
plt.scatter(buyDates, buyPrices, color='gold', label='Buy Signal', marker='^', s=100)
plt.scatter(sellDates, sellPrices, color='red', label='Sell Signal', marker='v', s=100)
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=10))
plt.gca().yaxis.tick_right()
plt.gca().yaxis.set_label_position("right")
ymin = data['Close_Value'].min() - data['Close_Value'].min()*0.1
ymax = data['Close_Value'].max() + data['Close_Value'].max()*0.1
yticks = np.linspace(ymin, ymax, 20)
plt.xticks(rotation=90)
plt.yticks(yticks)
plt.title(f'{stockName} Buy & Sell Signals', fontsize=14)
plt.ylabel('Price (USD)', fontsize=12)
plt.legend()
plt.grid()
finish_plot('volume.png')
# QQQ 평단 표시
# 마지막 종가 가져오기
#lastClosePrice = data['Close'].iloc[-1]
#if stockName == 'QQQ':
#    averagePurchasePrice = 526.1085
#    profitPercentage = ((lastClosePrice - averagePurchasePrice) / averagePurchasePrice) * 100
#    plt.axhline(y=averagePurchasePrice, color='red', linestyle='-', label='Average Purchase Price')
print("'Bollinger Band' : 가격이 평균에서 얼마나 벗어났는지 확인.")
print("상단 돌파 시 과매수, 하단 돌파 시 과매도. 밴드 수축 시 변동성 예고")
################################################################################################

################################################################################################
# 거래량 차트 그리기
colors = np.where(data['Close_Value'] > data['Close_Value'].shift(1), 'red', 'blue')
colors[0] = 'gray'
plt.figure(figsize=(12,6))
plt.bar(data.index, data['Volume_Value'], label=f'{stockName} Volume', color=colors)
plt.title(f"{stockName} Volume")
plt.ylabel("Volume")
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=10))
plt.gca().yaxis.tick_right()
plt.gca().yaxis.set_label_position("right")
plt.xticks(rotation=90)
plt.grid(True, alpha=0.4)
plt.legend()
plt.tight_layout()
finish_plot('rsi.png')
################################################################################################

################################################################################################
# RSI 차트 그리기
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['RSI'], label='RSI', color='darkorange')
plt.axhline(70, color='red', linestyle='--', label='Overbought (70)')
plt.axhline(30, color='blue', linestyle='--', label='Oversold (30)')
plt.title(f'{stockName} RSI Chart', fontsize=14)
plt.ylabel('RSI')
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=10))
plt.gca().yaxis.tick_right()
plt.gca().yaxis.set_label_position("right")
plt.xticks(rotation=90)
plt.grid(True)
plt.legend()
plt.tight_layout()
finish_plot('macd.png')
print("'RSI (14days)' : 최근 상승 vs 하락 강도를 비교해 과열/침체를 판단.")
print("70 이상 과열(매도 검토), 30 이하 침체(매수 검토)")
################################################################################################

################################################################################################
# MACD 차트 그리기
macd_hist = data['MACD'] - data['MACD_Signal']
plt.figure(figsize=(12,6))
plt.plot(data.index, data['MACD'], color='blue', label='MACD', linewidth=1)
plt.plot(data.index, data['MACD_Signal'], color='red', label='Signal', linewidth=1)
    
# MACD 히스토그램 (0보다 크면 빨강, 작으면 파랑)
colors = ['red' if x > 0 else 'blue' for x in macd_hist]
plt.bar(data.index, macd_hist, color=colors, alpha=0.5, label='Hist')
plt.axhline(0, color='black', linewidth=0.5) # 기준선 0
plt.title(f'{stockName} MACD Chart', fontsize=14)
plt.ylabel("MACD")
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=10))
plt.gca().yaxis.tick_right()
plt.gca().yaxis.set_label_position("right")
plt.xticks(rotation=90)
plt.legend()
plt.grid(True, alpha=0.3)
finish_plot('obv.png')
print("'MACD' : 두 이동평균선 간 거리 차이(추세 강도)를 분석 (단기 EMA - 장기 EMA)")
print("MACD선이 Signal선을 위로 돌파 → 매수 시그널")
print("MACD선이 Signal선을 아래로 돌파 → 매도 시그널")
################################################################################################

################################################################################################
# OBV 거래량 지표 차트 그리기
plt.figure(figsize=(12,6))
plt.plot(data.index, data['OBV'], label='OBV', color='blue')
plt.plot(data.index, data['OBV_Mean'], label='OBV Mean (20-day)', color='orange', linestyle='--')
plt.title(f'{stockName} OBV Chart', fontsize=14)
plt.ylabel('OBV')
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=10))
plt.gca().yaxis.tick_right()
plt.gca().yaxis.set_label_position("right")
plt.xticks(rotation=90)
plt.legend()
plt.grid()
finish_plot('cci.png')
print("'OBV' : 가격이 상승하면 거래량을 더하고, 하락하면 빼는 방식. 세력의 매집 여부를 판단 가능.")
print("OBV > OBV_Mean 위로 돌파 → 매수 시그널, 아래로 돌파 → 매도 시그널")
print("OBV 상승, 가격은 횡보 → 세력매집, 선행매수 기회")
print("OBV 하락, 가격은 상승 → 이탈조짐, 추세약화 경고")
################################################################################################

################################################################################################
# CCI 차트 그리기
plt.figure(figsize=(12,6))
plt.plot(data.index, data['CCI'], color='orange', label='CCI')
plt.axhline(100, color='red', linestyle='--', alpha=0.5, label='Overbought (100)')   # 과매수 기준선
plt.axhline(-100, color='blue', linestyle='--', alpha=0.5, label='Oversold (-100)') # 과매도 기준선
plt.axhline(0, color='gray', linewidth=0.5)                # 중심선
plt.fill_between(data.index, 100, data['CCI'], where=(data['CCI'] >= 100), color='red', alpha=0.2)
plt.fill_between(data.index, -100, data['CCI'], where=(data['CCI'] <= -100), color='blue', alpha=0.2)
plt.title(f'{stockName} CCI Chart', fontsize=14)
plt.ylabel('CCI')
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=10))
plt.gca().yaxis.tick_right()
plt.gca().yaxis.set_label_position("right")
plt.xticks(rotation=90)
plt.legend()
plt.grid(True)
finish_plot('stochastic.png')
print("'CCI' : 가격이 평균과 얼마나 다른지 측정")
print("+100 이상 과매수, -100 이하 과매도 구간")
################################################################################################

################################################################################################
# Stochastic Oscillator 차트 그리기
plt.figure(figsize=(12,6))
plt.plot(data.index, data['%K'], color='blue', label='%K (Fast)', linewidth=1)
plt.plot(data.index, data['%D'], color='red', label='%D (Slow)', linewidth=1)
plt.axhline(80, color='red', linestyle='--', alpha=0.5, label='Overbought (80)')  # 과매수 기준선
plt.axhline(20, color='blue', linestyle='--', alpha=0.5, label='Oversold (20)') # 과매도 기준선
plt.title(f'{stockName} Stochastic Chart', fontsize=14)
plt.ylabel("Stochastic (%)")
plt.legend()
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=10))
plt.gca().yaxis.tick_right()
plt.gca().yaxis.set_label_position("right")
plt.xticks(rotation=90)
plt.grid(True, alpha=0.3)
finish_plot('fear_greed.png')
print("Stochastic' : 종가가 최근 N일 동안의 최고/최저 범위에서 어디에 위치하는지 확인")
print("80 이상 과매수, 20 이하 과매도")
print("20 이하 구간에서 %K(빠른선)가 %D(느린선)를 상향 돌파 시 '매수'")
print("80 이상 구간에서 %K(빠른선)가 %D(느린선)를 하향 돌파 시 '매도'")
################################################################################################

################################################################################################
# 공포탐욕지수 차트
def fetch_fng_timeseries(start_date='2024-01-01'):
    url = f"https://production.dataviz.cnn.io/index/fearandgreed/graphdata/{start_date}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/114.0.0.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Data requests failed: {response.status_code}")

    data = response.json()['fear_and_greed_historical']['data']
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['x'] // 1000, unit='s')
    df['fng'] = df['y']
    df = df[['date', 'fng']].set_index('date')
    return df

def get_fng_signal(df):
    end_date = df.index.max()
    start_date = end_date - timedelta(days=7)
    recent = df.loc[start_date:end_date]
    #recent = df.last('7D')

    avg = round(recent['fng'].mean(), 2)
    lastFng = round(df['fng'].iloc[-1], 2)

    if lastFng <= 25:
        return f'📉 Fear and buy timing, Index - {lastFng}' 
    elif lastFng <= 45:
        return f'📉 Fear zone, Index - {lastFng}'
    elif lastFng <= 55:
        return f'⏸ Neutral and do not move, Index - {lastFng}'
    elif lastFng < 75:
        return f'📈 Greed zone, Index - {lastFng}'
    else:
        return f'📈 Greed and sell timing, Index - {lastFng}'

# ✅ 데이터 가져오기
try:
    fng_df = fetch_fng_timeseries(start)
    fng_signal = get_fng_signal(fng_df)
    current_fng = float(fng_df["fng"].iloc[-1])
except Exception as e:
    print(f"[WARNING] Fear & Greed 데이터 조회 실패: {e}")
    fng_df = pd.DataFrame(columns=["fng"])
    fng_signal = "N/A"
    current_fng = None


# ✅ 시계열 차트 그리기
plt.figure(figsize=(12, 6))
plt.plot(fng_df.index, fng_df['fng'], label="Fear & Greed Index", color='purple', linewidth=2)
plt.axhline(50, color='gray', linestyle='--', label='Neutral (50)')
plt.axhline(25, color='blue', linestyle='--', label='Fear (25)')
plt.axhline(75, color='red', linestyle='--', label='Greed (75)')

plt.title("CNN Fear & Greed Index", fontsize=14)
plt.ylabel("Index (0~100)")
plt.legend()
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=10))
plt.gca().yaxis.tick_right()
plt.gca().yaxis.set_label_position("right")
plt.xticks(rotation=90)
plt.grid(True)
plt.tight_layout()
finish_plot('usdkrw.png')
################################################################################################

################################################################################################
# 원달러 환율 차트
plt.figure(figsize=(12,6))
plt.plot(dataDER.index, dataDER['Close'], label='USD/KRW Close', color='blue')

plt.title('USD/KRW - Daily Close', fontsize=16)
plt.ylabel('KRW per USD', fontsize=12)
plt.legend()
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=10))
plt.gca().yaxis.tick_right()
plt.gca().yaxis.set_label_position("right")
plt.xticks(rotation=90)
plt.grid(True)
plt.tight_layout()
finish_plot('dxy.png')
################################################################################################

################################################################################################
# 달러인덱스 차트
plt.figure(figsize=(12,6))
plt.plot(dataDI.index, dataDI['Close'], label='Dollar Index', color='blue')

plt.title('Dollar Index - Daily Close', fontsize=16)
plt.ylabel('Index Value per USD', fontsize=12)
plt.legend()
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=10))
plt.gca().yaxis.tick_right()
plt.gca().yaxis.set_label_position("right")
plt.xticks(rotation=90)
plt.grid(True)
plt.tight_layout()
plt.show()
################################################################################################

################################################################################################
# 매수/매도 시그널 표시
# 종가 표시
#if stockName == 'QQQ':
#    current_price = data['Close_Value'].iloc[-1]  # 마지막 종가
#    return_rate = ((current_price - averagePurchasePrice) / averagePurchasePrice) * 100

# 최근 5개 매수 시그널
recent_buy_signals = data[data['Buy_Signal']].iloc[-5:]
buy_conditions = {
    "RSI 과매도 (RSI < 30)": buyRsi,
    "Golden Cross": buyGoldenCross,
    "MACD Golden Cross": buyMacd,
    "볼린저 밴드 하락 이탈": buyBollinger,
    "CCI 과매도 (CCI < -100)": buyCci,
    "스토캐스틱 골든크로스": buyStoch,
    "OBV 상승 돌파": buyObv
}

#최근 5개 매도 시그널
recent_sell_signals = data[data['Sell_Signal']].iloc[-5:]
sell_conditions = {
    "RSI 과매수 (RSI > 70)": sellRsi,
    "Dead Cross": sellDeadCross,
    "MACD Dead Cross": sellMacd,
    "볼린저 밴드 상단 돌파": sellBollinger,
    "CCI 과매수 (CCI > 100)": sellCci,
    "스토캐스틱 데드크로스": sellStoch,
    "OBV 하락 이탈": sellObv
}
################################################################################################

################################################################################################
# Print Data
print("#"*130)
print(f"{stockName} Summary")
print(f"Closing Price        : {endPrice:.2f} USD")
#print(f"Avg Purchase Price   : {averagePurchasePrice:.2f} USD")
#print(f"Rate of Return       : {return_rate:.2f}%")
print(f"MaxDrawDown          : {maxDrawdown * 100:.2f}%")
print(f"Rate of Return       : {cagr * 100:.2f}%")
print(f"Calmar Ratio         : {calmarRatio:.2f}")
print(f"RSI                  : {data['RSI'].iloc[-1]:.2f}")
print(f"Fear and Greed Index : {fng_signal}")
print(f"Dollar Exchange Rate : {ClosedDER:.2f} 원" if ClosedDER is not None else "Dollar Exchange Rate : N/A")
print("#"*130)

for date, row in recent_buy_signals.iterrows():
    row_close_value = row['Close_Value']
    date_str = pd.to_datetime(date).strftime('%Y-%m-%d')
    print(f" 🔍 {date_str} 매수 신호 발생 이유:")
    for condition_name, condition in buy_conditions.items():
        if condition.loc[date]:  # 해당 조건이 True인 경우 출력
            print(f"    ✅ {condition_name}")
print("#"*130)
for date, row in recent_sell_signals.iterrows():
    row_close_value = row['Close_Value']
    date_str = pd.to_datetime(date).strftime('%Y-%m-%d')
    print(f" 🔍 {date_str} 매도 신호 발생 이유:")
    for condition_name, condition in sell_conditions.items():
        if condition.loc[date]:  # 해당 조건이 True인 경우 출력
            print(f"    ❌ {condition_name}")
print("#"*130)
################################################################################################
# endregion

################################################################################################
# 장기 ETF 투자용 QUANT DECISION
#
# 철학:
# 1) 매도보다 보유를 우선
# 2) 상승추세가 유지되는 동안 조정은 매수 기회로 평가
# 3) 공포/큰 낙폭에서는 Accumulation을 강화
# 4) 단순 과매수/과매도 지표만으로 매도하지 않음
################################################################################################

def safe_float(v):
    try:
        if v is None or pd.isna(v):
            return None
        return float(v)
    except Exception:
        return None

latest = data.iloc[-1]

close = safe_float(latest.get("Close_Value"))
ma50 = safe_float(latest.get("MA50"))
ma100 = safe_float(latest.get("MA100"))
ma200 = safe_float(latest.get("MA200"))
rsi = safe_float(latest.get("RSI"))
atr_pct = None

if len(data) >= 15:
    tr1 = data["High_Value"] - data["Low_Value"]
    tr2 = (data["High_Value"] - data["Close_Value"].shift(1)).abs()
    tr3 = (data["Low_Value"] - data["Close_Value"].shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(14).mean().iloc[-1]
    if close not in (None, 0):
        atr_pct = safe_float(atr / close * 100)

ma200_prev_20 = safe_float(data["MA200"].iloc[-21]) if len(data) >= 221 else None
ma200_slope = None
if ma200 is not None and ma200_prev_20 not in (None, 0):
    ma200_slope = (ma200 / ma200_prev_20 - 1) * 100

relative_strength = safe_float(latest.get("Relative_Strength"))
rs_ma20 = safe_float(latest.get("RS_MA20"))

# 현재 고점 대비 조정폭
rolling_peak = data["Close_Value"].cummax().iloc[-1]
current_drawdown = None
if rolling_peak not in (None, 0):
    current_drawdown = (close / rolling_peak) - 1

# 시장 Regime
trend_score = 0
trend_reasons = []

if close is not None and ma200 is not None and close > ma200:
    trend_score += 1
    trend_reasons.append("가격이 MA200 위")
if ma50 is not None and ma200 is not None and ma50 > ma200:
    trend_score += 1
    trend_reasons.append("MA50 > MA200")
if ma200_slope is not None and ma200_slope > 0:
    trend_score += 1
    trend_reasons.append("MA200 상승")
if relative_strength is not None and rs_ma20 is not None and relative_strength > rs_ma20:
    trend_score += 1
    trend_reasons.append("시장 대비 상대강도 우위")

if trend_score == 4:
    market_regime = "STRONG_BULL"
elif trend_score == 3:
    market_regime = "BULL"
elif trend_score == 2:
    market_regime = "NEUTRAL"
elif trend_score == 1:
    market_regime = "WEAK"
else:
    market_regime = "BEAR"

# Accumulation Score: "싸졌는가" + "공포인가" + "추세가 아직 살아있는가"
accumulation_score = 0
accumulation_reasons = []

if current_drawdown is not None:
    if current_drawdown <= -0.20:
        accumulation_score += 3
        accumulation_reasons.append("고점 대비 -20% 이상 조정")
    elif current_drawdown <= -0.15:
        accumulation_score += 2
        accumulation_reasons.append("고점 대비 -15% 이상 조정")
    elif current_drawdown <= -0.10:
        accumulation_score += 1
        accumulation_reasons.append("고점 대비 -10% 이상 조정")

if current_fng is not None:
    if current_fng <= 25:
        accumulation_score += 3
        accumulation_reasons.append("Fear & Greed 극단적 공포")
    elif current_fng <= 40:
        accumulation_score += 2
        accumulation_reasons.append("Fear & Greed 공포")
    elif current_fng < 50:
        accumulation_score += 1
        accumulation_reasons.append("Fear & Greed 약한 공포")

if rsi is not None:
    if rsi <= 30:
        accumulation_score += 2
        accumulation_reasons.append("RSI 과매도")
    elif rsi <= 40:
        accumulation_score += 1
        accumulation_reasons.append("RSI 약세권")

# 장기 추세가 살아 있을 때만 조정 매수의 질을 높게 평가
if market_regime in ("STRONG_BULL", "BULL"):
    accumulation_score += 1
    accumulation_reasons.append("장기 상승추세 유지")

# Risk Check
risk_flags = []

if close is not None and ma200 is not None and close < ma200:
    risk_flags.append("가격이 MA200 아래")
if ma200_slope is not None and ma200_slope < 0:
    risk_flags.append("MA200 하락")
if relative_strength is not None and rs_ma20 is not None and relative_strength < rs_ma20:
    risk_flags.append("시장 대비 상대강도 약화")

risk_check = "NORMAL"
if len(risk_flags) >= 2:
    risk_check = "THESIS REVIEW"
elif len(risk_flags) == 1:
    risk_check = "CAUTION"

# 최종 결정
# 매도 신호를 최종 의사결정에 사용하지 않는다.
# 장기 ETF 투자철학상 "HOLD"가 기본값이며, 공포/조정일 때만 매수 강도를 높인다.
if risk_check == "THESIS REVIEW":
    final_signal = "BUY PAUSE"
elif market_regime in ("STRONG_BULL", "BULL") and accumulation_score >= 6:
    final_signal = "STRONG ACCUMULATE"
elif market_regime in ("STRONG_BULL", "BULL") and accumulation_score >= 3:
    final_signal = "ACCUMULATE"
elif market_regime in ("STRONG_BULL", "BULL"):
    final_signal = "NORMAL DCA"
elif market_regime == "NEUTRAL" and accumulation_score >= 4:
    final_signal = "ACCUMULATE"
else:
    final_signal = "HOLD"

print()
print("#" * 130)
print("FINAL QUANTITATIVE INVESTMENT DECISION")
print("#" * 130)
print(f"Analysis Date         : {ny_today}")
print(f"Asset                 : {ticker_symbol}")
print(f"Benchmark             : {benchmark_symbol}")
print(f"Current Price         : {close:.2f}" if close is not None else "Current Price         : N/A")
print()
print("[ MARKET REGIME ]")
print(f"Market Regime         : {market_regime}")
print(f"Trend Score           : {trend_score} / 4")
for reason in trend_reasons:
    print(f"  - {reason}")
print()
print("[ ACCUMULATION ]")
print(f"Accumulation Score    : {accumulation_score}")
print(f"Current Drawdown      : {current_drawdown * 100:.2f}%" if current_drawdown is not None else "Current Drawdown      : N/A")
for reason in accumulation_reasons:
    print(f"  - {reason}")
print()
print("[ RISK ]")
print(f"Risk Check            : {risk_check}")
if risk_flags:
    for flag in risk_flags:
        print(f"  - {flag}")
else:
    print("  - 구조적 추세 훼손 신호 없음")
print()
print("[ KEY INDICATORS ]")
print(f"MA50                  : {ma50:.2f}" if ma50 is not None else "MA50                  : N/A")
print(f"MA100                 : {ma100:.2f}" if ma100 is not None else "MA100                 : N/A")
print(f"MA200                 : {ma200:.2f}" if ma200 is not None else "MA200                 : N/A")
print(f"MA200 Slope           : {ma200_slope:.2f}%" if ma200_slope is not None else "MA200 Slope           : N/A")
print(f"RSI                   : {rsi:.2f}" if rsi is not None else "RSI                   : N/A")
print(f"ATR %                 : {atr_pct:.2f}%" if atr_pct is not None else "ATR %                 : N/A")
print(f"Relative Strength     : {relative_strength:.4f}" if relative_strength is not None else "Relative Strength     : N/A")
print(f"Fear & Greed          : {current_fng:.2f}" if current_fng is not None else "Fear & Greed          : N/A")
print()
print(f"FINAL SIGNAL          : {final_signal}")
print()
print("해석 기준:")
print("  STRONG ACCUMULATE : 상승추세가 유지되면서 큰 조정/공포가 겹친 경우")
print("  ACCUMULATE        : 장기추세가 유지되고 조정 매수 근거가 충분한 경우")
print("  NORMAL DCA        : 상승추세는 양호하지만 적극적 매수 근거는 부족한 경우")
print("  HOLD              : 방향성이 애매하거나 매수 기대수익이 충분하지 않은 경우")
print("  BUY PAUSE         : 장기 추세 훼손 가능성이 있어 신규매수를 일시 중단하고 검토")
print()
print("주의: 기존의 단기 Sell Signal은 '매도 명령'이 아니라 과열/추세약화 참고 신호입니다.")
print("#" * 130)
