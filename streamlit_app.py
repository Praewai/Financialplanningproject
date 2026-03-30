import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import io
import csv
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib.units import inch

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(page_title="Thai Financial Planner", layout="wide")
st.title("Post Retirement Financial Planner")

# =========================================================
# DISCLAIMER
# =========================================================
@st.dialog("⚠️ Disclaimer (คำเตือน)")
def show_disclaimer():
  st.markdown("""
  This website was created by Financial Engineering Students. We are not Financial Planners or Investment Advisors, and we do not have access to any non-public information.

No Guarantee: We cannot guarantee that the simulation is 100% accurate. All calculations and projections are based on mathematical models and should be treated as estimates.

Intended Purpose: This tool was created solely for financial planners to use as assistance for rough estimation and is not to be used as a replacement for professional financial planning.

Non-Regulated: We are not regulated by any Financial Services Authority.

Privacy: We do not store, collect, or monitor any personal data or financial information entered into this application. All data is processed in real-time and is cleared once the session ends. 
        
เว็บไซต์นี้จัดทำขึ้นโดยนักศึกษาภาควิชาวิศวกรรมการเงิน (Financial Engineering) ไม่ใช่ผู้วางแผนการเงิน (Financial Planner) หรือที่ปรึกษาการลงทุน (Investment Advisor) และผู้จัดทำไม่ได้มีการเข้าถึงข้อมูลภายใน (Non-public information) ใดๆ ทั้งสิ้น

การรับรองผล: เราไม่สามารถรับรองได้ว่าผลจากการจำลอง (Simulation) จะถูกต้องแม่นยำ 100% ข้อมูลการคาดการณ์ทั้งหมดเป็นเพียงการประมาณการเชิงคณิตศาสตร์เท่านั้น

วัตถุประสงค์: เครื่องมือนี้ถูกสร้างขึ้นเพื่อใช้เป็นเครื่องมือช่วยคำนวณเบื้องต้นสำหรับผู้วางแผนการเงินเท่านั้น ไม่ควรนำไปใช้ทดแทนการวางแผนการเงินแบบเต็มรูปแบบโดยผู้เชี่ยวชาญ

การกำกับดูแล: ผู้จัดทำไม่ได้อยู่ภายใต้การกำกับดูแลของหน่วยงานกำกับดูแลบริการทางการเงินใดๆ

นโยบายความเป็นส่วนตัว: เราไม่มีการจัดเก็บ บันทึก หรือเข้าถึงข้อมูลส่วนบุคคลและข้อมูลทางการเงินที่คุณกรอกเข้าสู่ระบบ ข้อมูลทั้งหมดจะถูกลบออกจากหน่วยความจำทันทีเมื่อเสร็จสิ้นการใช้งานหรือปิดหน้าเว็บไซต์""")
  
  if st.button("I accepted (ยอมรับ)"):
    st.rerun()


if "accepted_terms" not in st.session_state:
    show_disclaimer()
    st.session_state["accepted_terms"] = True

# =========================================================
# CORE SIMULATION ENGINE
# =========================================================
class RetirementSimulator:
    def __init__(self):
        # Life expectancy table (age: remaining years)
        self.life_expectancy = {
            60: 24, 61: 23, 62: 22, 63: 21, 64: 20, 65: 19, 66: 18, 67: 17,
            68: 16, 69: 15, 70: 14, 71: 13, 72: 12, 73: 11, 74: 10, 75: 9,
            76: 8, 77: 7, 78: 6, 79: 5, 80: 4, 81: 3, 82: 2, 83: 1, 84: 1
        }

    def get_life_expectancy(self, current_age):
        """Get remaining life expectancy for given age."""
        if current_age in self.life_expectancy:
            return self.life_expectancy[current_age]
        elif current_age > max(self.life_expectancy.keys()):
            return 1
        else:
            return max(self.life_expectancy.values())

    def simulate_returns(self, portfolio_allocation, asset_stats, n_simulations, n_years):
        """
        Simulate annual portfolio returns based on asset allocation.
        One return is drawn per year per simulation path (annual sampling).
        Correlation of 0.4 is applied between all asset pairs.
        """
        assets_list = [k for k in portfolio_allocation.keys()
                       if k in asset_stats and portfolio_allocation[k] > 0]
        if len(assets_list) == 0:
            return np.zeros((n_simulations, n_years))

        weights = np.array([portfolio_allocation[a] for a in assets_list], dtype=float)
        weights = weights / weights.sum()

        means = np.array([asset_stats[a]["mean"] for a in assets_list], dtype=float)
        stds  = np.array([asset_stats[a]["std"]  for a in assets_list], dtype=float)

        n_assets = len(assets_list)
        corr = np.eye(n_assets) + 0.4 * (np.ones((n_assets, n_assets)) - np.eye(n_assets))
        cov  = np.outer(stds, stds) * corr

        portfolio_returns = np.zeros((n_simulations, n_years))
        for sim in range(n_simulations):
            asset_returns = np.random.multivariate_normal(means, cov, n_years)
            portfolio_returns[sim] = asset_returns @ weights

        return portfolio_returns

    # ------------------------------------------------------------------
    # BUG FIX 8: build_cashflow_schedule now accepts timed_income_items
    # so expiring income sources (pension, rental, etc.) are correctly
    # removed from the net-withdrawal schedule when their term ends.
    # ------------------------------------------------------------------
    @staticmethod
    def build_cashflow_schedule(
        base_annual_expense: float,
        base_annual_income: float,
        timed_items: list,
        total_years: int,
        inflation_rate: float,
        withdrawal_rate: float = 0.0,
        initial_portfolio: float = 0.0,
        timed_income_items: list = None,   # BUG FIX 8: new parameter
    ) -> np.ndarray:
        """
        Returns array shape (total_years,) with inflation-adjusted net
        withdrawal per year.  Timed expense items use integer months_remaining
        so a 5-month debt is correctly prorated in its partial final year.
        Timed income items work identically but reduce income when they expire.
        """
        if timed_income_items is None:
            timed_income_items = []

        schedule = np.zeros(total_years)
        for yr in range(total_years):
            # --- Expense side: remove expired timed expense items ---
            timed_expense_reduction = 0.0
            for item in timed_items:
                months_rem = item["months_remaining"]
                annual_amt = item["annual_amount"]
                months_elapsed = yr * 12
                if months_elapsed >= months_rem:
                    timed_expense_reduction += annual_amt
                elif months_elapsed + 12 > months_rem:
                    months_active = months_rem - months_elapsed
                    timed_expense_reduction += annual_amt * (1 - months_active / 12.0)

            net_expense = (base_annual_expense - timed_expense_reduction) * (1 + inflation_rate) ** yr

            # --- Income side: remove expired timed income items (BUG FIX 8) ---
            timed_income_reduction = 0.0
            for item in timed_income_items:
                months_rem = item["months_remaining"]
                annual_amt = item["annual_amount"]
                months_elapsed = yr * 12
                if months_elapsed >= months_rem:
                    timed_income_reduction += annual_amt
                elif months_elapsed + 12 > months_rem:
                    months_active = months_rem - months_elapsed
                    timed_income_reduction += annual_amt * (1 - months_active / 12.0)

            net_income = (base_annual_income - timed_income_reduction) * (1 + inflation_rate) ** yr

            schedule[yr] = max(0.0, net_expense - net_income)
        return schedule

    # ------------------------------------------------------------------
    # WITHDRAWAL STRATEGIES
    # ------------------------------------------------------------------

    def _resolve_withdrawal(self, portfolio_value, year, base_wd,
                            withdrawal_rate, cashflow_schedule):
        """
        Determine the actual withdrawal for this year.
        BUG FIX 4: only apply rate_wd when withdrawal_rate > 0 AND
        portfolio_value > 0 to avoid divide-by-zero / incorrect zero-rate logic.
        """
        if cashflow_schedule is not None:
            cs_wd = cashflow_schedule[year] if year < len(cashflow_schedule) else 0.0
            # BUG FIX 4: guard withdrawal_rate > 0 check was already there,
            # but also guard portfolio_value > 0
            rate_wd = (portfolio_value * withdrawal_rate
                       if withdrawal_rate > 0 and portfolio_value > 0 else 0.0)
            return max(cs_wd, rate_wd)
        return base_wd

    def basic_strategy(self, initial_portfolio, withdrawal_rate, inflation_rate,
                       returns, years, cashflow_schedule=None):
        portfolio_value  = initial_portfolio
        withdrawal       = initial_portfolio * withdrawal_rate
        balances         = [portfolio_value]
        withdrawals      = []

        for year in range(years):
            wd = self._resolve_withdrawal(portfolio_value, year,
                                          withdrawal, withdrawal_rate, cashflow_schedule)
            withdrawals.append(max(0.0, wd))

            portfolio_value -= wd

            if portfolio_value <= 0:
                balances.extend([0.0] * (years - year))
                withdrawals.extend([0.0] * (years - 1 - year))
                break

            portfolio_value *= (1 + returns[year])
            balances.append(max(0.0, portfolio_value))

            if cashflow_schedule is None:
                withdrawal *= (1 + inflation_rate)

        return balances, withdrawals

    def forgoing_inflation_strategy(self, initial_portfolio, withdrawal_rate,
                                    inflation_rate, returns, years,
                                    cashflow_schedule=None):
        portfolio_value  = initial_portfolio
        withdrawal       = initial_portfolio * withdrawal_rate
        previous_balance = portfolio_value
        balances         = [portfolio_value]
        withdrawals      = []

        for year in range(years):
            wd = self._resolve_withdrawal(portfolio_value, year,
                                          withdrawal, withdrawal_rate, cashflow_schedule)
            withdrawals.append(max(0.0, wd))

            portfolio_value -= wd

            if portfolio_value <= 0:
                balances.extend([0.0] * (years - year))
                withdrawals.extend([0.0] * (years - 1 - year))
                break

            portfolio_value *= (1 + returns[year])
            balances.append(max(0.0, portfolio_value))

            if cashflow_schedule is None:
                if portfolio_value > previous_balance:
                    withdrawal *= (1 + inflation_rate)
                previous_balance = portfolio_value

        return balances, withdrawals

    def rmd_strategy(self, initial_portfolio, starting_age, returns, years,
                     cashflow_schedule=None, withdrawal_rate=0.0, inflation_rate=0.03):
        portfolio_value = initial_portfolio
        current_age     = starting_age
        balances        = [portfolio_value]
        withdrawals     = []

        for year in range(years):
            life_exp = self.get_life_expectancy(current_age)
            wd_rmd   = portfolio_value / life_exp if life_exp > 0 else portfolio_value

            if cashflow_schedule is not None:
                cs_wd = cashflow_schedule[year] if year < len(cashflow_schedule) else 0.0
                wd = max(wd_rmd, cs_wd)
            else:
                wd = wd_rmd

            withdrawals.append(max(0.0, wd))

            portfolio_value -= wd

            if portfolio_value <= 0:
                balances.extend([0.0] * (years - year))
                withdrawals.extend([0.0] * (years - 1 - year))
                break

            portfolio_value *= (1 + returns[year])
            balances.append(max(0.0, portfolio_value))
            current_age += 1

        return balances, withdrawals

    def guardrails_strategy(self, initial_portfolio, withdrawal_rate,
                            inflation_rate, returns, years,
                            cashflow_schedule=None):
        portfolio_value  = initial_portfolio
        withdrawal       = initial_portfolio * withdrawal_rate
        initial_rate     = withdrawal_rate
        balances         = [portfolio_value]
        withdrawals      = []

        for year in range(years):
            wd = self._resolve_withdrawal(portfolio_value, year,
                                          withdrawal, withdrawal_rate, cashflow_schedule)
            withdrawals.append(max(0.0, wd))

            portfolio_value -= wd

            if portfolio_value <= 0:
                balances.extend([0.0] * (years - year))
                withdrawals.extend([0.0] * (years - 1 - year))
                break

            portfolio_value *= (1 + returns[year])
            balances.append(max(0.0, portfolio_value))

            if cashflow_schedule is None:
                current_rate = withdrawal / portfolio_value if portfolio_value > 0 else 0.0
                if current_rate < initial_rate * 0.80:
                    withdrawal *= 1.10
                elif current_rate > initial_rate * 1.20:
                    withdrawal *= 0.90
                else:
                    withdrawal *= (1 + inflation_rate)

        return balances, withdrawals

    # ------------------------------------------------------------------
    # run_simulation
    # ------------------------------------------------------------------
    def run_simulation(
        self, initial_portfolio, portfolio_allocation, asset_stats,
        withdrawal_strategy, withdrawal_rate, years, inflation_rate,
        starting_age, inheritance_goal=0.0, returns_override=None,
        n_simulations=50000, cashflow_schedule=None,
    ):
        returns = (returns_override if returns_override is not None
                   else self.simulate_returns(portfolio_allocation, asset_stats,
                                              n_simulations, years))

        strategy_map = {
            "Basic Strategy":     self.basic_strategy,
            "Forgoing Inflation": self.forgoing_inflation_strategy,
            "RMD Strategy":       self.rmd_strategy,
            "Guardrails":         self.guardrails_strategy,
        }

        all_balances    = []
        all_withdrawals = []

        for sim in range(n_simulations):
            if withdrawal_strategy == "RMD Strategy":
                balances, wds = self.rmd_strategy(
                    initial_portfolio, starting_age, returns[sim], years,
                    cashflow_schedule=cashflow_schedule,
                    withdrawal_rate=withdrawal_rate,
                    inflation_rate=inflation_rate)
            else:
                balances, wds = strategy_map[withdrawal_strategy](
                    initial_portfolio, withdrawal_rate, inflation_rate,
                    returns[sim], years,
                    cashflow_schedule=cashflow_schedule)

            if len(balances) < years + 1:
                balances = balances + [0.0] * ((years + 1) - len(balances))
            else:
                balances = balances[:years + 1]
            if len(wds) < years:
                wds = wds + [0.0] * (years - len(wds))
            else:
                wds = wds[:years]

            all_balances.append(balances)
            all_withdrawals.append(wds)

        all_balances    = np.array(all_balances,    dtype=float)
        all_withdrawals = np.array(all_withdrawals, dtype=float)
        final_values    = all_balances[:, -1]

        # AFTER:
        if inheritance_goal > 0:
            surviving_mask = final_values > 0
            if surviving_mask.sum() > 0:
                inh_rate = float(np.mean(final_values[surviving_mask] >= inheritance_goal))
            else:
                inh_rate = 0.0
        else:
            inh_rate = -1.0

        surviving_finals = final_values[final_values > 0]
        median_surviving = float(np.median(surviving_finals)) if len(surviving_finals) > 0 else 0.0

        return {
            "survival_rate":            float(np.mean(final_values > 0)),
            "inheritance_success_rate": inh_rate,
            "median_balance":    np.median(all_balances,     axis=0),
            "median_surviving":  median_surviving,
            "percentile_10":     np.percentile(all_balances,  10, axis=0),
            "percentile_25":     np.percentile(all_balances,  25, axis=0),
            "percentile_75":     np.percentile(all_balances,  75, axis=0),
            "percentile_90":     np.percentile(all_balances,  90, axis=0),
            "returns_mean":      float(np.mean(returns)),
            "median_withdrawal": np.median(all_withdrawals,  axis=0),
            "withdrawal_p10":    np.percentile(all_withdrawals, 10, axis=0),
            "withdrawal_p90":    np.percentile(all_withdrawals, 90, axis=0),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def recommend_improvements(self, current_survival_rate, portfolio_allocation,
                               withdrawal_rate, min_survival_rate=0.85):
        recs = []
        if current_survival_rate >= min_survival_rate:
            return ["✅ Your strategy meets the target survival rate!"]
        if withdrawal_rate > 0.03:
            recommended = withdrawal_rate * 0.9
            recs.append(f"📉 **Reduce Spending:** Try lowering withdrawal from "
                        f"{withdrawal_rate*100:.1f}% to {recommended*100:.1f}%.")
        equity_keys = ["pct_seti", "pct_msci_stock", "pct_REIT", "pct_msci_reit"]
        equity_weight = sum(float(portfolio_allocation.get(k, 0)) for k in equity_keys)
        if equity_weight < 0.4:
            recs.append(f"📈 **Increase Growth:** Equity allocation seems low "
                        f"({equity_weight*100:.0f}%). Consider 40–60%.")
        elif equity_weight > 0.8:
            recs.append(f"🛡️ **Reduce Risk:** Equity allocation seems high "
                        f"({equity_weight*100:.0f}%). Consider adding bonds/cash.")
        recs.append("🔄 **Change Strategy:** Try 'Guardrails' or 'Forgoing Inflation' "
                    "to adapt spending automatically during market downturns.")
        deficit = min_survival_rate - current_survival_rate
        recs.append(f"💰 **Increase Portfolio:** Gap to target ≈ {deficit*100:.1f}%. "
                    "Consider increasing initial savings or delaying retirement.")
        return recs

    def find_optimal_withdrawal_rate(
        self, initial_portfolio, portfolio_allocation, asset_stats,
        withdrawal_strategy, initial_rate, years, inflation_rate, starting_age,
        inheritance_goal=0.0, min_survival_rate=0.85, n_simulations=50000,
        cashflow_schedule=None,
    ):
        low_rate       = 0.01
        high_rate      = min(0.12, max(0.06, initial_rate * 2))
        tolerance      = 0.001
        best_rate      = initial_rate
        max_iterations = 20

        for _ in range(max_iterations):
            if (high_rate - low_rate) <= tolerance:
                break
            test_rate = (low_rate + high_rate) / 2
            results = self.run_simulation(
                initial_portfolio, portfolio_allocation, asset_stats,
                withdrawal_strategy, test_rate, years, inflation_rate,
                starting_age, inheritance_goal=inheritance_goal,
                n_simulations=n_simulations, cashflow_schedule=cashflow_schedule)
            if results["survival_rate"] >= min_survival_rate:
                best_rate = test_rate
                low_rate  = test_rate
            else:
                high_rate = test_rate

        return best_rate

# =========================================================
# UI HELPERS
# =========================================================
if "current_step" not in st.session_state:
    st.session_state["current_step"] = 0

steps = ["👤 1. ข้อมูลผู้ใช้","🧩 2.แบบประเมินความเสี่ยง","📊 3.การจัดสรรสินทรัพย์","💸 4. กลยุทธ์การถอนเงิน"]

def update_nav(): st.session_state["nav_radio"] = steps[st.session_state["current_step"]]
def next_step():
    if st.session_state["current_step"] < len(steps)-1:
        st.session_state["current_step"] += 1; update_nav()
def prev_step():
    if st.session_state["current_step"] > 0:
        st.session_state["current_step"] -= 1; update_nav()
def jump_step():
    st.session_state["current_step"] = steps.index(st.session_state["nav_radio"])

def money_input(label, default_val, key_suffix):
    data_key = f"v_{key_suffix}"; fmt_key = f"fmt_{key_suffix}"; ui_key = f"ui_{key_suffix}"
    if data_key not in st.session_state:
        val = float(default_val)
        st.session_state[data_key] = val; st.session_state[fmt_key] = f"{val:,.0f}"
    def on_change():
        user_input = st.session_state.get(ui_key, "0")
        try: clean_val = float(str(user_input).replace(",","").strip())
        except: clean_val = 0.0
        st.session_state[data_key] = clean_val
        fmt = f"{clean_val:,.0f}"
        st.session_state[fmt_key] = fmt; st.session_state[ui_key] = fmt
    st.text_input(label, value=st.session_state[fmt_key], key=ui_key, on_change=on_change)
    return st.session_state[data_key]

def get_num(key_suffix):
    return float(st.session_state.get(f"v_{key_suffix}", 0.0))

def timed_expense_input(label, key_suffix, show_timer=True, timer_label="จ่ายอีก"):
    freq_key  = f"freq_{key_suffix}"
    timer_key = f"timer_on_{key_suffix}"
    dur_key   = f"dur_{key_suffix}"
    unit_key  = f"unit_{key_suffix}"

    options_freq = ["ต่อเดือน (Monthly)", "ต่อปี (Yearly)"]
    options_unit = ["เดือน (Months)", "ปี (Years)"]

    if freq_key  not in st.session_state: st.session_state[freq_key]  = options_freq[0]
    if timer_key not in st.session_state: st.session_state[timer_key] = False
    if dur_key   not in st.session_state: st.session_state[dur_key]   = 12
    if unit_key  not in st.session_state: st.session_state[unit_key]  = "เดือน (Months)"

    col_lbl, col_amt, col_freq, col_toggle = st.columns([2,2,2,1], vertical_alignment="center")
    with col_lbl:   st.markdown(f"**{label}**")
    with col_amt:   amount = money_input("", 0, key_suffix)
    with col_freq:
        cur_idx = options_freq.index(st.session_state[freq_key]) if st.session_state[freq_key] in options_freq else 0
        st.radio("", options_freq, index=cur_idx, horizontal=True,
                 key=f"wfr_{freq_key}", label_visibility="collapsed",
                 on_change=lambda: st.session_state.update({freq_key: st.session_state[f"wfr_{freq_key}"]}))
    if show_timer:
        with col_toggle:
            st.checkbox("⏱", value=st.session_state[timer_key], key=f"wtm_{timer_key}",
                        help="กำหนดระยะเวลาที่เหลือ",
                        on_change=lambda: st.session_state.update({timer_key: st.session_state[f"wtm_{timer_key}"]}))

    months_remaining = None
    if show_timer and st.session_state[timer_key]:
        _, c_lbl2, c_val, c_unit, _ = st.columns([0.3,1.2,1,1.5,1])
        with c_lbl2: st.markdown(f"↳ {timer_label}")
        with c_val:
            st.number_input("", min_value=1, max_value=600,
                            value=int(st.session_state[dur_key]),
                            key=f"wdv_{dur_key}", label_visibility="collapsed",
                            on_change=lambda: st.session_state.update({dur_key: st.session_state[f"wdv_{dur_key}"]}))
        with c_unit:
            cur_unit_idx = options_unit.index(st.session_state[unit_key]) if st.session_state[unit_key] in options_unit else 0
            st.radio("", options_unit, index=cur_unit_idx, horizontal=False,
                     key=f"wun_{unit_key}", label_visibility="collapsed",
                     on_change=lambda: st.session_state.update({unit_key: st.session_state[f"wun_{unit_key}"]}))
        if "เดือน" in st.session_state[unit_key]:
            months_remaining = int(st.session_state[dur_key])
        else:
            months_remaining = int(st.session_state[dur_key]) * 12

    freq = st.session_state[freq_key]
    monthly_amount = float(amount)
    if "Monthly" in freq:
        annual_amount = monthly_amount * 12
    else:
        annual_amount = monthly_amount
        monthly_amount = monthly_amount / 12.0

    return annual_amount, monthly_amount, months_remaining


def cashflow_input(label, key_suffix):
    options  = ["ต่อเดือน (Monthly)", "ต่อปี (Yearly)"]
    freq_key = f"freq_{key_suffix}"
    if freq_key not in st.session_state: st.session_state[freq_key] = options[0]
    def on_change(): st.session_state[freq_key] = st.session_state[f"wcf_{freq_key}"]
    c_lbl, c_inp, c_frq = st.columns([2,2,2], vertical_alignment="center")
    with c_lbl: st.markdown(f"{label}")
    with c_inp: amount = money_input("", 0, key_suffix)
    with c_frq:
        cur_idx = options.index(st.session_state[freq_key]) if st.session_state[freq_key] in options else 0
        st.radio("", options, index=cur_idx, horizontal=True,
                 key=f"wcf_{freq_key}", on_change=on_change, label_visibility="collapsed")
    freq = st.session_state[freq_key]
    return float(amount * 12) if "Monthly" in freq else float(amount)


def get_annual_safe(key_suffix):
    val = st.session_state.get(f"v_{key_suffix}")
    if val is None: val = st.session_state.get(key_suffix, 0.0)
    try: val = float(val)
    except: val = 0.0
    freq = st.session_state.get(f"freq_{key_suffix}", "ต่อปี (Yearly)")
    if freq and ("Monthly" in str(freq) or "เดือน" in str(freq)):
        return val * 12
    return val


def graph_with_info(fig, title: str, info_text: str, key: str):
    """Renders fig then shows an expandable ℹ️ info box below it."""
    st.pyplot(fig)
    with st.expander(f"ℹ️ วิธีอ่านกราฟ: {title}", expanded=False):
        st.markdown(info_text)
    plt.close(fig)


# =========================================================
# EXPORT HELPERS
# =========================================================
def build_full_report_csv(export_data, res, alloc, years=30):
    def fnum(x, nd=2):
        try: return f"{float(x or 0):,.{nd}f}"
        except: return "0.00"
    def fpct(x, nd=2):
        try: return f"{float(x or 0)*100:.{nd}f}%"
        except: return ""
    def to_int(x, default=60):
        try: return int(float(x))
        except: return default

    ASSET_LABELS = {
        "pct_deposit":"Fixed Deposit","pct_gov_bond":"Thai Gov Bond",
        "pct_seti":"SET Index","pct_REIT":"Thai REIT",
        "pct_msci_stock":"MSCI World Equity","pct_msci_gov_bond":"MSCI Gov Bond",
        "pct_gold":"Gold","pct_msci_reit":"Global REIT",
    }
    rows = []
    rows.append(["SECTION","FIELD","VALUE"])
    rows.append(["PROFILE","Name",export_data.get("name","ไม่ระบุ")])
    rows.append(["PROFILE","Retire Age",export_data.get("retire_age","")])
    rows.append(["PROFILE","Life Expectancy",export_data.get("life_exp","")])
    rows.append(["PROFILE","Inheritance Goal (THB)",fnum(export_data.get("inheritance_goal"),2)])
    rows.append(["SETTINGS","Inflation",fpct(export_data.get("inflation"))])
    rows.append([])

    rows.append(["SECTION","INCOME ITEM","YEARLY (THB)","MONTHLY (THB)"])
    for item, val in export_data.get("inc_detail",{}).items():
        if val > 0: rows.append(["INCOME_DETAIL",item,fnum(val),fnum(val/12)])
    rows.append(["INCOME_SUMMARY","TOTAL INCOME",
                 fnum(export_data.get("total_income",0)),
                 fnum(export_data.get("total_income",0)/12)])
    rows.append([])

    rows.append(["SECTION","EXPENSE ITEM","YEARLY (THB)","MONTHLY (THB)","MONTHS REMAINING"])
    for item, info in export_data.get("exp_fixed_detail",{}).items():
        if isinstance(info, dict):
            amt = info.get("amount",0); mo = info.get("months_remaining",None)
            mo_str = str(mo) if mo is not None else "Permanent"
            if amt > 0: rows.append(["EXPENSE_FIXED",item,fnum(amt),fnum(amt/12),mo_str])
        elif info > 0:
            rows.append(["EXPENSE_FIXED",item,fnum(info),fnum(info/12),"Permanent"])
    for item, val in export_data.get("exp_var_detail",{}).items():
        if val > 0: rows.append(["EXPENSE_VAR",item,fnum(val),fnum(val/12),"Permanent"])
    rows.append(["EXPENSE_SUMMARY","TOTAL EXPENSE",
                 fnum(export_data.get("total_expense",0)),
                 fnum(export_data.get("total_expense",0)/12),""])
    rows.append([])

    rows.append(["SECTION","ASSET/DEBT","VALUE (THB)","MONTHS REMAINING"])
    for item, val in export_data.get("asset_detail",{}).items():
        if val > 0: rows.append(["ASSET",item,fnum(val),""])
    rows.append(["ASSET_TOTAL","INVESTABLE ASSETS",fnum(export_data.get("investable",0)),""])
    for item, info in export_data.get("debt_detail",{}).items():
        if isinstance(info, dict):
            amt = info.get("amount",0); mo = info.get("months_remaining",None)
            mo_str = str(mo) if mo is not None else "Permanent"
            if amt > 0: rows.append(["DEBT",item,fnum(amt),mo_str])
        elif info > 0: rows.append(["DEBT",item,fnum(info),"Permanent"])
    rows.append(["DEBT_TOTAL","TOTAL DEBT",fnum(export_data.get("total_debt",0)),""])
    rows.append([])

    sim_strat = export_data.get("sim_strat","-")
    wd_rate   = export_data.get("wd_rate",None)
    rows.append(["SIMULATION","Strategy",sim_strat])
    if wd_rate is not None: rows.append(["SIMULATION","Withdrawal Rate",fpct(wd_rate)])
    if res is not None:
        rows.append(["SIMULATION","Survival Rate",f"{res['survival_rate']*100:.1f}%"])
        rows.append(["SIMULATION","Median End Balance",fnum(res["median_balance"][-1],0)])
    rows.append([])

    rows.append(["SECTION","ASSET ALLOCATION","WEIGHT (%)"])
    if alloc:
        for k,v in alloc.items():
            rows.append(["ALLOCATION",ASSET_LABELS.get(k,k),f"{float(v)*100:.2f}%"])
    rows.append([])

    rows.append(["YEARLY PROJECTION"])
    rows.append(["Year","Age","Median_Balance","P10_Balance","P90_Balance",
                 "Median_Withdrawal","P10_Withdrawal","P90_Withdrawal","P10_Depleted"])
    retire_age_int = to_int(export_data.get("retire_age"),60)
    if res is not None:
        mb=res["median_balance"]; p10b=res["percentile_10"]; p90b=res["percentile_90"]
        mw=res["median_withdrawal"]; p10w=res["withdrawal_p10"]; p90w=res["withdrawal_p90"]
        for y in range(1, years+1):
            age = retire_age_int+(y-1)
            rows.append([y,age,round(mb[y],2),round(p10b[y],2),round(p90b[y],2),
                         round(mw[y-1],2),round(p10w[y-1],2),round(p90w[y-1],2),
                         1 if p10b[y]<=0 else 0])
    out = io.StringIO()
    csv.writer(out).writerows(rows)
    return out.getvalue().encode("utf-8-sig")


# BUG FIX 6: build_pdf_bytes always returns bytes — moved buffer.getvalue()
# outside the "if res is not None" block so it never returns None implicitly.
def build_pdf_bytes(data, res):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    FONT_REG="Helvetica"; FONT_BOLD="Helvetica-Bold"
    S_T=16; S_S=12; S_L=10; S_B=10

    # ---- Cover page ----
    c.setFont(FONT_BOLD, 25)
    c.drawCentredString(width/2, height/2+50, "Retirement Financial Planning Report")
    c.setFont(FONT_REG, S_S)
    c.drawString(1*inch, 1.5*inch, f"Customer: {data.get('name')}")
    c.showPage()

    # ---- Disclaimer page ----
    c.setFont(FONT_BOLD, S_S); c.drawCentredString(width/2, height-1*inch, "Disclaimer / Warning")
    c.setFont(FONT_REG, S_B)
    for i, line in enumerate([
        "This tool was created by Financial Engineering Students, not a regulated financial advisor.",
        "Results are simulations only and cannot guarantee future outcomes.",
        "Use this as rough estimation assistance only."
    ]):
        c.drawCentredString(width/2, height-1.5*inch - i*20, line)
    c.showPage()

    # ---- Table of Contents ----
    c.setFont(FONT_BOLD, S_S); c.drawString(1*inch, height-1*inch, "Table of Contents")
    c.setFont(FONT_REG, S_L)
    c.drawString(1.2*inch, height-1.6*inch, "1. Financial Health .................................................. Page 1")
    c.drawString(1.2*inch, height-1.9*inch, "2. Asset Allocation & Simulation Result ................. Page 2")
    c.showPage()

    # ---- Page 1: Financial Health ----
    c.setFont(FONT_BOLD, S_T); c.drawString(50, height-50, "Financial Planning: Financial Health")
    y = height-90
    c.setFont(FONT_BOLD, S_S); c.drawString(50, y, "A. Personal Information")
    c.setFont(FONT_REG, S_B); y -= 25
    c.drawString(60, y, f"Name: {data.get('name','N/A')}")
    c.drawString(250, y, f"Retire Age: {data.get('retire_age')} Years")
    c.drawString(400, y, f"Life Expectancy: {data.get('life_exp')} Years")
    y -= 22; c.drawString(60, y, f"Inheritance Goal: {data.get('inheritance_goal', 0):,.2f} THB")

    y -= 45; c.setFont(FONT_BOLD, S_S); c.drawString(50, y, "B. Debt Summary")
    c.setFont(FONT_REG, S_B); yc = y-25; has_debt = False
    for k, info in data.get("debt_detail", {}).items():
        amt = info.get("amount", 0) if isinstance(info, dict) else info
        mo  = info.get("months_remaining", None) if isinstance(info, dict) else None
        if amt > 0:
            mo_str = f" (pay for {mo} months)" if mo else ""
            c.drawString(60, yc, f"- {k}: {amt:,.2f} THB{mo_str}"); yc -= 22; has_debt = True
    if not has_debt:
        c.drawString(60, yc, "- No outstanding debt"); yc -= 22
    yd = yc-15; c.setFont(FONT_BOLD, S_B)
    c.drawString(60, yd, f"Total Liabilities: {data.get('total_debt', 0):,.2f} THB")

    y = yd-45; c.setFont(FONT_BOLD, S_S); c.drawString(50, y, "C. Post-Retirement Cash Flow (Annual)")
    xl = 60; xr = 320; cy = y-25; sy = cy
    c.setFont(FONT_BOLD, S_L); c.drawString(xl, cy, "Income Sources")
    c.setFont(FONT_REG, S_B); yi = cy
    for k, v in data.get("inc_detail", {}).items():
        if v > 0: yi -= 20; c.drawString(xl+15, yi, f"- {k}: {v:,.0f} THB")
    c.setFont(FONT_BOLD, S_L); c.drawString(xr, sy, "Expenses Breakdown")
    c.setFont(FONT_REG, S_B); ye = sy
    all_exp = {**data.get("exp_fixed_detail", {}), **data.get("exp_var_detail", {})}
    for k, v in all_exp.items():
        amt = v.get("amount", 0) if isinstance(v, dict) else v
        if amt > 0: ye -= 20; c.drawString(xr+15, ye, f"- {k}: {amt:,.0f} THB")
    total_inc = sum(data.get("inc_detail", {}).values())
    total_exp = 0
    for v in list(data.get("exp_fixed_detail", {}).values()) + list(data.get("exp_var_detail", {}).values()):
        total_exp += v.get("amount", 0) if isinstance(v, dict) else v
    net_flow = total_inc - total_exp
    y = min(yi, ye) - 40
    c.setStrokeColorRGB(0.8, 0.8, 0.8); c.line(60, y+15, 535, y+15)
    c.setFont(FONT_BOLD, S_B); c.setFillColorRGB(0, 0, 0)
    c.drawString(60, y, "Total Income:"); c.drawRightString(280, y, f"{total_inc:,.0f} THB")
    c.drawString(320, y, "Total Expenses:"); c.drawRightString(535, y, f"{total_exp:,.0f} THB")
    y -= 60; c.setFont(FONT_BOLD, S_S); c.drawString(50, y, "Financial Health Summary")
    y -= 30; c.setFont(FONT_REG, S_B)
    c.drawString(60, y, f"Investable Assets: {data.get('investable', 0):,.2f} THB")
    if net_flow < 0: c.setFillColorRGB(0.8, 0, 0)
    c.drawRightString(535, y, f"Net Cashflow/Year: {net_flow:,.2f} THB"); y -= 22
    c.drawRightString(535, y, f"Net Cashflow/Month: {net_flow/12:,.2f} THB")
    c.setFillColorRGB(0, 0, 0)
    c.showPage()

    # ---- Page 2: Asset Allocation & Simulation ----
    yt = height-50; c.setFont(FONT_BOLD, S_T)
    c.drawString(50, yt, "Asset Allocation & Simulation Result")

    # FIX 1 & 2: Use allocation weights (pct_*) not raw THB values for the pie chart
    ASSET_LABELS = {
        "pct_deposit": "Fixed Deposit", "pct_bond": "Thai Gov Bond",
        "pct_seti": "SET Index", "pct_REIT": "Thai REIT",
        "pct_msci_stock": "MSCI World Equity", "pct_msci_bond": "MSCI Gov Bond",
        "pct_gold": "Gold", "pct_msci_reit": "Global REIT",
    }
    alloc_weights = data.get("alloc_weights", {})  # NEW: pass this in export_data
    # Fallback: derive from asset_detail if alloc_weights not provided
    if not alloc_weights:
        adat = data.get("asset_detail", {})
        total_val = sum(v for v in adat.values() if v > 0)
        if total_val > 0:
            alloc_weights = {k: v/total_val for k, v in adat.items() if v > 0}

    yp = yt-45; c.setFont(FONT_BOLD, S_S); c.drawString(50, yp, "Asset Allocation Details")

    if alloc_weights:
        lbs  = [ASSET_LABELS.get(k, k) for k, v in alloc_weights.items() if v > 0.001]
        vals = [v for k, v in alloc_weights.items() if v > 0.001]
        if vals:
            fig_pie, ax_pie = plt.subplots(figsize=(4, 4),dpi=200)
            wedges, texts, autotexts = ax_pie.pie(
                vals, labels=None, autopct='%1.1f%%', startangle=140,
                pctdistance=0.75
            )
            ax_pie.legend(wedges, lbs, loc="lower center",
                          bbox_to_anchor=(0.5, -0.25), fontsize=7, ncol=2)
            plt.tight_layout()
            buf_pie = io.BytesIO()
            plt.savefig(buf_pie, format='png',dpi=200, transparent=True, bbox_inches='tight')
            plt.close(fig_pie)
            c.drawImage(ImageReader(buf_pie), 30, yp-220, width=220, height=220)

    yal = yp-50; c.setFont(FONT_BOLD, S_L); c.drawString(280, yal, "[Portfolio Allocation]")
    c.setFont(FONT_REG, S_B)
    for k, v in alloc_weights.items():
        if v > 0.001:
            yal -= 18
            c.drawString(290, yal, f"- {ASSET_LABELS.get(k, k)}: {v*100:.1f}%")

    yg = yp-250

    if res is not None:
        retire_age = data.get("retire_age", 60)

        c.setFont(FONT_BOLD, S_S); c.drawString(50, yg, "Wealth Projection (Monte Carlo — 50,000 paths)")

        # FIX 3, 4, 5, 6: Match Streamlit chart — 25/75 band, age axis, depletion line, M formatter
        fig_mc, ax_mc = plt.subplots(figsize=(10, 4))
        xr2 = list(range(len(res["median_balance"])))
        med = res["median_balance"]
        p10 = res["percentile_10"]
        p25 = res["percentile_25"]
        p75 = res["percentile_75"]
        p90 = res["percentile_90"]

        ax_mc.fill_between(xr2, p10, p90, alpha=0.10, color='steelblue',
                           label="10th–90th Percentile")
        ax_mc.fill_between(xr2, p25, p75, alpha=0.30, color='steelblue',
                           label="25th–75th Percentile")
        ax_mc.plot(xr2, p10, color='steelblue', linewidth=0.7, linestyle=':', alpha=0.5)
        ax_mc.plot(xr2, p90, color='steelblue', linewidth=0.7, linestyle=':', alpha=0.5)
        ax_mc.plot(xr2, med, label="Median (50th Pctl)", color='steelblue', linewidth=2.5)
        ax_mc.axhline(0, color='red', linestyle="--", linewidth=1.5, label="Portfolio Depleted (฿0)")

        inh_goal = data.get("inheritance_goal", 0.0)
        if inh_goal and inh_goal > 0:
            ax_mc.axhline(inh_goal, color='purple', linestyle="-.", linewidth=1.5,
                          label=f"Inheritance Goal ({inh_goal:,.0f} THB)")

        # FIX 5: Depletion vertical line
        depletion_yr = next((i for i, v in enumerate(med) if v <= 0), None)
        if depletion_yr:
            ax_mc.axvline(depletion_yr, color='orange', linestyle=':', linewidth=1.5,
                          label=f"Median depletes @ Year {depletion_yr}")

        ax_mc.set_xlabel("Year from Retirement")
        ax_mc.set_ylabel("Portfolio Value (THB)")
        ax_mc.set_title("Wealth Projection — Monte Carlo Simulation")

        # FIX 4: Age labels on top axis
        ax2 = ax_mc.twiny()
        ax2.set_xlim(ax_mc.get_xlim())
        tick_positions = xr2[::5]
        ax2.set_xticks(tick_positions)
        ax2.set_xticklabels([f"Age {retire_age + i}" for i in tick_positions], fontsize=8)

        # FIX 6: Y-axis M/K formatter
        ax_mc.get_yaxis().set_major_formatter(
            plt.FuncFormatter(lambda v, p: f"{v/1e6:.1f}M" if abs(v) >= 1e6 else format(int(v), ','))
        )

        ax_mc.legend(loc='upper right', fontsize='small')
        plt.tight_layout()
        buf_mc = io.BytesIO()
        plt.savefig(buf_mc, format='png', dpi=120, bbox_inches='tight')
        plt.close(fig_mc)
        c.drawImage(ImageReader(buf_mc), 50, yg-230, width=500, height=220)

        ys = yg-260
        c.setStrokeColorRGB(0.7, 0.7, 0.7); c.line(50, ys+15, width-50, ys+20)
        ys -= 20; c.setFont(FONT_BOLD, S_S); c.drawString(50, ys, "Simulation Outcome Summary")
        c.setFont(FONT_REG, S_B); ys -= 25
        sr  = res.get("survival_rate", 0) * 100
        me = res.get("median_surviving", res["median_balance"][-1])

        # FIX 7: Guard inheritance_success_rate against -1.0
        raw_ihs = res.get("inheritance_success_rate", -1.0)
        ihs_str = f"{raw_ihs*100:.1f}%" if raw_ihs >= 0 else "N/A (no goal set)"

        c.drawString(60, ys, f"Survival Rate: {sr:.1f}%")
        c.drawString(300, ys, f"Median End Balance: {me:,.2f} THB"); ys -= 20
        c.drawString(60, ys, f"Inheritance Success: {ihs_str}")
        c.drawString(300, ys, f"Strategy: {data.get('sim_strat', '-')}"); ys -= 20
        c.drawString(60, ys, f"Withdrawal Rate: {data.get('wd_rate', 0)*100:.2f}%")
        c.drawString(300, ys, f"Inflation: {data.get('inflation', 0.03)*100:.2f}%")

    c.save()
    return buffer.getvalue()


def parse_bloomberg_file(uploaded_file):
    try:
        filename=uploaded_file.name.lower(); header_idx=None
        if filename.endswith(('.xlsx','.xls')): df_raw=pd.read_excel(uploaded_file,header=None,nrows=20)
        else: df_raw=pd.read_csv(uploaded_file,header=None,nrows=20)
        for r,row in df_raw.iterrows():
            rt=row.astype(str).str.upper().str.cat(sep=' ')
            if "DATE" in rt and any(k in rt for k in ['PX','LAST','PRICE','TOT','RETURN','GROSS']):
                header_idx=r; break
        if header_idx is None: return None,"No 'Date' column found."
        uploaded_file.seek(0)
        if filename.endswith(('.xlsx','.xls')): df=pd.read_excel(uploaded_file,header=header_idx)
        else: df=pd.read_csv(uploaded_file,header=header_idx)
        date_col=next((c for c in df.columns if "date" in str(c).lower()),None)
        price_col=None
        for c in df.columns:
            cu=str(c).upper()
            if ("TOT" in cu or "GROSS" in cu) and ("RETURN" in cu or "INDEX" in cu): price_col=c; break
        if not price_col:
            for c in df.columns:
                if "TOT_RETURN" in str(c).upper(): price_col=c; break
        if not price_col:
            for c in df.columns:
                cu=str(c).upper()
                if "PX" in cu or "LAST" in cu or "CLOSE" in cu or "PRICE" in cu: price_col=c; break
        if not date_col or not price_col: return None,f"Columns missing. Found: {list(df.columns)}"
        df[date_col]=pd.to_datetime(df[date_col],errors='coerce')
        df.set_index(date_col,inplace=True)
        series=pd.to_numeric(df[price_col],errors='coerce').dropna().sort_index()
        series.name=f"{filename} ({price_col})"
        return series,None
    except Exception as e: return None,str(e)


# =========================================================
# NAV BAR
# =========================================================
if "nav_radio" not in st.session_state:
    st.session_state["nav_radio"] = steps[0]

st.radio("Go to:", steps, key="nav_radio", horizontal=True,
         label_visibility="collapsed", on_change=jump_step)
st.progress((st.session_state["current_step"]+1)/len(steps))
st.divider()

# =========================================================
# PAGE 1: FINANCIAL HEALTH
# =========================================================
if st.session_state["current_step"] == 0:
    st.header("👤 1. ข้อมูลผู้ใช้ (Financial Health)")
    st.subheader("A. ข้อมูลส่วนตัว")

    for k,v in [("user_name",""),("retire_age",60),("life_expectancy",85)]:
        if k not in st.session_state: st.session_state[k]=v
    if "ui_retire_age"      not in st.session_state: st.session_state["ui_retire_age"]=60
    if "ui_life_expectancy" not in st.session_state: st.session_state["ui_life_expectancy"]=85

    def validate_ages():
        st.session_state["retire_age"]      = int(st.session_state.get("ui_retire_age",60))
        st.session_state["life_expectancy"] = int(st.session_state.get("ui_life_expectancy",85))
        if st.session_state["life_expectancy"] < st.session_state["retire_age"]:
            st.session_state["life_expectancy"]=st.session_state["retire_age"]
            st.session_state["ui_life_expectancy"]=st.session_state["life_expectancy"]

    if st.session_state["life_expectancy"] < st.session_state["retire_age"]:
        st.session_state["life_expectancy"]=st.session_state["retire_age"]
        st.session_state["ui_life_expectancy"]=st.session_state["life_expectancy"]

    c1,c2,c3=st.columns(3)
    with c1: st.text_input("ชื่อ",value=st.session_state["user_name"],key="ui_user_name",
                            on_change=lambda: st.session_state.update({"user_name":st.session_state["ui_user_name"]}))
    with c2: st.number_input("อายุเกษียณ",min_value=40,max_value=100,
                              value=int(st.session_state["ui_retire_age"]),key="ui_retire_age",on_change=validate_ages)
    with c3: st.number_input("อายุขัย",min_value=int(st.session_state["ui_retire_age"]),max_value=120,
                              value=int(st.session_state["ui_life_expectancy"]),key="ui_life_expectancy",on_change=validate_ages)

    # ---- ASSETS ----
    st.subheader("B. ทรัพย์สิน (Investable Assets Only)")
    with st.expander("📝 รายละเอียดทรัพย์สิน",expanded=True):
        i1,i2=st.columns(2)
        with i1:
            money_cash    = money_input("เงินสด/เงินฝาก (Cash)",         0,"cash_dep")
            money_bond    = money_input("ตราสารหนี้ (Thai Bond)",          0,"bond")
            money_stock   = money_input("หุ้นไทย (Thai stock)",            0,"stock")
            money_reit    = money_input("PF&REIT",                          0,"reit")
        with i2:
            money_glbond  = money_input("ตราสารหนี้โลก (Global Bond)",     0,"gl_bond")
            money_glstock = money_input("หุ้นต่างประเทศ (Global stock)",   0,"gl_stock")
            money_glreit  = money_input("Global REIT",                      0,"gl_reit")
            money_gold    = money_input("ทองคำ (Gold)",                    0,"gold_invest")

    investable_assets = money_cash+money_bond+money_stock+money_glstock+money_reit+money_glbond+money_glreit+money_gold
    st.metric("💰 รวมเงินลงทุนทั้งหมด", f"{investable_assets:,.0f}")
    st.session_state["start_port"] = investable_assets

    # ---- DEBT SECTION ----
    # Helper: total amount over duration (monthly × months) or annual if permanent
    def _lifetime(ann, mo, mrem):
        if mrem is not None:
            return mo * mrem
        return ann
    st.subheader("C. หนี้สิน (Debt)")
    st.caption("💡 กด ⏱ เพื่อระบุว่าผ่อนอีกกี่เดือน/ปี — ระบบจะหยุดนับรายจ่ายนี้เมื่อครบกำหนด")

    with st.expander("📝 รายละเอียดหนี้สินรวม",expanded=True):
        st.markdown("💳 หนี้สินต่างๆ — กรอก **ค่างวดรายเดือน/รายปี** (ไม่ใช่ยอดคงค้าง)")

        debt_home_ann,  debt_home_mo,  debt_home_mrem  = timed_expense_input("หนี้บ้าน (Home Loan)",       "debt_home",  show_timer=True)
        debt_car_ann,   debt_car_mo,   debt_car_mrem   = timed_expense_input("หนี้รถ (Car Loan)",          "debt_car",   show_timer=True)
        debt_cc_ann,    debt_cc_mo,    debt_cc_mrem    = timed_expense_input("บัตรเครดิต (Credit Card)",   "debt_cc",    show_timer=True)
        debt_other_ann, debt_other_mo, debt_other_mrem = timed_expense_input("หนี้สินอื่น (Other Debt)",   "debt_other", show_timer=True)

        total_debt_annual = debt_home_ann + debt_car_ann + debt_cc_ann + debt_other_ann

        if total_debt_annual > 0:
            st.markdown("---")
            st.markdown("##### 📅 ตัวอย่างภาระหนี้รายปี/รายเดือน (ไม่รวมเงินเฟ้อ)")
            preview_years = min(int(st.session_state.get("life_expectancy",85)-st.session_state.get("retire_age",60)),30)
            debt_items_meta = [
                {"name":"Home",   "monthly":debt_home_mo,  "annual":debt_home_ann,  "months_rem":debt_home_mrem},
                {"name":"Car",    "monthly":debt_car_mo,   "annual":debt_car_ann,   "months_rem":debt_car_mrem},
                {"name":"CC",     "monthly":debt_cc_mo,    "annual":debt_cc_ann,    "months_rem":debt_cc_mrem},
                {"name":"Other",  "monthly":debt_other_mo, "annual":debt_other_ann, "months_rem":debt_other_mrem},
            ]
            rows_preview = []
            for yr in range(preview_years):
                row = {"ปีที่": yr+1}
                yr_total_ann = 0.0; yr_total_mo = 0.0
                for item in debt_items_meta:
                    if item["annual"] <= 0: continue
                    mo_rem = item["months_rem"]
                    if mo_rem is None:
                        ann_this_yr = item["annual"]
                        mo_this_yr  = item["monthly"]
                    else:
                        months_elapsed = yr * 12
                        if months_elapsed >= mo_rem:
                            ann_this_yr = 0.0; mo_this_yr = 0.0
                        elif months_elapsed + 12 > mo_rem:
                            active_months = mo_rem - months_elapsed
                            ann_this_yr = item["monthly"] * active_months
                            mo_this_yr  = item["monthly"] * (active_months / 12.0)
                        else:
                            ann_this_yr = item["annual"]
                            mo_this_yr  = item["monthly"]
                    row[f"{item['name']} (ปี)"]   = ann_this_yr
                    row[f"{item['name']} (เดือน)"] = mo_this_yr
                    yr_total_ann += ann_this_yr; yr_total_mo += mo_this_yr
                row["รวม/ปี"]    = yr_total_ann
                row["รวม/เดือน"] = yr_total_mo
                rows_preview.append(row)

            if rows_preview:
                prev_df = pd.DataFrame(rows_preview).set_index("ปีที่")
                fmt_df = prev_df.copy()
                for col in fmt_df.columns:
                    fmt_df[col] = fmt_df[col].apply(lambda x: f"{x:,.0f}" if x > 0 else "-")
                st.dataframe(fmt_df, use_container_width=True)

    total_debt_balance = debt_home_ann + debt_car_ann + debt_cc_ann + debt_other_ann
    total_debt_lifetime = (
        _lifetime(debt_home_ann,  debt_home_mo,  debt_home_mrem)  +
        _lifetime(debt_car_ann,   debt_car_mo,   debt_car_mrem)   +
        _lifetime(debt_cc_ann,    debt_cc_mo,    debt_cc_mrem)    +
        _lifetime(debt_other_ann, debt_other_mo, debt_other_mrem)
    )
    st.metric("💳 ค่างวดรวมต่อปี", f"{total_debt_balance:,.0f} บาท")
    st.info(f"💳 รวมภาระหนี้ตลอดช่วงเวลา: **{total_debt_lifetime:,.0f} บาท**")

    st.session_state["timed_debt_items"] = [
        {"name":"Home Loan",   "annual_amount":debt_home_ann,  "months_remaining":debt_home_mrem},
        {"name":"Car Loan",    "annual_amount":debt_car_ann,   "months_remaining":debt_car_mrem},
        {"name":"Credit Card", "annual_amount":debt_cc_ann,    "months_remaining":debt_cc_mrem},
        {"name":"Other Debt",  "annual_amount":debt_other_ann, "months_remaining":debt_other_mrem},
    ]

    # ---- CASH FLOW ----
    st.subheader("D. กระแสเงินสด (Cash Flow) — หลังเกษียณ")

    with st.expander("📥 1. รายได้ (Income)", expanded=True):
        # BUG FIX 8: capture months_remaining for each income source
        inc_pension_ann, inc_pension_mo, inc_pension_mrem = timed_expense_input("เงินบำนาญ (Pension)", "inc_pension", show_timer=True, timer_label="ได้รับอีก")
        inc_rent_ann,    inc_rent_mo,    inc_rent_mrem    = timed_expense_input("ค่าเช่า (Rental)",     "inc_rent",    show_timer=True, timer_label="ได้รับอีก")
        inc_div_ann,     inc_div_mo,     inc_div_mrem     = timed_expense_input("ดอกเบี้ย/ปันผล (Dividend)", "inc_div", show_timer=True, timer_label="ได้รับอีก")
        inc_other_ann,   inc_other_mo,   inc_other_mrem   = timed_expense_input("รายได้อื่นๆ (Other)", "inc_other",   show_timer=True, timer_label="ได้รับอีก")
        total_income = inc_pension_ann + inc_rent_ann + inc_div_ann + inc_other_ann

        total_income_lifetime = (
            _lifetime(inc_pension_ann, inc_pension_mo, inc_pension_mrem) +
            _lifetime(inc_rent_ann,    inc_rent_mo,    inc_rent_mrem)    +
            _lifetime(inc_div_ann,     inc_div_mo,     inc_div_mrem)     +
            _lifetime(inc_other_ann,   inc_other_mo,   inc_other_mrem)
        )

    # BUG FIX 8: store timed income items so expiring income is removed from schedule
    st.session_state["timed_income_items"] = [
        {"name":"Pension",  "annual_amount":inc_pension_ann, "months_remaining":inc_pension_mrem},
        {"name":"Rental",   "annual_amount":inc_rent_ann,    "months_remaining":inc_rent_mrem},
        {"name":"Dividend", "annual_amount":inc_div_ann,     "months_remaining":inc_div_mrem},
        {"name":"Other",    "annual_amount":inc_other_ann,   "months_remaining":inc_other_mrem},
    ]

    st.success(f"💰 **รวมรายได้:** {total_income:,.0f} บาท/ปี (เฉลี่ย {total_income/12:,.0f} บาท/เดือน) | รวมตลอดช่วงเวลา: **{total_income_lifetime:,.0f} บาท**")

    with st.expander("💸 2. รายจ่าย (Expenses)"):
        st.markdown("🔹 รายจ่ายคงที่ (Fixed)")
        st.caption("💡 กด ⏱ สำหรับรายจ่ายที่มีระยะเวลาสิ้นสุด")

        exp_house_ann, exp_house_mo, exp_house_mrem = timed_expense_input("ค่าที่อยู่ (Housing)",   "exp_house",   show_timer=True)
        exp_ins_ann,   exp_ins_mo,   exp_ins_mrem   = timed_expense_input("ประกัน (Insurance)",     "exp_ins",     show_timer=True)
        exp_sub_ann,   exp_sub_mo,   exp_sub_mrem   = timed_expense_input("Subscription",           "exp_sub",     show_timer=True)
        exp_fix_ann,   exp_fix_mo,   exp_fix_mrem   = timed_expense_input("อื่นๆ (Other Fixed)",    "exp_fix_oth", show_timer=True)

        total_fixed = exp_house_ann+exp_ins_ann+exp_sub_ann+exp_fix_ann
        total_fixed_lifetime = (
            _lifetime(exp_house_ann, exp_house_mo, exp_house_mrem) +
            _lifetime(exp_ins_ann,   exp_ins_mo,   exp_ins_mrem)   +
            _lifetime(exp_sub_ann,   exp_sub_mo,   exp_sub_mrem)   +
            _lifetime(exp_fix_ann,   exp_fix_mo,   exp_fix_mrem)
        )
        st.info(f"รวม Fixed: {total_fixed:,.0f} บาท/ปี | รวมตลอดช่วงเวลา: **{total_fixed_lifetime:,.0f} บาท**")

        st.markdown("---")
        st.markdown("🔸 รายจ่ายผันแปร (Non-Fixed)")
        exp_trans_ann,  exp_trans_mo,  exp_trans_mrem  = timed_expense_input("ค่าเดินทาง (Transport)", "exp_trans",   show_timer=True)
        exp_food_ann,   exp_food_mo,   exp_food_mrem   = timed_expense_input("ค่าอาหาร (Food)",        "exp_food",    show_timer=True)
        exp_ent_ann,    exp_ent_mo,    exp_ent_mrem    = timed_expense_input("สันทนาการ (Entertain)",   "exp_ent",     show_timer=True)
        exp_travel_ann, exp_travel_mo, exp_travel_mrem = timed_expense_input("ท่องเที่ยว (Travel)",    "exp_travel",  show_timer=True)
        exp_health_ann, exp_health_mo, exp_health_mrem = timed_expense_input("รักษาพยาบาล (Health)",   "exp_health",  show_timer=True)
        exp_var_ann,    exp_var_mo,    exp_var_mrem    = timed_expense_input("อื่นๆ (Other Variable)",  "exp_var_oth", show_timer=True)

        total_variable = exp_trans_ann+exp_food_ann+exp_ent_ann+exp_travel_ann+exp_health_ann+exp_var_ann
        total_variable_lifetime = (
            _lifetime(exp_trans_ann,  exp_trans_mo,  exp_trans_mrem)  +
            _lifetime(exp_food_ann,   exp_food_mo,   exp_food_mrem)   +
            _lifetime(exp_ent_ann,    exp_ent_mo,    exp_ent_mrem)    +
            _lifetime(exp_travel_ann, exp_travel_mo, exp_travel_mrem) +
            _lifetime(exp_health_ann, exp_health_mo, exp_health_mrem) +
            _lifetime(exp_var_ann,    exp_var_mo,    exp_var_mrem)
        )
        st.info(f"รวม Variable: {total_variable:,.0f} บาท/ปี | รวมตลอดช่วงเวลา: **{total_variable_lifetime:,.0f} บาท**")
        total_expense = total_fixed + total_variable
        total_expense_lifetime = total_fixed_lifetime + total_variable_lifetime

    st.error(f"📉 **รวมรายจ่าย:** {total_expense:,.0f} บาท/ปี (เฉลี่ย {total_expense/12:,.0f} บาท/เดือน) | รวมตลอดช่วงเวลา: **{total_expense_lifetime:,.0f} บาท**")

    st.session_state["timed_expense_items"] = [
        {"name":"Housing",     "annual_amount":exp_house_ann, "months_remaining":exp_house_mrem},
        {"name":"Insurance",   "annual_amount":exp_ins_ann,   "months_remaining":exp_ins_mrem},
        {"name":"Subscription","annual_amount":exp_sub_ann,   "months_remaining":exp_sub_mrem},
        {"name":"Other Fixed", "annual_amount":exp_fix_ann,   "months_remaining":exp_fix_mrem},
    ]

    st.session_state["v_total_income"]  = total_income
    st.session_state["v_total_expense"] = total_expense
    st.session_state["v_net_cashflow"]  = total_income - total_expense

    yearly_savings = total_income - total_expense

    st.markdown("### 📊 สรุปสถานะการเงิน (หลังเกษียณ)")
    m1,m2,m3=st.columns(3)
    m1.metric("เงินลงทุนได้ (Investable)",       f"{investable_assets:,.0f}")
    m2.metric("เงินคงเหลือ/ปี",                  f"{yearly_savings:,.0f}")
    m3.metric("หนี้สินรวม",                       f"{total_debt_balance:,.0f}")
    if yearly_savings < 0:
        st.warning(f"⚠️ รายจ่ายมากกว่ารายได้ {abs(yearly_savings):,.0f} บาท/ปี")

    st.session_state.update({
        "start_port":  investable_assets,
        "money_save":  yearly_savings,
        "money_debt":  total_debt_balance,
        "v_cash_dep":  money_cash, "v_bond":     money_bond,
        "v_stock":     money_stock,"v_reit":     money_reit,
        "v_gl_bond":   money_glbond,"v_gl_stock": money_glstock,
        "v_gl_reit":   money_glreit,"v_gold":    money_gold,
    })

    st.session_state["inflation"] = st.slider(
        "เงินเฟ้อคาดการณ์ (%)",0.0,10.0,
        st.session_state.get("inflation",0.03)*100,0.1)/100

    st.subheader("Inheritance Goals")
    with st.expander("📝 เป้าหมายมรดกที่ต้องการ",expanded=True):
        st.session_state["inheritance_goal"] = money_input("จำนวนเงินที่ต้องการ (THB)",0,"inheritance_goal")

    _,c_nav2=st.columns([10,3])
    with c_nav2: st.button("Next Step ➡",on_click=next_step,type="primary")


# =========================================================
# PAGE 2: RISK ASSESSMENT
# =========================================================
elif st.session_state["current_step"] == 1:
    st.header("🧩 2. แบบประเมินความเสี่ยง")

    questions_data = [
        {"q": "Q1: ปัจจุบันคุณกำลังอยู่ในช่วงชีวิตใด", "choices": [{"label": "อายุยังไม่เกิน 30 ปี เริ่มต้นทำงาน เก็บเงินเก็บทอง", "score": 3}, {"label": "อายุเกิน 30 แต่ไม่เกิน 55 ปี อยู่ในวัยทำงาน มีเงินเก็บเงินก้อน", "score": 2}, {"label": "อายุเกิน 55 ปี ใกล้เกษียณอยากพักผ่อน", "score": 1}]},
        {"q": "Q2: ในเรื่องการลงทุนเมื่อพูดถึง “ความผันผวน” คุณนึกถึงอะไรเป็นอันดับแรก", "choices": [{"label": "นี่แหละโอกาสทอง ขึ้นก็ขาย ลงก็ซื้อ ได้กำไรตั้งหลายรอบ", "score": 3}, {"label": "ที่ไหนมีความผันผวน ที่นั่นมีความไม่แน่นอน", "score": 2}, {"label": "แย่แล้วถ้าราคาตก ก็ขาดทุนสิ!!", "score": 1}]},
        {"q": "Q3: สไตล์การลงทุนที่ผ่านมาของคุณเป็นแบบไหน", "choices": [{"label": "กล้าได้กล้าเสีย ถึงเวลาต้องยอมตัดขาดทุน แล้วไปลุยใหม่ สร้างกำไรสูงๆ", "score": 3}, {"label": "ช้าแต่ชัวร์ ได้น้อยดีกว่าไม่ได้ แต่ไม่อยากขาดทุน", "score": 1}, {"label": "แล้วแต่จังหวะ แล้วแต่โอกาส บางทีก็เสี่ยงบ้าง มีกำไรพอประมาณ", "score": 2}]},
        {"q": "Q4: หากลงทุนแล้วขาดทุน อะไรคือสาเหตุในความคิดของคุณ", "choices": [{"label": "การตัดสินใจที่ผิดพลาดของตัวเรา", "score": 3}, {"label": "เป็นเพราะความไม่แน่นอนของตลาดและภาวะการลงทุน", "score": 1}, {"label": "ก็ทั้งตัวเราแล้วก็ภาวะการลงทุนนั่นแหละ", "score": 2}]},
        {"q": "Q5: ลองหลับตาแล้วมองไปข้างหน้าในอีก 1 ปี คุณอยากเห็นอะไรจากเงินลงทุน", "choices": [{"label": "ผลตอบแทนแน่นอน 5%", "score": 1}, {"label": "หวังกำไรถึง 10% แต่ถ้าโชคไม่ดีขาดทุนก็ยอมได้สัก 5%", "score": 2}, {"label": "หวังกำไรถึง 20% แต่ถ้าโชคไม่ดีขาดทุนก็ยอมได้สัก 10%", "score": 3}]},
        {"q": "Q6: ถ้าคุณโชคดีถูกล๊อตเตอรี่ได้เงินรางวัล 500,000 บาท คุณจะนำเงินไปลงทุนอะไร", "choices": [{"label": "ฝากประจำหรือพันธบัตรรัฐบาล เงินต้นอยู่ครบ ผลตอบแทนน้อยหน่อยแต่แน่นอน", "score": 1}, {"label": "แบ่งครึ่งหนึ่งไปซื้อหุ้นสามัญ อีกครึ่งหนึ่งไปซื้อพันธบัตรรัฐบาล", "score": 2}, {"label": "โชคดีแบบนี้ไม่ต้องกลัว ซื้อหุ้นไปเลย", "score": 3}]},
        {"q": "Q7: การได้ไปท่องเที่ยวต่างประเทศแบบหรูหรา เป็นความใฝ่ฝันของคุณที่อุตส่าห์เก็บหอมรอมริบมานานหลายปี ทว่าก่อนจองโปรแกรมท่องเที่ยว คุณโดนเลิกจ้างกะทันหันจากนโยบายลดจำนวนพนักงานของบริษัท คุณจะตัดสินใจอย่างไร", "choices": [{"label": "ยกเลิกโปรแกรมท่องเที่ยว จนกว่าจะหางานใหม่ได้", "score": 1}, {"label": "เปลี่ยนแผนท่องเที่ยว ไปแบบประหยัดแทน", "score": 2}, {"label": "จองโปรแกรมและไปเที่ยวตามเดิม กลับมาค่อยว่ากัน", "score": 3}]},
        {"q": "Q8: คุณได้ร่วมรายการเกมโชว์ เล่นได้ถึงรอบลึกๆ และมาถึงทางเลือกที่ว่าจะเล่นต่อหรือหยุดเล่น ด้วยเงื่อนไขต่างๆ คุณจะเลือกอย่างไร", "choices": [{"label": "หยุดเล่นแล้วรับเงินรางวัล 30,000 บาท", "score": 1}, {"label": "เล่นต่อกับคำถาม 2 ตัวเลือก ตอบถูกรับเงิน 60,000 บาท ตอบผิดไม่ได้อะไรเลย", "score": 2}, {"label": "เล่นต่อกับคำถาม 4 ตัวเลือก ตอบถูกรับเงิน 120,000 บาท ตอบผิดไม่ได้อะไรเลย", "score": 3}]},
        {"q": "Q9: เพื่อนของคุณที่เก่งด้านการค้าที่ดิน มาชวนลงทุนซื้อที่ดินด้วยกัน และคาดว่าราคามีโอกาสจะเพิ่มจากตารางวาละ 20,000 บาท เป็น 40,000 บาท ในอีก 1 ปีข้างหน้า แต่ก็มีโอกาสที่ราคาจะไม่เพิ่มขึ้นอยู่เหมือนกัน คุณจะร่วมลงทุนก็ต่อเมื่อโอกาสที่ราคาที่ดินจะเพิ่มขึ้นเป็นแบบใด ", "choices": [{"label": "ถึงจะเป็นไปได้น้อย ก็อยากลงทุนด้วย", "score": 3}, {"label": "ต้องมีความเป็นไปได้ปานกลาง ถึงจะลงทุนด้วย", "score": 2}, {"label": "ต้องเป็นไปได้มากๆ หน่อย ถึงจะลงทุนด้วย", "score": 1}]},
        {"q": "Q10: เจ้าของธุรกิจแห่งหนึ่งชวนคุณไปทำงานด้วย โดยมีเงื่อนไขระหว่าง ให้รับผลตอบแทนเป็นเงินเดือนที่แน่นอน หรือรับเงินเดือนน้อยหน่อยแต่มีค่านายหน้าตามผลงานยอดขายที่ทำได้ คุณจะเลือกรับผลตอบแทนแบบใด", "choices": [{"label": "เอารายได้แน่นอนดีกว่า เลือกรับเงินเดือนเป็นหลัก ค่านายหน้านิดหน่อย", "score": 1}, {"label": "เลือกแบบสมดุล รับเงินเดือนครึ่งหนึ่ง ค่านายหน้าอีกครึ่งหนึ่ง", "score": 2}, {"label": "เลือกรับรายได้ตามผลงาน เน้นค่านายหน้าเป็นหลัก เงินเดือนเล็กน้อย", "score": 3}]}
    ]

    def persistent_radio(key_suffix,options,label):
        idx_key=f"idx_{key_suffix}"; ui_key=f"ui_{key_suffix}"
        def on_change():
            obj=st.session_state[ui_key]
            try: st.session_state[idx_key]=options.index(obj)
            except: st.session_state[idx_key]=None
        saved_idx=st.session_state.get(idx_key,None)
        return st.radio(label,options,format_func=lambda x:x["label"],
                        index=saved_idx,key=ui_key,on_change=on_change,label_visibility="collapsed")

    @st.dialog("🎯 ข้อแนะนำการลงทุนตามระดับความเสี่ยงของคุณ")
    def show_risk_advice():
        profile=st.session_state.get("risk_profile","ไม่ระบุ")
        score=st.session_state.get("risk_score",0)
        st.write(f"ระดับความเสี่ยง: **{profile}** (คะแนน: {score})")
        # Show post-retirement note when user selected Q1 choice 1 or 2
        # (idx 0 = "อายุไม่เกิน 30", idx 1 = "อายุเกิน 30 ไม่เกิน 55") — both are pre-retirement ages
        q1_idx = st.session_state.get("idx_q_0", None)
        post_retirement_note = q1_idx in (0, 1)

        if "1" in profile:
            advice="เน้นรักษาเงินต้น"
            data={"สินทรัพย์":["เงินฝาก/ตราสารหนี้","หุ้น","ทางเลือก"],"สัดส่วน":["85%","10%","5%"]}
        elif "2" in profile:
            advice="ยอมรับความเสี่ยงได้บ้าง"
            data={"สินทรัพย์":["เงินฝาก/ตราสารหนี้","หุ้น","ทางเลือก"],"สัดส่วน":["80%","15%","5%"]}
        elif "3" in profile:
            advice="สมดุล 60/40"
            data={"สินทรัพย์":["เงินฝาก/ตราสารหนี้","หุ้น","ทางเลือก"],"สัดส่วน":["60%","30%","10%"]}
        elif "4" in profile:
            advice="เน้นสร้างความมั่งคั่ง"
            data={"สินทรัพย์":["ตราสารหนี้","หุ้น","ทางเลือก"],"สัดส่วน":["50%","40%","10%"]}
        else:
            advice="เน้นการเติบโตสูงสุด"
            data={"สินทรัพย์":["ตราสารหนี้","หุ้น","ทางเลือก"],"สัดส่วน":["40%","40%","20%"]}
        if post_retirement_note:
            st.warning("⚠️ **หมายเหตุ:** การวางแผนนี้เป็นการวางแผนสำหรับช่วงหลังเกษียณ (Post-Retirement Planning) ซึ่งเน้นการรักษาเงินทุนและสร้างรายได้ที่มั่นคง มากกว่าการเติบโตของพอร์ต")
        st.info(f"💡 {advice}"); st.table(data)
        if st.button("ยืนยันและไปหน้าถัดไป ➡",type="primary",use_container_width=True):
            next_step(); st.rerun()

    total_score=0; all_answered=True
    for i,item in enumerate(questions_data):
        st.subheader(item["q"])
        choice=persistent_radio(f"q_{i}",item["choices"],f"R_{i}")
        st.divider()
        if choice is None: all_answered=False
        else: total_score+=int(choice["score"])

    if all_answered:
        if   total_score>=26: profile="ระดับ 5: เสี่ยงสูงมาก (Aggressive)"
        elif total_score>=22: profile="ระดับ 4: เสี่ยงสูง (Moderate Aggressive)"
        elif total_score>=18: profile="ระดับ 3: เสี่ยงปานกลางค่อนข้างสูง (Moderate)"
        elif total_score>=14: profile="ระดับ 2: เสี่ยงปานกลางค่อนข้างต่ำ (Moderate Conservative)"
        else:                 profile="ระดับ 1: เสี่ยงต่ำ (Conservative)"
        st.success(f"คะแนน: {total_score} - {profile}")
        st.session_state["risk_profile"]=profile; st.session_state["risk_score"]=total_score
        c1,c2=st.columns([1,8])
        with c1: st.button("⬅ Back",on_click=prev_step)
        with c2:
            if st.button("Next Step ➡",type="primary",disabled=not all_answered): show_risk_advice()
    else:
        c1,c2=st.columns([1,8])
        with c1: st.button("⬅ Back",on_click=prev_step)
        with c2: st.button("Next Step ➡",on_click=next_step,type="primary")


# =========================================================
# PAGE 3: ASSET ALLOCATION
# =========================================================
elif st.session_state["current_step"] == 2:
    st.header("📊 3. จัดพอร์ตการลงทุน (Asset Allocation)")

    curr_cash    =st.session_state.get("v_cash_dep",   0.0)
    curr_bond    =st.session_state.get("v_bond",       0.0)
    curr_stock   =st.session_state.get("v_stock",      0.0)
    curr_reit    =st.session_state.get("v_reit",       0.0)
    curr_gl_bond =st.session_state.get("v_gl_bond",    0.0)
    curr_gl_stock=st.session_state.get("v_gl_stock",   0.0)
    curr_gl_reit =st.session_state.get("v_gl_reit",    0.0)
    curr_gold    =st.session_state.get("v_gold_invest",0.0)

    val_deposit    =money_input("Fix Deposit (THB)",                         curr_cash,    "p3_deposit")
    c1,c2=st.columns(2)
    with c1:
        st.subheader("Thai Assets")
        val_gov_bond =money_input("Bond (THB) - THAT Index",      curr_bond,    "p3_gov_bond")
        val_seti     =money_input("Thai Stock (THB) - SET Index",            curr_stock,   "p3_seti")
        val_reit     =money_input("PF&REIT (THB) - SETPREIT Index",          curr_reit,    "p3_reit")
    with c2:
        st.subheader("Global Assets")
        val_msci_gov  =money_input("Global Bond (THB) - LEGATRUU",       curr_gl_bond, "p3_msci_gov")
        val_msci_stock=money_input("Global Stock (THB) - MXWD Index",        curr_gl_stock,"p3_msci_stock")
        val_msci_reits=money_input("Global REITs (THB) - NDUWREIT Index",    curr_gl_reit, "p3_mscireits")
        val_gold      =money_input("Gold (THB) - XAUTHB",                    curr_gold,    "p3_gold")

    total_port_value=(val_deposit+val_gov_bond+val_seti+val_reit+
                      val_msci_gov+val_msci_stock+val_msci_reits+val_gold)

    if total_port_value>0:
        st.markdown(f"### 💰 Total Portfolio: **{total_port_value:,.0f}** THB")
        alloc={
            "pct_deposit":       val_deposit    /total_port_value,
            "pct_bond":      val_gov_bond   /total_port_value,
            "pct_seti":          val_seti       /total_port_value,
            "pct_gold":          val_gold       /total_port_value,
            "pct_REIT":          val_reit       /total_port_value,
            "pct_msci_bond": val_msci_gov   /total_port_value,
            "pct_msci_stock":    val_msci_stock /total_port_value,
            "pct_msci_reit":     val_msci_reits /total_port_value,
        }
        st.session_state["saved_alloc"]=alloc
        st.session_state["final_total_wealth"]=total_port_value

        col_l, col_c, col_r = st.columns([2,1,2])
        with col_c:
            fig_pie, ax_pie = plt.subplots(figsize=(2.5, 2.5), dpi=150)
            ax_pie.set_title("Portfolio Allocation", fontsize=7, fontweight='bold', pad=12)
            labels_pie = [k.replace("pct_","").upper() for k,v in alloc.items() if v>0.001]
            vals_pie   = [v for k,v in alloc.items() if v>0.001]
            ax_pie.pie(vals_pie, autopct="%1.0f%%", startangle=140, textprops={'fontsize': 6})
            ax_pie.legend(labels_pie, loc="upper center", bbox_to_anchor=(0.5, -0.05),
                        fontsize=5, ncol=2, frameon=False)
            plt.tight_layout()
            st.pyplot(fig_pie, use_container_width=True)
            plt.close(fig_pie)

        with st.expander("ℹ️ วิธีอ่านกราฟ Pie Chart การจัดสรรสินทรัพย์", expanded=False):
            st.markdown(
                "**วิธีอ่านกราฟ Pie Chart การจัดสรรสินทรัพย์**\n\n"
                "- แต่ละสีแสดงสัดส่วนของสินทรัพย์แต่ละประเภทในพอร์ต\n"
                "- ตัวเลข % คือน้ำหนักของสินทรัพย์นั้นในพอร์ตรวม\n"
                "- **เงินฝาก/พันธบัตร** = ความเสี่ยงต่ำ ผลตอบแทนต่ำ เสถียรภาพสูง\n"
                "- **หุ้น (SET/MSCI)** = ความเสี่ยงสูง ผลตอบแทนสูง ผันผวนมาก\n"
                "- **REIT** = รายได้สม่ำเสมอจากอสังหาริมทรัพย์ ความเสี่ยงปานกลาง\n"
                "- **ทองคำ** = สินทรัพย์ป้องกันความเสี่ยง มักเคลื่อนไหวสวนทางกับหุ้น\n"
                "- พอร์ตที่ดีควรกระจายความเสี่ยงหลายสินทรัพย์ ไม่กระจุกที่เดียว"
            )
    else:
        st.warning("⚠️ Please enter asset values to continue.")

    st.divider()
    profile = st.session_state.get("risk_profile", "")
    score   = st.session_state.get("risk_score", 0)

    if profile:
        st.subheader("💡 คำแนะนำการลงทุนตามระดับความเสี่ยงของคุณ")
        st.write(f"ระดับความเสี่ยง: **{profile}** (คะแนน: {score})")

        if "ระดับ 1" in profile or (score <= 13):
            advice = "เน้นรักษาเงินต้น"
            alloc_data = {"สินทรัพย์": ["เงินฝาก/ตราสารหนี้", "หุ้น", "สินทรัพย์ทางเลือก"],
                          "สัดส่วนแนะนำ": ["85%", "10%", "5%"]}
        elif "ระดับ 2" in profile or (score <= 17):
            advice = "ยอมรับความเสี่ยงได้บ้าง"
            alloc_data = {"สินทรัพย์": ["เงินฝาก/ตราสารหนี้", "หุ้น", "สินทรัพย์ทางเลือก"],
                          "สัดส่วนแนะนำ": ["80%", "15%", "10%"]}
        elif "ระดับ 3" in profile or (score <= 21):
            advice = "สมดุล 60/40"
            alloc_data = {"สินทรัพย์": ["เงินฝาก/ตราสารหนี้", "หุ้น", "สินทรัพย์ทางเลือก"],
                          "สัดส่วนแนะนำ": ["60%", "30%", "10%"]}
        elif "ระดับ 4" in profile or (score <= 25):
            advice = "เน้นสร้างความมั่งคั่ง"
            alloc_data = {"สินทรัพย์": ["ตราสารหนี้", "หุ้น", "สินทรัพย์ทางเลือก"],
                          "สัดส่วนแนะนำ": ["50%", "40%", "10%"]}
        else:
            advice = "เน้นการเติบโตสูงสุด"
            alloc_data = {"สินทรัพย์": ["ตราสารหนี้", "หุ้น", "สินทรัพย์ทางเลือก"],
                          "สัดส่วนแนะนำ": ["40%", "40%", "20%"]}

        st.info(f"💡 **คำแนะนำ:** {advice}")
        st.table(alloc_data)

        # Show post-retirement note if user selected Q1 choice 1 or 2 (pre-retirement age groups)
        q1_idx = st.session_state.get("idx_q_0", None)
        if q1_idx in (0, 1):
            st.warning("⚠️ **หมายเหตุ:** การวางแผนนี้เป็นการวางแผนสำหรับช่วงหลังเกษียณ (Post-Retirement Planning) ซึ่งเน้นการรักษาเงินทุนและสร้างรายได้ที่มั่นคง มากกว่าการเติบโตของพอร์ต")
    else:
        st.info("💡 กรุณาทำแบบประเมินความเสี่ยงในหน้า 2 เพื่อรับคำแนะนำการจัดพอร์ตที่เหมาะกับคุณ")

    cn1,cn2=st.columns([1,8])
    cn1.button("⬅ Back",on_click=prev_step)
    cn2.button("Next Step ➡",type="primary",on_click=next_step)


# =========================================================
# PAGE 4: SIMULATION + EXPORT
# =========================================================
elif st.session_state["current_step"] == 3:
    st.header("💸 4.กลยุทธ์การถอนเงิน")

    YEARS=30; N_SIM=50000

    asset_stats={
        "pct_deposit":       {"mean":0.0200,"std":0.0100},
        "pct_gov_bond":      {"mean":0.0348,"std":0.0244},
        "pct_seti":          {"mean":0.0547,"std":0.2729},
        "pct_REIT":          {"mean":0.0692,"std":0.1420},
        "pct_msci_stock":    {"mean":0.0945,"std":0.1564},
        "pct_msci_gov_bond": {"mean":0.0557,"std":0.0983},
        "pct_msci_reit":     {"mean":0.0730,"std":0.1762},
        "pct_gold":          {"mean":0.0836,"std":0.1601},
    }

    alloc       = st.session_state.get("saved_alloc",{})
    start_port  = st.session_state.get("start_port",1_000_000.0)
    inflation   = st.session_state.get("inflation",0.03)
    retire_age  = st.session_state.get("retire_age",60)
    inheritance = float(st.session_state.get("v_inheritance_goal") or st.session_state.get("inheritance_goal") or 0.0)

    timed_debt_items    = st.session_state.get("timed_debt_items",[])
    timed_expense_items = st.session_state.get("timed_expense_items",[])
    # BUG FIX 8: retrieve timed income items collected on page 1
    timed_income_items  = st.session_state.get("timed_income_items",[])

    # Combine debt + fixed expense timed items for cashflow schedule
    all_timed_expense = [
        {"annual_amount": it["annual_amount"], "months_remaining": it["months_remaining"]}
        for it in (timed_debt_items + timed_expense_items)
        if it.get("months_remaining") is not None and it["annual_amount"] > 0
    ]
    # BUG FIX 8: timed income items for cashflow schedule
    all_timed_income = [
        {"annual_amount": it["annual_amount"], "months_remaining": it["months_remaining"]}
        for it in timed_income_items
        if it.get("months_remaining") is not None and it["annual_amount"] > 0
    ]
    has_timed = len(all_timed_expense) > 0 or len(all_timed_income) > 0

    base_annual_expense = st.session_state.get("v_total_expense", 0.0)
    base_annual_income  = st.session_state.get("v_total_income",  0.0)

    # Strategy & withdrawal rate selection
    c1,c2=st.columns(2)
    strat_options=["Basic Strategy","Forgoing Inflation","RMD Strategy","Guardrails"]
    with c1:
        strat_selection=st.selectbox("กลยุทธ์",strat_options)
        st.session_state["sim_strat"]=strat_selection
    with c2:
        wd_rate=st.number_input("อัตราการถอน (%)",3.0,10.0,4.0,0.1)/100

    # BUG FIX 8: pass timed_income_items into build_cashflow_schedule
    cf_schedule = None
    if has_timed:
        cf_schedule = RetirementSimulator.build_cashflow_schedule(
            base_annual_expense=base_annual_expense,
            base_annual_income=base_annual_income,
            timed_items=all_timed_expense,
            total_years=YEARS,
            inflation_rate=inflation,
            withdrawal_rate=wd_rate,
            initial_portfolio=start_port,
            timed_income_items=all_timed_income,   # BUG FIX 8
        )

    # ---- Data source ----
    st.markdown("### 📂 Data Assumptions")
    data_mode=st.radio("Choose Source:",["Use Default Data","Upload Bloomberg Files"],horizontal=True)
    custom_mean=None; custom_cov=None

    if data_mode=="Upload Bloomberg Files":
        st.info("💡 Upload Asset files AND a USD/THB Exchange Rate file if the currency is USD")
        with st.expander("วิธีการ download ข้อมูลจาก bloomberge",expanded=False):
            st.markdown("1 ...")
        uploaded_files=st.file_uploader("Upload Excel/CSV files here:",type=["csv","xlsx","xls"],accept_multiple_files=True)
        sys_map={"Select Option...":"ignore","🔴 USD/THB Exchange Rate":"rate_usd_thb",
                 "-----------------------":"ignore","Thai Bond":"pct_gov_bond",
                 "Thai Equity (SET)":"pct_seti","Thai REITs":"pct_REIT",
                 "Global Stocks (MSCI)":"pct_msci_stock","Global Bond":"pct_msci_gov_bond",
                 "Global REITs":"pct_msci_reit","Gold":"pct_gold"}
        if uploaded_files:
            file_configs=[]; parsed_series={}
            for f in uploaded_files:
                c1a,c2a,c3a=st.columns([3,3,2])
                c1a.write(f"📄 **{f.name}**")
                choice_label=c2a.selectbox("Map to:",list(sys_map.keys()),key=f"map_{f.name}",label_visibility="collapsed")
                asset_code=sys_map[choice_label]
                currency="THB"
                if asset_code not in ("ignore","rate_usd_thb"):
                    currency=c3a.radio("Currency:",["THB","USD"],key=f"curr_{f.name}",horizontal=True,label_visibility="collapsed")
                if asset_code!="ignore": file_configs.append({"file":f,"code":asset_code,"curr":currency})
            if file_configs:
                usd_rate_series=None
                for cfg in file_configs:
                    s,err=parse_bloomberg_file(cfg["file"])
                    if not err:
                        parsed_series[cfg["code"]]=s
                        if cfg["code"]=="rate_usd_thb": usd_rate_series=s
                    else: st.error(f"❌ {cfg['file'].name}: {err}")
                needs_usd=any(c["curr"]=="USD" for c in file_configs)
                if needs_usd and usd_rate_series is None:
                    st.error("🚨 USD files need USD/THB rate file!")
                else:
                    merged_df=pd.DataFrame()
                    for cfg in file_configs:
                        code=cfg["code"]
                        if code=="rate_usd_thb": continue
                        raw=parsed_series.get(code)
                        if raw is not None:
                            if cfg["curr"]=="USD":
                                aln=pd.concat([raw,usd_rate_series],axis=1,join='inner').dropna()
                                aln.columns=["Asset_USD","FX_Rate"]
                                merged_df[code]=aln["Asset_USD"]*aln["FX_Rate"]
                                st.caption(f"✅ Converted {cfg['file'].name} to THB")
                            else: merged_df[code]=raw
                    if not merged_df.empty:
                        try:
                            df_mo=merged_df.resample('ME').last().pct_change().dropna()
                            if len(df_mo)>36:
                                st.success(f"✅ {len(df_mo)} months of data ready.")
                                vdf=pd.DataFrame({"Monthly Return":df_mo.mean()*100,"Annual Return":df_mo.mean()*12*100,"Volatility":df_mo.std()*(12**0.5)*100})
                                st.dataframe(vdf.style.format("{:.2f}%"))
                                custom_mean=(df_mo.mean()*12).to_dict(); custom_cov=df_mo.cov()*12
                                st.session_state["custom_mean"]=custom_mean; st.session_state["custom_cov"]=custom_cov
                            else: st.warning("⚠️ Need >3 years of data.")
                        except Exception as e: st.error(f"Error: {e}")

    if data_mode=="Upload Bloomberg Files" and "custom_mean" in st.session_state:
        custom_mean=st.session_state["custom_mean"]; custom_cov=st.session_state["custom_cov"]

    if data_mode == "Use Default Data":
        st.info("💡 Default data is provided by Bloomberg from 28/2/1990 to 28/11/2025 in monthly ")

    # ---- RUN SIMULATION ----
    if st.button("🚀 Run Simulation",type="primary"):
        sim=RetirementSimulator()
        with st.spinner("Simulating 50,000 paths..."):
            mc_returns=sim.simulate_returns(alloc,asset_stats,N_SIM,YEARS)
            st.session_state["mc_returns"]=mc_returns
            res=sim.run_simulation(
                initial_portfolio=start_port, portfolio_allocation=alloc,
                asset_stats=asset_stats, withdrawal_strategy=strat_selection,
                withdrawal_rate=wd_rate, n_simulations=N_SIM, years=YEARS,
                inflation_rate=inflation, starting_age=retire_age,
                inheritance_goal=inheritance, cashflow_schedule=cf_schedule)
        st.session_state["res"]=res
        st.session_state["sim_strat"]=strat_selection
        st.session_state["wd_rate"]=wd_rate
        st.session_state.pop("export_pdf_bytes",None)
        st.session_state.pop("export_csv_bytes",None)

    # ---- RESULTS ----
    if "res" in st.session_state:
        res=st.session_state["res"]
        inh_goal=st.session_state.get("inheritance_goal",0.0)
        success    =res["survival_rate"]*100
        inh_rate   =res.get("inheritance_success_rate", -1.0)
        median_end =res.get("median_surviving", res["median_balance"][-1])

        st.divider()
        has_inh_goal = (inh_goal > 0 and inh_rate >= 0)
        if has_inh_goal:
            m1,m2,m3=st.columns(3)
        else:
            m1,m2=st.columns(2)

        color_surv="green" if success>85 else "red"
        m1.markdown(f"### Survival Rate: :{color_surv}[{success:.1f}%]")
        m1.caption("Chance money lasts > 30 years")
        m2.metric("Median End Balance (if survive)",f"{median_end:,.0f} THB")
        if has_inh_goal:
            inh_success = inh_rate * 100
            color_inh="green" if inh_success>50 else "orange"
            m3.markdown(f"### Inheritance Success: :{color_inh}[{inh_success:.1f}%]")
            m3.caption(f"Chance to leave ≥ {inh_goal:,.0f} THB")

        # ---- Wealth projection chart ----
        fig,ax=plt.subplots(figsize=(10,5))
        x=list(range(len(res["median_balance"])))

        p10 = res["percentile_10"]
        p25 = res["percentile_25"]
        p75 = res["percentile_75"]
        p90 = res["percentile_90"]
        med = res["median_balance"]

        ax.fill_between(x, p10, p90, alpha=0.10,
                        color='steelblue', label="10th–90th Percentile (extreme range)")
        ax.fill_between(x, p25, p75, alpha=0.30,
                        color='steelblue', label="25th–75th Percentile (likely range)")
        ax.plot(x, p10, color='steelblue', linewidth=0.7, linestyle=':', alpha=0.5)
        ax.plot(x, p90, color='steelblue', linewidth=0.7, linestyle=':', alpha=0.5)
        ax.plot(x, med, label="Median (50th Pctl)", color='steelblue', linewidth=2.5)
        ax.axhline(0, color='red', linestyle="--", linewidth=1.5, label="Portfolio Depleted (฿0)")
        if inh_goal > 0:
            ax.axhline(inh_goal, color='purple', linestyle="-.", linewidth=1.5,
                       label=f"Inheritance Goal ({inh_goal:,.0f} THB)")
        depletion_yr = next((i for i,v in enumerate(med) if v <= 0), None)
        if depletion_yr:
            ax.axvline(depletion_yr, color='orange', linestyle=':', linewidth=1.5,
                       label=f"Median depletes @ Year {depletion_yr}")

        ax.set_xlabel("Year from Retirement")
        ax.set_ylabel("Portfolio Value (THB)")
        ax.set_title("Wealth Projection — Monte Carlo Simulation (50,000 paths)")
        ax2 = ax.twiny(); ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(x[::5])
        ax2.set_xticklabels([f"Age {retire_age+i}" for i in x[::5]], fontsize=8)
        ax.legend(loc='upper right', fontsize=8)
        ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda v,p: f"{v/1e6:.1f}M" if abs(v)>=1e6 else format(int(v),',')))

        graph_with_info(fig,"Wealth Projection (Monte Carlo)",
            """**วิธีอ่านกราฟ Wealth Projection (Monte Carlo Simulation)**

📌 **กราฟนี้แสดงอะไร:**
จำลอง 50,000 สถานการณ์ตลาดที่เป็นไปได้ในอนาคต โดยใช้ผลตอบแทนและความผันผวนทางประวัติศาสตร์

📊 **ส่วนประกอบกราฟ:**
- **เส้นน้ำเงินหนา (Median Balance)** = มูลค่าพอร์ตที่น่าจะเป็นมากที่สุด
- **พื้นที่สีฟ้า (10th–90th Percentile)** = ช่วงผลลัพธ์ที่เป็นไปได้ 80% ของทุกสถานการณ์
- **เส้นแดงประ (0 THB)** = เส้นขีดล่าง หากมูลค่าพอร์ตต่ำกว่านี้ = เงินหมด
- **เส้นม่วง (Inheritance Goal)** = เป้าหมายมรดกที่คุณตั้งไว้""",
            key="wealthproj_info")

        # ---- Recommendations ----
        sim=RetirementSimulator()
        st.subheader("💡 Recommendations")
        if success<85:
            st.error(f"⚠️ Survival Rate ({success:.1f}%) is below 85% target.")
            recs=sim.recommend_improvements(
                current_survival_rate=res["survival_rate"],
                portfolio_allocation=alloc,
                withdrawal_rate=st.session_state.get("wd_rate",wd_rate))
            with st.expander("👉 View Action Plan",expanded=True):
                for r in recs: st.info(r)
        else:
            st.success("✅ Your plan looks solid! High chance of success.")

        # ---- Optimizer ----
        st.divider()
        if st.button("🔍 Find Optimal Withdrawal Rate"):
            with st.spinner("Optimizing..."):
                opt_rate=sim.find_optimal_withdrawal_rate(
                    initial_portfolio=start_port, portfolio_allocation=alloc,
                    asset_stats=asset_stats,
                    withdrawal_strategy=st.session_state.get("sim_strat","Basic Strategy"),
                    initial_rate=st.session_state.get("wd_rate",wd_rate),
                    years=YEARS, inflation_rate=inflation, starting_age=retire_age,
                    inheritance_goal=inheritance, cashflow_schedule=cf_schedule)
            curr=st.session_state.get("wd_rate",wd_rate); diff=opt_rate-curr
            co1,co2=st.columns(2)
            co1.metric("Current Rate",f"{curr*100:.2f}%")
            co2.metric("Optimal Rate",f"{opt_rate*100:.2f}%",f"{diff*100:.2f}%")
            if diff>0: st.success(f"🎉 You can safely increase your withdrawal by {diff*100:.2f}%!")
            else: st.warning(f"⚠️ Reduce your withdrawal by {abs(diff*100):.2f}% to be safe.")

    # ---- EXPORT ----
    st.divider()
    st.subheader("💾 Save Your Plan")

    if st.button("✅ Prepare Export Files"):
        res   =st.session_state.get("res")
        alloc =st.session_state.get("saved_alloc",{})
        ret_age_final =int(st.session_state.get("retire_age",60))
        life_exp_final=int(st.session_state.get("life_expectancy",85))
        total_income  =st.session_state.get("v_total_income",0.0)
        total_expense =st.session_state.get("v_total_expense",0.0)
        investable    =st.session_state.get("start_port",0.0)
        total_debt    =st.session_state.get("money_debt",0.0)

        # BUG FIX 1: use correct session-state keys for income sources
        # "inc_pension" not "inc_sal" — matches the timed_expense_input key used on page 1
        export_data={
            "name":             st.session_state.get("user_name","ไม่ระบุชื่อ"),
            "retire_age":       ret_age_final,
            "life_exp":         life_exp_final,
            "inflation":        st.session_state.get("inflation",0.03),
            "inheritance_goal": st.session_state.get("inheritance_goal",0.0),
            "sim_strat":        st.session_state.get("sim_strat", strat_selection),
            "wd_rate":          st.session_state.get("wd_rate", wd_rate),
            "total_income":     total_income,
            "total_expense":    total_expense,
            "yearly_savings":   total_income-total_expense,
            "investable":       investable,
            "total_debt":       total_debt,
            "net_worth":        investable-total_debt,
            # BUG FIX 1: "inc_pension" is the correct key (was "inc_sal" — key does not exist)
            "inc_detail": {
                "Pension":  get_annual_safe("inc_pension"),
                "Rental":   get_annual_safe("inc_rent"),
                "Dividend": get_annual_safe("inc_div"),
                "Other":    get_annual_safe("inc_other"),
            },
            "exp_fixed_detail": {
                "Housing":      {"amount": get_annual_safe("exp_house"),   "months_remaining": st.session_state.get("dur_exp_house")},
                "Insurance":    {"amount": get_annual_safe("exp_ins"),     "months_remaining": st.session_state.get("dur_exp_ins")},
                "Subscription": {"amount": get_annual_safe("exp_sub"),     "months_remaining": st.session_state.get("dur_exp_sub")},
                "Other Fixed":  {"amount": get_annual_safe("exp_fix_oth"), "months_remaining": st.session_state.get("dur_exp_fix_oth")},
            },
            "exp_var_detail": {
                "Transport":     get_annual_safe("exp_trans"),
                "Food":          get_annual_safe("exp_food"),
                "Entertain":     get_annual_safe("exp_ent"),
                "Travel":        get_annual_safe("exp_travel"),
                "Health":        get_annual_safe("exp_health"),
                "Other Variable":get_annual_safe("exp_var_oth"),
            },
            "asset_detail": {
                "Cash":          get_num("cash_dep"),
                "Thai Bond":     get_num("bond"),
                "Global Bond":   get_num("gl_bond"),
                "Thai Equity":   get_num("stock"),
                "Global Equity": get_num("gl_stock"),
                "PF&REIT":       get_num("reit"),
                "Global REIT":   get_num("gl_reit"),
                "Gold":          get_num("gold_invest"),
            },
            # BUG FIX 2: debt amounts use the timed_expense_input keys "debt_home",
            # "debt_car", "debt_cc", "debt_other" — NOT "debt_home_bal" etc. (those don't exist)
            # BUG FIX 3: months_remaining comes from the stored timed_debt_items list
            "debt_detail": {
                item["name"]: {
                    "amount":           item["annual_amount"],
                    "months_remaining": item["months_remaining"],
                }
                for item in st.session_state.get("timed_debt_items", [])
            },
            "alloc_weights":st.session_state.get("saved_alloc",{})
        }

        st.session_state["export_data"]      = export_data
        st.session_state["export_csv_bytes"] = build_full_report_csv(export_data, res, alloc)
        st.session_state["export_pdf_bytes"] = build_pdf_bytes(export_data, res)
        st.success("เตรียมไฟล์สำเร็จ! ✅")

    c1,c2=st.columns(2)
    with c1:
        st.download_button("📄 Download Full Report CSV",
            data=st.session_state.get("export_csv_bytes",b""),
            file_name="full_retirement_report.csv",mime="text/csv",
            disabled="export_csv_bytes" not in st.session_state)
    with c2:
        st.download_button("📕 Download Full Report PDF",
            data=st.session_state.get("export_pdf_bytes") or b"",
            file_name="full_retirement_report.pdf",mime="application/pdf",
            disabled="export_pdf_bytes" not in st.session_state)

    st.button("⬅ Back",on_click=prev_step)
