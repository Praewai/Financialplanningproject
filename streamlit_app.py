import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from matplotlib.backends.backend_pdf import PdfPages
import csv

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
This website was created by Financial Engineering Students not Financial Planner nor Investment Advisor and we do not have access to any non public information.
We cannot guarantee that the simulation will be 100% correct.
This was created solely for financial planner to use as an assistance for rough estimation and not to be use as a replacement of one.
We are not regulated by any Financial Service Authority.

เว็บไซต์นี้จัดทำขึ้นโดยนักศึกษาภาควิชาวิศวกรรมการเงิน (Financial Engineering) ไม่ใช่ผู้วางแผนการเงิน (Financial Planner) หรือที่ปรึกษาการลงทุน (Investment Advisor) และผู้จัดทำไม่ได้มีการเข้าถึงข้อมูลภายใน (Non-public information) ใดๆ ทั้งสิ้น
เราไม่สามารถรับรองได้ว่าผลจากการจำลอง (Simulation) จะถูกต้องแม่นยำ 100% เครื่องมือนี้ถูกสร้างขึ้นเพื่อใช้เป็นเครื่องช่วยคำนวณเบื้องต้นสำหรับผู้วางแผนการเงินเท่านั้น ไม่ควรนำไปใช้ทดแทนการวางแผนการเงินแบบเต็มรูปแบบ และเราไม่ได้อยู่ภายใต้การกำกับดูแลของหน่วยงานกำกับดูแลบริการทางการเงินใดๆ""")
    if st.button("I understand (รับทราบ)"):
        st.rerun()

if "accepted_terms" not in st.session_state:
    show_disclaimer()
    st.session_state["accepted_terms"] = True
# =========================================================
# CORE SIMULATION ENGINE
# =========================================================
class RetirementSimulator:
    def __init__(self):
        self.life_expectancy = {
            60: 24, 61: 23, 62: 22, 63: 21, 64: 20, 65: 19, 66: 18, 67: 17,
            68: 16, 69: 15, 70: 14, 71: 13, 72: 12, 73: 11, 74: 10, 75: 9,
            76: 8, 77: 7, 78: 6, 79: 5, 80: 4, 81: 3, 82: 2, 83: 1, 84: 1
        }

    def get_life_expectancy(self, current_age):
        if current_age in self.life_expectancy:
            return self.life_expectancy[current_age]
        elif current_age > max(self.life_expectancy.keys()):
            return 1
        else:
            return max(self.life_expectancy.values())

    def simulate_returns(self, portfolio_allocation, asset_stats, n_simulations, n_years):
        assets_list = [k for k in portfolio_allocation.keys() if k in asset_stats]
        if len(assets_list) == 0:
            return np.zeros((n_simulations, n_years))

        weights = np.array([portfolio_allocation[a] for a in assets_list], dtype=float)
        weights = weights / (weights.sum() if weights.sum() != 0 else 1.0)

        means = np.array([asset_stats[a]["mean"] for a in assets_list], dtype=float)
        stds  = np.array([asset_stats[a]["std"]  for a in assets_list], dtype=float)

        n_assets = len(assets_list)
        corr = np.eye(n_assets) + 0.4 * (np.ones((n_assets, n_assets)) - np.eye(n_assets))
        cov = np.outer(stds, stds) * corr

        portfolio_returns = np.zeros((n_simulations, n_years))
        for sim in range(n_simulations):
            asset_returns = np.random.multivariate_normal(means, cov, n_years)
            portfolio_returns[sim] = asset_returns @ weights
        return portfolio_returns

    # -------------------------
    # STRATEGIES (ALL return balances + withdrawals)
    # -------------------------
    def basic_strategy(self, initial_portfolio, withdrawal_rate, inflation_rate, returns, years):
        portfolio_value = initial_portfolio
        withdrawal = initial_portfolio * withdrawal_rate

        balances = [portfolio_value]
        withdrawals = []

        for year in range(years):
            withdrawals.append(max(0.0, withdrawal))
            portfolio_value -= withdrawal

            if portfolio_value <= 0:
                balances.extend([0.0] * (years - year))
                withdrawals.extend([0.0] * (years - 1 - year))
                break

            portfolio_value *= (1 + returns[year])
            balances.append(max(0.0, portfolio_value))
            withdrawal *= (1 + inflation_rate)

        return balances, withdrawals

    def forgoing_inflation_strategy(self, initial_portfolio, withdrawal_rate, inflation_rate, returns, years):
        portfolio_value = initial_portfolio
        withdrawal = initial_portfolio * withdrawal_rate

        balances = [portfolio_value]
        withdrawals = []
        prev_balance = portfolio_value

        for year in range(years):
            withdrawals.append(max(0.0, withdrawal))
            portfolio_value -= withdrawal

            if portfolio_value <= 0:
                balances.extend([0.0] * (years - year))
                withdrawals.extend([0.0] * (years - 1 - year))
                break

            portfolio_value *= (1 + returns[year])
            balances.append(max(0.0, portfolio_value))

            if portfolio_value > prev_balance:
                withdrawal *= (1 + inflation_rate)
            prev_balance = portfolio_value

        return balances, withdrawals

    def rmd_strategy(self, initial_portfolio, starting_age, returns, years):
        portfolio_value = initial_portfolio
        current_age = starting_age

        balances = [portfolio_value]
        withdrawals = []

        for year in range(years):
            life_exp = self.get_life_expectancy(current_age)
            withdrawal = portfolio_value / life_exp if life_exp > 0 else portfolio_value

            withdrawals.append(max(0.0, withdrawal))
            portfolio_value -= withdrawal

            if portfolio_value <= 0:
                balances.extend([0.0] * (years - year))
                withdrawals.extend([0.0] * (years - 1 - year))
                break

            portfolio_value *= (1 + returns[year])
            balances.append(max(0.0, portfolio_value))
            current_age += 1

        return balances, withdrawals

    def guardrails_strategy(self, initial_portfolio, withdrawal_rate, inflation_rate, returns, years):
        portfolio_value = initial_portfolio
        withdrawal = initial_portfolio * withdrawal_rate
        initial_rate = withdrawal_rate

        balances = [portfolio_value]
        withdrawals = []

        for year in range(years):
            withdrawals.append(max(0.0, withdrawal))
            portfolio_value -= withdrawal

            if portfolio_value <= 0:
                balances.extend([0.0] * (years - year))
                withdrawals.extend([0.0] * (years - 1 - year))
                break

            portfolio_value *= (1 + returns[year])
            balances.append(max(0.0, portfolio_value))

            current_rate = withdrawal / portfolio_value if portfolio_value > 0 else 0.0
            if current_rate < initial_rate * 0.8:
                withdrawal *= 1.10
            elif current_rate > initial_rate * 1.2:
                withdrawal *= 0.90
            else:
                withdrawal *= (1 + inflation_rate)

        return balances, withdrawals

    # -------------------------
    # RUN SIMULATION (pad arrays properly)
    # -------------------------
    def run_simulation(
        self,
        initial_portfolio,
        portfolio_allocation,
        asset_stats,
        withdrawal_strategy,
        withdrawal_rate,
        years,
        inflation_rate,
        starting_age,
        returns_override=None, 
        inheritance_goal=0.0,
        n_simulations=50000
        ):
        returns = returns_override if returns_override is not None else self.simulate_returns(
        portfolio_allocation, asset_stats, n_simulations, years
        )

        strategy_map = {
            "Basic Strategy": self.basic_strategy,
            "Forgoing Inflation": self.forgoing_inflation_strategy,
            "RMD Strategy": self.rmd_strategy,
            "Guardrails": self.guardrails_strategy,
        }

        all_balances = []
        all_withdrawals = []

        for sim in range(n_simulations):
            if withdrawal_strategy == "RMD Strategy":
                balances, wds = strategy_map[withdrawal_strategy](
                initial_portfolio, starting_age, returns[sim], years
                )
            else:
                balances, wds = strategy_map[withdrawal_strategy](
                    initial_portfolio, withdrawal_rate, inflation_rate, returns[sim], years
                )

            if len(balances) < years + 1:
                balances = balances + [0.0] * ((years + 1) - len(balances))
            else:
                balances = balances[: years + 1]

            if len(wds) < years:
                wds = wds + [0.0] * (years - len(wds))
            else:
                wds = wds[:years]

            all_balances.append(balances)
            all_withdrawals.append(wds)

        all_balances = np.array(all_balances, dtype=float)       # (sim, years+1)
        all_withdrawals = np.array(all_withdrawals, dtype=float) # (sim, years)

        final_values = all_balances[:, -1]

        return {
            "survival_rate": float(np.mean(final_values > 0)),
            "inheritance_success_rate": float(np.mean(final_values >= inheritance_goal)),
            "median_balance": np.median(all_balances, axis=0),
            "percentile_10": np.percentile(all_balances, 10, axis=0),
            "percentile_90": np.percentile(all_balances, 90, axis=0),
            "returns_mean": float(np.mean(returns)),

            "median_withdrawal": np.median(all_withdrawals, axis=0),
            "withdrawal_p10": np.percentile(all_withdrawals, 10, axis=0),
            "withdrawal_p90": np.percentile(all_withdrawals, 90, axis=0),
        }

    # -------------------------
    # RECOMMENDATIONS (fix key mismatch)
    # -------------------------
    def recommend_improvements(self, current_survival_rate, portfolio_allocation, withdrawal_rate, min_survival_rate=0.85):
        recs = []
        if current_survival_rate >= min_survival_rate:
            return ["✅ Your strategy meets the target survival rate!"]

        if withdrawal_rate > 0.03:
            rec_rate = withdrawal_rate * 0.9
            recs.append(f"📉 **Reduce Spending:** Try lowering withdrawal from {withdrawal_rate*100:.1f}% to {rec_rate*100:.1f}%.")

        # equity-ish keys in YOUR alloc
        equity_keys = ["pct_seti", "pct_msci_stock","pct_REITTH","pct_MSCIREITs"]
        equity_weight = sum(float(portfolio_allocation.get(k, 0)) for k in equity_keys)

        if equity_weight < 0.4:
            recs.append(f"📈 **Increase Growth:** Equity allocation seems low ({equity_weight*100:.0f}%). Consider 40–60%.")
        elif equity_weight > 0.8:
            recs.append(f"🛡️ **Reduce Risk:** Equity allocation seems high ({equity_weight*100:.0f}%). Consider adding bonds/cash.")

        recs.append("🔄 **Change Strategy:** Try 'Guardrails' or 'Forgoing Inflation' to adapt during drawdowns.")

        deficit = min_survival_rate - current_survival_rate
        recs.append(f"💰 **Save More:** Consider increasing initial portfolio or reducing spending; gap to target ≈ {(deficit*100):.1f}%.")

        return recs

    # -------------------------
    # OPTIMIZER (wd_rate only)
    # -------------------------
    def find_optimal_withdrawal_rate(
        self,
        initial_portfolio,
        portfolio_allocation,
        asset_stats,
        withdrawal_strategy,
        initial_rate,
        years,
        inflation_rate,
        starting_age,
        min_survival_rate=0.85,
        n_simulations=50000,
    ):
        low_rate = 0.01
        high_rate = min(0.12, max(0.06, initial_rate * 2))
        tolerance = 0.001
        best_rate = initial_rate
        max_iterations = 20

        for _ in range(max_iterations):
            if (high_rate - low_rate) <= tolerance:
                break
            test_rate = (low_rate + high_rate) / 2
            results = self.run_simulation(
                initial_portfolio,
                portfolio_allocation,
                asset_stats,
                withdrawal_strategy,
                test_rate,
                years,
                inflation_rate,
                starting_age,
                n_simulations=n_simulations
            )
            if results["survival_rate"] >= min_survival_rate:
                best_rate = test_rate
                low_rate = test_rate
            else:
                high_rate = test_rate

        return best_rate
    
    def sensitivity_withdrawal_rate(
        self,
        initial_portfolio,
        portfolio_allocation,
        asset_stats,
        withdrawal_strategy,
        wd_grid,
        years,
        inflation_rate,
        starting_age,
        n_simulations=50000,
    ):
        results = []
        for wd in wd_grid:
            res = self.run_simulation(
                initial_portfolio=initial_portfolio,
                portfolio_allocation=portfolio_allocation,
                asset_stats=asset_stats,
                withdrawal_strategy=withdrawal_strategy,
                withdrawal_rate=wd,
                n_simulations=n_simulations,
                years=years,
                inflation_rate=inflation_rate,
                starting_age=starting_age,
                returns_override=None,  # ✅ reuse
            )
            results.append({
                "withdrawal_rate": wd,
                "survival_rate": res["survival_rate"],
                "median_end_balance": float(res["median_balance"][-1]),
            })
        return results
# =========================================================
# UI HELPER FUNCTIONS
# =========================================================
if "current_step" not in st.session_state:
    st.session_state["current_step"] = 0

steps = ["👤 1. ข้อมูลผู้ใช้", "🧩 2.แบบประเมินความเสี่ยง", "📊 3.การจัดสรรสินทรัพย์", "💸 4. กลยุทธ์การถอนเงิน"]

def update_nav():
    st.session_state["nav_radio"] = steps[st.session_state["current_step"]]

def next_step():
    if st.session_state["current_step"] < len(steps) - 1:
        st.session_state["current_step"] += 1
        update_nav()

def prev_step():
    if st.session_state["current_step"] > 0:
        st.session_state["current_step"] -= 1
        update_nav()

def jump_step():
    st.session_state["current_step"] = steps.index(st.session_state["nav_radio"])

def money_input(label, default_val, key_suffix):
    # --- KEYS ---
    data_key = f"v_{key_suffix}"    # Stores Float (e.g., 1000000.0)
    fmt_key = f"fmt_{key_suffix}"   # Stores String (e.g., "1,000,000")
    ui_key = f"ui_{key_suffix}"     # Widget Key (Temporary UI)

    # --- INITIALIZATION (Run once) ---
    if data_key not in st.session_state:
        val = float(default_val)
        st.session_state[data_key] = val
        st.session_state[fmt_key] = f"{val:,.0f}"

    # --- SYNC FUNCTION (The Magic Fix) ---
    def on_change():
        # 1. Get what the user typed
        user_input = st.session_state.get(ui_key, "0")
        
        try:
            # 2. Clean it (remove commas, spaces)
            clean_val = float(str(user_input).replace(",", "").strip())
        except:
            clean_val = 0.0
            
        # 3. Save the Float (for calculations)
        st.session_state[data_key] = clean_val
        
        # 4. Format it with commas (for display)
        formatted_str = f"{clean_val:,.0f}"
        st.session_state[fmt_key] = formatted_str
        
        # 5. FORCE THE WIDGET TO UPDATE IMMEDIATELY
        # This makes the comma appear instantly when you press Enter
        st.session_state[ui_key] = formatted_str

    # --- RENDER WIDGET ---
    st.text_input(
        label,
        value=st.session_state[fmt_key],  # Load saved formatted text
        key=ui_key,                       # Unique UI key
        on_change=on_change
    )
    
    return st.session_state[data_key]

def pct_input(label, key):
    # 1. PERMANENT STORAGE KEY
    data_key = f"p_{key}"
    
    # Initialize if missing
    if data_key not in st.session_state:
        st.session_state[data_key] = 0.0

    # 2. SYNC FUNCTION
    def on_change():
        widget_key = f"ui_{key}"
        # Copy widget value to permanent storage
        st.session_state[data_key] = st.session_state[widget_key]

    # 3. RENDER WIDGET
    st.number_input(
        f"{label} (%)",
        min_value=0.0,
        max_value=100.0,
        value=float(st.session_state[data_key]), # <--- Loads saved data
        step=5.0,
        key=f"ui_{key}",  # <--- Unique UI key
        format="%.1f",
        on_change=on_change
    )
    
    return st.session_state[data_key]
def get_val_num(key_suffix):
    return float(st.session_state.get(f"v_{key_suffix}", 0.0))
def get_num(key_suffix):
    # This grabs the PERMANENT data key we created above
    return float(st.session_state.get(f"v_{key_suffix}", 0.0))
def name_input(label, key):
    # 1. PERMANENT STORAGE KEY (e.g., "v_user_name")
    data_key = f"v_{key}"
    widget_key = f"ui_{key}"

    # Initialize if missing
    if data_key not in st.session_state:
        st.session_state[data_key] = ""

    # 2. SYNC FUNCTION (Widget -> Storage)
    def on_change():
        st.session_state[data_key] = st.session_state[widget_key]

    # 3. RENDER WIDGET
    st.text_input(
        label,
        value=st.session_state[data_key],  # <--- Loads saved name
        key=widget_key,                    # <--- Unique UI key
        on_change=on_change
    )
    return st.session_state[data_key]
def build_full_report_csv(export_data, res, alloc, years=30):
    def fnum(x, nd=2, default=0.0):
        try:
            return f"{float(x or 0):,.{nd}f}"
        except:
            return f"{default:,.{nd}f}"

    def fpct(x, nd=2):
        try:
            return f"{float(x or 0)*100:.{nd}f}%"
        except:
            return ""

    def to_int(x, default=None):
        try:
            return int(float(x))
        except:
            return default

    ASSET_LABELS = {
        "pct_deposit": "Fixed Deposit",
        "pct_gov_bond": "Thai Gov Bond 1Y",
        "pct_seti": "SET Index",
        "pct_XAUTHB": "Gold (THB)",
        "pct_REITTH": "Thai REIT",
        "pct_msci_stock": "MSCI World Equity",
        "pct_msci_gov_bond": "MSCI Gov Bond",
        "pct_XAUUSD": "Gold (USD)",
        "pct_MSCIREITs": "Global REIT",
    }

    rows = []

    # =========================================================
    # SECTION A: PROFILE & SETTINGS
    # =========================================================
    rows.append(["SECTION", "FIELD", "VALUE"])
    rows.append(["PROFILE", "Name", export_data.get("name", "ไม่ระบุชื่อ")])
    rows.append(["PROFILE", "Retire Age", export_data.get("retire_age", "")])
    rows.append(["PROFILE", "Life Expectancy", export_data.get("life_exp", "")])
    rows.append(["PROFILE", "Inheritance Goal (THB)", fnum(export_data.get("inheritance_goal"), 2)])
    rows.append(["SETTINGS", "Inflation", fpct(export_data.get("inflation"))])
    rows.append([])

    # =========================================================
    # SECTION B: INCOME DETAIL -> TOTAL
    # =========================================================
    rows.append(["SECTION", "INCOME ITEM", "YEARLY (THB)", "MONTHLY (THB)"])
    inc_items = export_data.get("inc_detail", {})
    for item, val in inc_items.items():
        if val > 0:
            rows.append(["INCOME_DETAIL", item, fnum(val, 2), fnum(val/12, 2)])
    
    total_inc = export_data.get("total_income", 0.0)
    rows.append(["INCOME_SUMMARY", "TOTAL INCOME", fnum(total_inc, 2), fnum(total_inc/12, 2)])
    rows.append([])

    # =========================================================
    # SECTION C: EXPENSE DETAIL -> TOTAL
    # =========================================================
    rows.append(["SECTION", "EXPENSE ITEM", "YEARLY (THB)", "MONTHLY (THB)"])
    
    # Fixed Expenses
    fixed_items = export_data.get("exp_fixed_detail", {})
    for item, val in fixed_items.items():
        if val > 0:
            rows.append(["EXPENSE_FIXED", item, fnum(val, 2), fnum(val/12, 2)])
            
    # Variable Expenses
    var_items = export_data.get("exp_var_detail", {})
    for item, val in var_items.items():
        if val > 0:
            rows.append(["EXPENSE_VARIABLE", item, fnum(val, 2), fnum(val/12, 2)])

    total_exp = export_data.get("total_expense", 0.0)
    net_save = export_data.get("yearly_savings", 0.0)
    rows.append(["EXPENSE_SUMMARY", "TOTAL EXPENSE", fnum(total_exp, 2), fnum(total_exp/12, 2)])
    rows.append(["CASHFLOW", "NET SAVINGS (Surplus/Deficit)", fnum(net_save, 2), fnum(net_save/12, 2)])
    rows.append([])

    # =========================================================
    # SECTION D: ASSETS & DEBT -> NET WORTH
    # =========================================================
    rows.append(["SECTION", "ASSET/DEBT ITEM", "VALUE (THB)"])
    asset_items = export_data.get("asset_detail", {})
    for item, val in asset_items.items():
        if val > 0:
            rows.append(["ASSET_DETAIL", item, fnum(val, 2)])
    
    investable = export_data.get("investable", 0.0)
    rows.append(["ASSET_SUMMARY", "TOTAL INVESTABLE ASSETS", fnum(investable, 2)])
    
    debt_items = export_data.get("debt_detail", {})
    for item, val in debt_items.items():
        if val > 0:
            rows.append(["DEBT_DETAIL", item, fnum(val, 2)])
            
    total_debt = export_data.get("total_debt", 0.0)
    rows.append(["DEBT_SUMMARY", "TOTAL DEBT", fnum(total_debt, 2)])
    
    rows.append(["SUMMARY", "NET WORTH", fnum(export_data.get("net_worth"), 2)])
    rows.append([])

    # =========================================================
    # SECTION E: SIMULATION & ALLOCATION
    # =========================================================
    sim_strat = export_data.get("sim_strat", "-")
    wd_rate = export_data.get("wd_rate", None)
    rows.append(["SIMULATION", "Strategy", sim_strat])
    if wd_rate is not None:
        rows.append(["SIMULATION", "Withdrawal Rate", fpct(wd_rate)])

    if res is not None:
        rows.append(["SIMULATION", "Survival Rate", f"{res['survival_rate']*100:.1f}%"])
        rows.append(["SIMULATION", "Median End Balance (Year 30)", fnum(res["median_balance"][-1], 0)])

    rows.append([])
    rows.append(["SECTION", "ASSET ALLOCATION", "WEIGHT (%)"])
    if alloc:
        for k, v in alloc.items():
            label = ASSET_LABELS.get(k, k)
            rows.append(["ALLOCATION", label, f"{float(v)*100:.2f}%"])
    
    # =========================================================
    # SECTION F: YEARLY PROJECTION
    # =========================================================
    rows.append([])
    rows.append(["YEARLY PROJECTION (30Y)"])
    rows.append(["Year", "Age", "Median_Balance", "P10_Balance", "P90_Balance", "Median_Withdrawal", "P10_Withdrawal", "P90_Withdrawal", "P10_Depleted_Flag"])

    retire_age_int = to_int(export_data.get("retire_age"), 60)
    if res is not None:
        mb, p10b, p90b = res.get("median_balance"), res.get("percentile_10"), res.get("percentile_90")
        mw, p10w, p90w = res.get("median_withdrawal"), res.get("withdrawal_p10"), res.get("withdrawal_p90")

        for y in range(1, years + 1):
            age = retire_age_int + (y - 1)
            rows.append([
                y, age, round(mb[y], 2), round(p10b[y], 2), round(p90b[y], 2),
                round(mw[y-1], 2), round(p10w[y-1], 2), round(p90w[y-1], 2),
                1 if p10b[y] <= 0 else 0
            ])

    out = io.StringIO()
    csv.writer(out).writerows(rows)
    return out.getvalue().encode("utf-8-sig")

def build_pdf_bytes(data, res):
    
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:

        # Page 1
        import matplotlib.pyplot as plt
        f1, a1 = plt.subplots(figsize=(8, 11))
        a1.axis("off")
        a1.set_title("Financial Report", fontsize=18, fontweight="bold", pad=20)

        y = 0.85
        a1.text(0.1, y, f"Name: {data['name']}", fontsize=12, fontweight="bold"); y -= 0.03
        a1.text(0.1, y, f"Retire Age: {data['retire_age']} | Life Exp: {data['life_exp']}", fontsize=11); y -= 0.05

        a1.text(0.1, y, "SUMMARY", fontsize=12, fontweight="bold"); y -= 0.04
        a1.text(0.1, y, f"Investable Assets: {data['investable']:,.0f} THB", fontsize=11); y -= 0.03
        a1.text(0.1, y, f"Total Debt: {data['total_debt']:,.0f} THB", fontsize=11); y -= 0.03
        a1.text(0.1, y, f"Yearly Savings: {data['yearly_savings']:,.0f} THB", fontsize=11); y -= 0.03
        a1.text(0.1, y, f"Net Worth: {data['net_worth']:,.0f} THB", fontsize=11); y -= 0.05

        details = [
            ["Total Income", f"{data['total_income']:,.0f}"],
            ["Total Expense", f"{data['total_expense']:,.0f}"],
        ]
        t1 = a1.table(cellText=details, colLabels=["Item", "THB"], bbox=[0.1, 0.35, 0.8, 0.18])
        t1.auto_set_font_size(False); t1.set_fontsize(10)
        pdf.savefig(f1); plt.close(f1)

        # Page 2
        if res is not None:
            f2, a2 = plt.subplots(figsize=(8, 11))
            a2.axis("off")
            a2.set_title("Simulation Results", fontsize=16, pad=20)

            sim_table = [
                ["Strategy", data.get("sim_strat", "-")],
                ["Withdrawal Rate", f"{data.get('wd_rate', 0)*100:.2f}%"],
                ["Success Rate", f"{res['survival_rate']*100:.1f}%"],
                ["Median End Balance", f"{res['median_balance'][-1]:,.0f} THB"],
            ]
            t2 = a2.table(cellText=sim_table, colLabels=["Metric", "Result"], bbox=[0.1, 0.73, 0.8, 0.20])
            t2.auto_set_font_size(False); t2.set_fontsize(10)

            ax = f2.add_axes([0.1, 0.12, 0.8, 0.50])
            x = range(len(res["median_balance"]))
            ax.fill_between(x, res["percentile_10"], res["percentile_90"], alpha=0.2)
            ax.plot(x, res["median_balance"])
            ax.set_title("Wealth Projection")
            ax.set_xlabel("Year")
            ax.set_ylabel("Portfolio Value (THB)")

            pdf.savefig(f2); plt.close(f2)

    return buf.getvalue()

def parse_bloomberg_file(uploaded_file):
    try:
            filename = uploaded_file.name.lower()
            header_idx = None
            df_raw = None
            
            # A. Read raw rows to find the Header
            if filename.endswith(('.xlsx', '.xls')):
                df_raw = pd.read_excel(uploaded_file, header=None, nrows=20)
            else:
                df_raw = pd.read_csv(uploaded_file, header=None, nrows=20)

            # --- HEADER SEARCH ---
            # Look for row containing "Date" and a price keyword
            for r, row in df_raw.iterrows():
                row_text = row.astype(str).str.upper().str.cat(sep=' ')
                if "DATE" in row_text and any(k in row_text for k in ['PX', 'LAST', 'PRICE', 'TOT', 'RETURN', 'GROSS']):
                    header_idx = r
                    break
            
            if header_idx is None: return None, "No 'Date' column found."

            # B. Reload full file with correct header
            uploaded_file.seek(0)
            if filename.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file, header=header_idx)
            else:
                df = pd.read_csv(uploaded_file, header=header_idx)

            # --- DEBUG: Show found columns ---
            # (This helps us see if the column name is weird)
            print(f"[{filename}] Found Columns: {list(df.columns)}")

            # C. Identify Columns
            
            # 1. Find Date Column
            date_col = next((c for c in df.columns if "date" in str(c).lower()), None)
            
            # 2. Find Price Column (UPDATED PRIORITY)
            price_col = None
            
            # Priority 1: TOTAL RETURN / GROSS RETURN (The Gold Standard)
            # We check for "TOT" or "GROSS" combined with "RETURN" or "INDEX"
            for c in df.columns:
                c_up = str(c).upper()
                # Catch: TOT_RETURN, TOT_RETURN_INDEX_GROSS_DVDS, GROSS_RETURN
                if ("TOT" in c_up or "GROSS" in c_up) and ("RETURN" in c_up or "INDEX" in c_up):
                    price_col = c
                    break
            
            # Priority 2: Fallback to simple "TOT_RETURN" if complex match failed
            if not price_col:
                for c in df.columns:
                    if "TOT_RETURN" in str(c).upper():
                        price_col = c
                        break

            # Priority 3: Standard Price (Last Resort)
            if not price_col:
                for c in df.columns:
                    c_up = str(c).upper()
                    if "PX" in c_up or "LAST" in c_up or "CLOSE" in c_up or "PRICE" in c_up:
                        price_col = c
                        break

            if not date_col or not price_col: 
                return None, f"Columns missing. Found: {list(df.columns)}"

            # D. Clean & Sort
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df.set_index(date_col, inplace=True)
            
            series = pd.to_numeric(df[price_col], errors='coerce').dropna().sort_index()
            
            # Set name to show user EXACTLY what we picked
            series.name = f"{filename} ({price_col})"
            
            return series, None

    except Exception as e: 
        return None, str(e)
# =========================================================
# NAV BAR
# =========================================================
if "nav_radio" not in st.session_state:
    st.session_state["nav_radio"] = steps[0]

st.radio("Go to:", steps, key="nav_radio", horizontal=True, label_visibility="collapsed", on_change=jump_step)
st.progress((st.session_state["current_step"] + 1) / len(steps))
st.divider()
# ========================================================
# PAGE 1: FINANCIAL HEALTH 
# =========================================================
if st.session_state["current_step"] == 0:
    st.header("👤 1. ข้อมูลผู้ใช้ (Financial Health)")
    st.subheader("A. ข้อมูลส่วนตัว")
    if "user_name" not in st.session_state:
        st.session_state["user_name"] = ""
    if "retire_age" not in st.session_state:
        st.session_state["retire_age"] = 60
    if "life_expectancy" not in st.session_state:
        st.session_state["life_expectancy"] = 85
        
    if "ui_user_name" not in st.session_state:
        st.session_state["ui_user_name"] = st.session_state["user_name"]
    if "ui_retire_age" not in st.session_state:
        st.session_state["ui_retire_age"] = int(st.session_state["retire_age"])
    if "ui_life_expectancy" not in st.session_state:
        st.session_state["ui_life_expectancy"] = int(st.session_state["life_expectancy"])

    def validate_ages():
        st.session_state["retire_age"] = int(st.session_state.get("ui_retire_age", 60))
    st.session_state["life_expectancy"] = int(st.session_state.get("ui_life_expectancy", 85))

    # enforce rule
    if st.session_state["life_expectancy"] < st.session_state["retire_age"]:
        st.session_state["life_expectancy"] = st.session_state["retire_age"]
        st.session_state["ui_life_expectancy"] = st.session_state["life_expectancy"]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.text_input(
            "ชื่อ", 
            value=st.session_state["user_name"], 
            key="ui_user_name",
            on_change=lambda: st.session_state.update({"user_name": st.session_state["ui_user_name"]})
        )
    with c2:
        st.number_input(
        "อายุเกษียณ",
        min_value=40,
        max_value=100,
        value=int(st.session_state["ui_retire_age"]),   # ✅ IMPORTANT
        key="ui_retire_age",
        on_change=validate_ages,
    )

    with c3:
        st.number_input(
            "อายุขัย",
            min_value=int(st.session_state["ui_retire_age"]),  # ✅ bind to UI value
            max_value=120,
            value=int(st.session_state["ui_life_expectancy"]), # ✅ IMPORTANT
            key="ui_life_expectancy",
            on_change=validate_ages,
        )
    # Assets
    st.subheader("B. ทรัพย์สิน (Investable Assets Only)")
    with st.expander("📝 รายละเอียดทรัพย์สิน", expanded=True):
        st.markdown("💰 สินทรัพย์เพื่อการลงทุน")
        i1, i2 = st.columns(2)
        with i1:
            money_cash = money_input("เงินสด/เงินฝาก (Cash)", 0, "cash_dep")
            money_bond = money_input("ตราสารหนี้ (Bond)", 0, "bond")
        with i2:
            money_stock = money_input("หุ้นไทย (Thai Equity)", 0, "stock")
            money_glstock = money_input("หุ้นต่างประเทศ (Global Equity)", 0, "gl_stock")
            other_invest = money_input("ทองคำ/ทรัพย์สินเพื่อการลงทุนอื่นๆ (Gold/Alternative)", 0, "other_invest")
    investable_assets = money_cash + money_bond + money_stock + money_glstock + other_invest
    st.metric("💰 รวมเงินลงทุนทั้งหมด",f"{investable_assets:,.0f}")
    st.session_state["start_port"] = investable_assets

    # Debt
    st.subheader("C. หนี้สิน (Debt)")
    with st.expander("📝 รายละเอียดหนี้สินรวม", expanded=True):
        st.markdown("💳 หนี้สินต่างๆ")
        lc1, lc2 = st.columns(2)
        with lc1:
            debt_home = money_input("หนี้บ้าน", 0, "debt_home")
            debt_car = money_input("หนี้รถ", 0, "debt_car")
        with lc2:
            debt_cc = money_input("บัตรเครดิต", 0, "debt_cc")
            debt_other = money_input("หนี้สินอื่น", 0, "debt_other")
        total_debt = debt_home + debt_car + debt_cc + debt_other
    st.metric("💳 หนี้สินรวมทั้งหมด", f"{total_debt:,.0f}")

    st.subheader("D. กระแสเงินสด (Cash Flow) — หลังเกษียณ")
    # --- 1. DEFINE THE HELPER FUNCTION ---
    def cashflow_input(label, key_suffix):
        """
        Creates a aligned row: [Label] | [Money Input] | [Freq Select]
        Matches the style of your 'money_input' but adds a frequency toggle.
        """
        # Create 3 columns with vertical centering (keeps everything straight)
        c_lbl, c_inp, c_frq = st.columns([2,2,2], vertical_alignment="center")
        
        with c_lbl:
            st.markdown(f"{label}")
            
        with c_inp:
            # Calls YOUR existing money_input function
            # We pass "" as the label because we already showed it in the left column
            amount = money_input("", 0, key_suffix) 
            
        with c_frq:
            freq_key = f"freq_{key_suffix}"
            if freq_key not in st.session_state:
                st.session_state[freq_key] = "ต่อเดือน (Monthly)"
            
            freq = st.radio(
                "", 
                ["ต่อเดือน (Monthly)", "ต่อปี (Yearly)"],
                horizontal=True,
                key=freq_key, 
                label_visibility="collapsed"
            )
        
        # Calculate Annual Amount immediately
        if "Monthly" in freq:
            return float(amount * 12)
        else:
            return float(amount)

    # --- 2. INCOME SECTION (Clean & Simple) ---
    with st.expander(" 📥 1. รายได้ (Income)"):
        inc_pension = cashflow_input("เงินบำนาญ (Pension)", "inc_sal")
        inc_rent    = cashflow_input("ค่าเช่า (Rental)", "inc_rent")
        inc_div     = cashflow_input("ดอกเบี้ย/ปันผล (Dividend)", "inc_div")
        inc_other   = cashflow_input("รายได้อื่นๆ (Other)", "inc_other")

        # Calculations
        total_income = inc_pension + inc_rent + inc_div + inc_other
        total_income_mo = total_income / 12
    st.success(f"💰 **รวมรายได้:** {total_income:,.0f} บาท/ปี (เฉลี่ย {total_income_mo:,.0f} บาท/เดือน)")

    # --- 3. EXPENSE SECTION (Clean & Simple) ---
    with st.expander("💸 2. รายจ่าย (Expenses)"):
        st.markdown("🔹 รายจ่ายคงที่ (Fixed)")
        exp_loan  = cashflow_input("ค่าผ่อนรถ/บ้าน (Loan)", "exp_loan")
        exp_house = cashflow_input("ค่าที่อยู่ (Housing)", "exp_house")
        exp_ins   = cashflow_input("ประกัน (Insurance)", "exp_ins")
        exp_sub   = cashflow_input("Subscription", "exp_sub")
        exp_fix   = cashflow_input("อื่นๆ (Other Fixed)", "exp_fix_oth")
        
        total_fixed = exp_loan + exp_house + exp_ins + exp_sub + exp_fix
        st.info(f"รวม Fixed: {total_fixed:,.0f} บาท/ปี (เฉลี่ย {total_fixed/12:,.0f} บาท/เดือน)")
        
        st.markdown("---") 

        # Variable Expenses
        st.markdown("🔸 รายจ่ายผันแปร (Non-Fixed)")
        exp_trans  = cashflow_input("ค่าเดินทาง (Transport)", "exp_trans")
        exp_food   = cashflow_input("ค่าอาหาร (Food)", "exp_food")
        exp_ent    = cashflow_input("สันทนาการ (Entertain)", "exp_ent")
        exp_travel = cashflow_input("ท่องเที่ยว (Travel)", "exp_travel")
        exp_health = cashflow_input("รักษาพยาบาล (Health)", "exp_health")
        exp_var    = cashflow_input("อื่นๆ (Other Variable)", "exp_var_oth")
        
        total_variable = exp_trans + exp_food + exp_ent + exp_travel + exp_health + exp_var
        st.info(f"รวม Variable: {total_variable:,.0f} บาท/ปี (เฉลี่ย {total_variable/12:,.0f} บาท/เดือน)")

        # Total Expense
        total_expense = total_fixed + total_variable
        total_expense_mo = total_expense / 12

    st.error(f"📉 **รวมรายจ่าย:** {total_expense:,.0f} บาท/ปี (เฉลี่ย {total_expense_mo:,.0f} บาท/เดือน)")

    # --- 4. SAVE RESULTS ---
    st.session_state["v_total_income"] = total_income
    st.session_state["v_total_expense"] = total_expense
    st.session_state["v_net_cashflow"] = total_income - total_expense
    
    yearly_savings = total_income - total_expense
    net_worth = investable_assets - total_debt

    st.markdown("### 📊 สรุปสถานะการเงิน (หลังเกษียณ)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("มูลค่าสุทธิ (Net Worth)", f"{net_worth:,.0f}")
    m2.metric("เงินลงทุนได้ (Investable)", f"{investable_assets:,.0f}")
    m3.metric("เงินคงเหลือ/ปี (Income-Expense)", f"{yearly_savings:,.0f}")
    m4.metric("หนี้สินรวม", f"{total_debt:,.0f}")

    if yearly_savings < 0:
        st.warning(f"⚠️ รายจ่ายมากกว่ารายได้ {abs(yearly_savings):,.0f} บาท/ปี (ยังสามารถจำลองต่อได้)")

    # store core
    st.session_state["start_port"] = investable_assets
    st.session_state["money_save"] = yearly_savings
    st.session_state["money_debt"] = total_debt 
    
    st.session_state["v_cash_dep"] = money_cash
    st.session_state["v_bond"] = money_bond
    st.session_state["v_stock"] = money_stock
    st.session_state["v_gl_stock"] = money_glstock
    st.session_state["v_other_invest"] = other_invest

    # inflation 
    st.session_state["inflation"] = st.slider(
        "เงินเฟ้อคาดการณ์ (%)",0.0, 10.0, 
        st.session_state.get("inflation", 0.03) * 100,0.1) / 100

    st.subheader("Inheritance Goals")
    with st.expander("📝 เป้าหมายมรดกที่ต้องการ", expanded=True):
        st.session_state["inheritance_goal"] = money_input("จำนวนเงินที่ต้องการ (THB)", 0, "inheritance_goal")
        
    c_nav1,c_nav2 = st.columns([10, 3])
    with c_nav2:
        st.button("Next Step ➡", on_click=next_step, type="primary")
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

    # --- HELPER: PERSISTENT RADIO BUTTON ---
    def persistent_radio(key_suffix, options, label):
        # 1. DEFINE KEYS
        idx_key = f"idx_{key_suffix}"  # Stores the Integer Index (0, 1, 2)
        ui_key = f"ui_{key_suffix}"    # Widget Key

        # 2. SYNC FUNCTION
        def on_change():
            # Get the selected object (dict)
            selected_obj = st.session_state[ui_key]
            # Find its index in the options list
            try:
                new_idx = options.index(selected_obj)
            except:
                new_idx = None
            # Save the index permanently
            st.session_state[idx_key] = new_idx

        # 3. RENDER WIDGET
        # Retrieve saved index (default to None if not found)
        saved_idx = st.session_state.get(idx_key, None)

        return st.radio(
            label,
            options,
            format_func=lambda x: x["label"],
            index=saved_idx,    # <--- Loads previous selection
            key=ui_key,         # Unique UI key
            on_change=on_change,
            label_visibility="collapsed"
        )

    # --- RENDER QUESTIONS ---
    total_score = 0
    all_answered = True
    
    for i, item in enumerate(questions_data):
        st.subheader(item["q"])
        
        # Use our new persistent helper
        choice = persistent_radio(f"q_{i}", item["choices"], f"Radio_{i}")
        
        st.divider()
        
        if choice is None:
            all_answered = False
        else:
            total_score += int(choice["score"])

    # --- SCORING ---
    if all_answered:
        if total_score >= 26:
            profile = "Aggressive (เชิงรุก)"
        elif total_score >= 16:
            profile = "Moderate (ปานกลาง)"
        else:
            profile = "Conservative (ระมัดระวัง)"
        st.success(f"คะแนน: {total_score} - {profile}")
        
        # Save profile for later use
        st.session_state["risk_profile"] = profile
        st.session_state["risk_score"] = total_score

    c1, c2 = st.columns([1, 8])
    with c1:
        st.button("⬅ Back", on_click=prev_step)
    with c2:
        st.button("Next Step ➡", on_click=next_step, type="primary", disabled=not all_answered)

# =========================================================
# PAGE 3: ASSET ALLOCATION (Clean Input Version)
# =========================================================
elif st.session_state["current_step"] == 2:
    st.header("📊 3. จัดพอร์ตการลงทุน (Asset Allocation)")

    curr_cash     = st.session_state.get("v_cash_dep", 0.0)
    curr_bond     = st.session_state.get("v_bond", 0.0)
    curr_stock    = st.session_state.get("v_stock", 0.0)
    curr_gl_stock = st.session_state.get("v_gl_stock", 0.0)
    curr_other    = st.session_state.get("v_other_invest", 0.0) 
    
    val_deposit = money_input("Fix Deposit (THB)", curr_cash, "p3_deposit")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Thai Assets")
        val_gov_bond = money_input("Government Bond 1Y (THB)", curr_bond, "p3_gov_bond")
        val_seti     = money_input("Thai Equity (THB)", curr_stock, "p3_seti")
        val_xauthb   = money_input("Gold (XAUTHB) (THB)", 0.0, "p3_xauthb")
        val_reitth   = money_input("Thai REITs (THB)", curr_other, "p3_reitth")

    with c2:
        st.subheader("Global Assets")
        val_msci_gov   = money_input("Global Govt Bond (THB)", 0.0, "p3_msci_gov")
        val_msci_stock = money_input("Global Stock (THB)", curr_gl_stock, "p3_msci_stock")
        val_xauusd     = money_input("Gold (XAUUSD) (THB)", 0.0, "p3_xauusd")
        val_mscireits  = money_input("Global REITs (THB)", 0.0, "p3_mscireits")

    total_port_value = (val_deposit + val_gov_bond + val_seti + val_xauthb + val_reitth +
                        val_msci_gov + val_msci_stock + val_xauusd + val_mscireits)

    if total_port_value > 0:
        st.markdown(f"### 💰 Total Portfolio: **{total_port_value:,.0f}** THB")
        alloc = {
            "pct_deposit": val_deposit / total_port_value,
            "pct_gov_bond": val_gov_bond / total_port_value,
            "pct_seti": val_seti / total_port_value,
            "pct_XAUTHB": val_xauthb / total_port_value,
            "pct_REITTH": val_reitth / total_port_value,
            "pct_msci_gov_bond": val_msci_gov / total_port_value,
            "pct_msci_stock": val_msci_stock / total_port_value,
            "pct_XAUUSD": val_xauusd / total_port_value,
            "pct_MSCIREITs": val_mscireits / total_port_value
        }
        
        st.session_state["saved_alloc"] = alloc
        st.session_state["final_total_wealth"] = total_port_value
            
    else:
        st.warning("⚠️ Please enter asset values to continue.")

    # --- NAV ---
    cn1, cn2 = st.columns([1, 8])
    cn1.button("⬅ Back", on_click=prev_step)
    cn2.button("Next Step ➡", type="primary", on_click=next_step)
# =========================================================
# PAGE 4: SIMULATION + EXPORT (wd_rate only, no cashflow mode)
# =========================================================
elif st.session_state["current_step"] == 3:
    st.header("💸 4.กลยุทธ์การถอนเงิน")

    YEARS = 30
    N_SIM = 50000

    asset_stats = {
        "pct_deposit": {"mean": 0.0206, "std": 0.0125},
        "pct_gov_bond": {"mean": 0.0505, "std": 0.0572},
        "pct_seti": {"mean": 0.1227, "std": 0.3266},
        "pct_XAUTHB": {"mean": 0.065, "std": 0.150},
        "pct_REITTH": {"mean": 0.070, "std": 0.200},

        "pct_msci_stock": {"mean": 0.0926, "std": 0.1852},
        "pct_msci_gov_bond": {"mean": 0.0926, "std": 0.1852},
        "pct_XAUUSD": {"mean": 0.1175, "std": 0.1752},
        "pct_MSCIREITs": {"mean": 0.0926, "std": 0.1853}
    }

    alloc = st.session_state.get("saved_alloc", {})
   # --- 2. DATA SOURCE ---
    st.markdown("### 📂 Data Assumptions")
    data_mode = st.radio("Choose Source:", ["Use Default Assumptions", "Upload Bloomberg Files"], horizontal=True)

    custom_mean = None
    custom_cov = None

    if data_mode == "Upload Bloomberg Files":
        st.info("💡 **Instructions:** Upload your Asset files AND a USD/THB Exchange Rate file.")
        
        uploaded_files = st.file_uploader(
            "Upload Excel/CSV files here:", 
            type=["csv", "xlsx", "xls"], 
            accept_multiple_files=True
        )
        
        # MAPPING: Includes the specific "USD/THB Exchange Rate" option
        sys_map = {
            "Select Option...": "ignore",
            "🔴 USD/THB Exchange Rate": "rate_usd_thb",  # <--- CRITICAL
            "-----------------------": "ignore",
            "Fix Deposit": "pct_deposit",
            "Thai Gov Bond": "pct_gov_bond",
            "Thai Equity (SET)": "pct_seti",
            "Gold (XAUTHB)": "pct_XAUTHB",
            "Thai REITs": "pct_REITTH",
            "Global Stocks (MSCI)": "pct_msci_stock",
            "Global Bond": "pct_msci_gov_bond",
            "Gold (USD)": "pct_XAUUSD",
            "Global REITs": "pct_MSCIREITs"
        }
        
        if uploaded_files:
            st.markdown("#### 🔗 Map Files & Select Currency")
            
            # Temporary storage
            file_configs = []     
            parsed_series = {}    
            
            # --- STEP 1: UI LOOP (Get User Inputs) ---
            for f in uploaded_files:
                # Layout: File Name | Asset Type | Currency
                c1, c2, c3 = st.columns([3, 3, 2])
                
                c1.write(f"📄 **{f.name}**")
                
                # 1. Asset Type Selector
                choice_label = c2.selectbox(
                    "Map to:", 
                    list(sys_map.keys()), 
                    key=f"map_{f.name}", 
                    label_visibility="collapsed"
                )
                asset_code = sys_map[choice_label]
                
                # 2. Currency Selector (Only if it's an asset, not the rate itself)
                currency = "THB"
                if asset_code != "ignore" and asset_code != "rate_usd_thb":
                    currency = c3.radio(
                        "Currency:", 
                        ["THB", "USD"], 
                        key=f"curr_{f.name}", 
                        horizontal=True,
                        label_visibility="collapsed"
                    )
                
                # Store config if valid
                if asset_code != "ignore":
                    file_configs.append({
                        "file": f, 
                        "code": asset_code, 
                        "curr": currency
                    })

            # --- STEP 2: PARSING & PROCESSING ---
            if file_configs:
                # A. Parse All Files
                usd_rate_series = None
                
                for cfg in file_configs:
                    s, err = parse_bloomberg_file(cfg["file"]) # Uses your "Ultimate Parser"
                    if not err:
                        parsed_series[cfg["code"]] = s
                        # If this is the FX rate, save it specifically
                        if cfg["code"] == "rate_usd_thb":
                            usd_rate_series = s
                    else:
                        st.error(f"❌ Error in {cfg['file'].name}: {err}")

                # B. Check if we need FX rate but don't have it
                needs_usd = any(c["curr"] == "USD" for c in file_configs)
                if needs_usd and usd_rate_series is None:
                    st.error("🚨 You selected files as **USD**, but you haven't mapped a **USD/THB Exchange Rate** file yet!")
                
                # C. Build Final Dataframe
                else:
                    merged_df = pd.DataFrame()
                    
                    for cfg in file_configs:
                        code = cfg["code"]
                        if code == "rate_usd_thb": continue # Skip adding the raw rate to the portfolio
                        
                        raw_data = parsed_series.get(code)
                        
                        if raw_data is not None:
                            if cfg["curr"] == "USD":
                                # --- APPLY YOUR FORMULA (via Price alignment) ---
                                # Formula: R_thb = (1+R_usd)*(1+R_fx) - 1
                                # Implementation: Price_THB = Price_USD * FX_Rate
                                
                                # 1. Align Dates (Inner Join)
                                aligned = pd.concat([raw_data, usd_rate_series], axis=1, join='inner').dropna()
                                aligned.columns = ["Asset_USD", "FX_Rate"]
                                
                                # 2. Convert to THB
                                thb_series = aligned["Asset_USD"] * aligned["FX_Rate"]
                                merged_df[code] = thb_series
                                
                                st.caption(f"✅ Converted **{cfg['file'].name}** to THB (Matched {len(thb_series)} days)")
                            else:
                                # Already THB
                                merged_df[code] = raw_data

                    # D. Verification & Saving
                    if not merged_df.empty:
                        try:
                            # Use Annualized Monthly Returns (Best Method)
                            df_monthly = merged_df.resample('ME').last().pct_change().dropna()
                            
                            if len(df_monthly) > 36:
                                st.success(f"✅ Data Ready! Found {len(df_monthly)} months of valid data.")
                                
                                # --- VERIFICATION TABLE ---
                                st.markdown("##### 🔍 Data Verification (All in THB)")
                                verify_df = pd.DataFrame({
                                    "Monthly Return": df_monthly.mean() * 100,
                                    "Annual Return": df_monthly.mean() * 12 * 100,
                                    "Volatility": df_monthly.std() * (12**0.5) * 100
                                })
                                st.dataframe(verify_df.style.format("{:.2f}%"))
                                # --------------------------

                                custom_mean = (df_monthly.mean() * 12).to_dict()
                                custom_cov = df_monthly.cov() * 12
                                
                                st.session_state["custom_mean"] = custom_mean
                                st.session_state["custom_cov"] = custom_cov
                            else:
                                st.warning("⚠️ Not enough overlapping data (need >3 years).")
                        except Exception as e:
                            st.error(f"Calculation Error: {e}")

    # Use saved custom data
    if data_mode == "Upload Bloomberg Files" and "custom_mean" in st.session_state:
        custom_mean = st.session_state["custom_mean"]
        custom_cov = st.session_state["custom_cov"]

    c1, c2 = st.columns(2)
    strat_options = ["Basic Strategy", "Forgoing Inflation", "RMD Strategy", "Guardrails"]
    with c1:
        strat_selection = st.selectbox("กลยุทธ์", strat_options)
        st.session_state["sim_strat"] = strat_selection
    with c2:
        wd_rate = st.number_input("อัตราการถอน (%)", 3.0, 10.0, 4.0, 0.1) / 100

    start_port = st.session_state.get("start_port", 1_000_000.0)
    inflation = st.session_state.get("inflation", 0.03)
    retire_age = st.session_state.get("retire_age", 60)

    # =========================
    # 1) RUN SIMULATION (generate returns once, reuse later)
    # =========================
    if st.button("🚀 Run Simulation", type="primary"):
        sim = RetirementSimulator()
        with st.spinner("Simulating..."):
            mc_returns = sim.simulate_returns(alloc, asset_stats, N_SIM, YEARS)
            st.session_state["mc_returns"] = mc_returns  

            res = sim.run_simulation(
                initial_portfolio=start_port,
                portfolio_allocation=alloc,
                asset_stats=asset_stats,
                withdrawal_strategy=strat_selection,
                withdrawal_rate=wd_rate,
                n_simulations=N_SIM,
                years=YEARS,
                inflation_rate=inflation,
                starting_age=retire_age)

        st.session_state["res"] = res
        st.session_state["sim_strat"] = strat_selection
        st.session_state["wd_rate"] = wd_rate

        # clear export cache
        st.session_state.pop("export_pdf_bytes", None)
        st.session_state.pop("export_csv_bytes", None)

    # =========================
    # 2) RESULTS
    # =========================
    if "res" in st.session_state:
        res = st.session_state["res"]
        inh_goal = st.session_state.get("inheritance_goal", 0.0) # Retrieve goal
        
        success = res["survival_rate"] * 100
        inh_success = res.get("inheritance_success_rate", 0.0) * 100 # Retrieve new rate
        median_end = res["median_balance"][-1]

        st.divider()
        
        # Updated Metrics Layout
        m1, m2, m3 = st.columns(3)
        
        color_surv = "green" if success > 85 else "red"
        m1.markdown(f"### Survival Rate: :{color_surv}[{success:.1f}%]")
        m1.caption("Chance money lasts > 30 years")
        
        m2.metric("Median End Balance", f"{median_end:,.0f} THB")
        
        # New Metric for Inheritance
        color_inh = "green" if inh_success > 50 else "orange" # You can adjust this threshold
        m3.markdown(f"### Inheritance Success: :{color_inh}[{inh_success:.1f}%]")
        m3.caption(f"Chance to leave ≥ {inh_goal:,.0f}")

        # ✅ Graph Update
        fig, ax = plt.subplots(figsize=(10, 5))
        x = range(len(res["median_balance"]))
        
        # Plot areas
        ax.fill_between(x, res["percentile_10"], res["percentile_90"], alpha=0.2, label="10-90th Pctl")
        ax.plot(x, res["median_balance"], label="Median Balance")
        
        # 1. Zero Line (Depleted)
        ax.axhline(0, color='red', linestyle="--", linewidth=1, label="Depleted (0)")
        
        # 2. Inheritance Line (Only if goal > 0)
        if inh_goal > 0:
            ax.axhline(inh_goal, color='purple', linestyle="-.", linewidth=1.5, label=f"Inheritance Goal ({inh_goal:,.0f})")

        ax.legend(loc='upper left')
        ax.set_xlabel("Year")
        ax.set_ylabel("Portfolio Value (THB)")
        ax.set_title("Wealth Projection")
        
        # Format Y-axis to standard comma notation
        ax.get_yaxis().set_major_formatter(
            plt.FuncFormatter(lambda x, p: format(int(x), ',')))

        st.pyplot(fig)

        # =========================
        # 5) Recommendations
        # =========================
        sim = RetirementSimulator()
        st.subheader("💡 Recommendations")
        if success < 85:
            st.error(f"⚠️ Survival Rate ({success:.1f}%) is below 85% target.")
            recs = sim.recommend_improvements(
                current_survival_rate=res["survival_rate"],
                portfolio_allocation=alloc,
                withdrawal_rate=st.session_state.get("wd_rate", wd_rate),
            )
            with st.expander("👉 View Action Plan", expanded=True):
                for r in recs:
                    st.info(r)
        else:
            st.success("✅ Your plan looks solid! You have a high chance of success.")

        # =========================
        # 6) Optimizer
        # =========================
        st.divider()
        if st.button("🔍 Find Optimal Withdrawal Rate"):
            with st.spinner("Optimizing..."):
                opt_rate = sim.find_optimal_withdrawal_rate(
                    initial_portfolio=start_port,
                    portfolio_allocation=alloc,
                    asset_stats=asset_stats,
                    withdrawal_strategy=st.session_state.get("sim_strat", "Basic Strategy"),
                    initial_rate=st.session_state.get("wd_rate", wd_rate),
                    years=YEARS,
                    inflation_rate=inflation,
                    starting_age=retire_age,
                )

            curr = st.session_state.get("wd_rate", wd_rate)
            diff = opt_rate - curr

            c_opt1, c_opt2 = st.columns(2)
            c_opt1.metric("Current Rate", f"{curr*100:.2f}%")
            c_opt2.metric("Optimal Rate", f"{opt_rate*100:.2f}%", f"{diff*100:.2f}%")

            if diff > 0:
                st.success(f"🎉 You can safely increase your withdrawal by {diff*100:.2f}%!")
            else:
                st.warning(f"⚠️ You should reduce your withdrawal by {abs(diff*100):.2f}% to be safe.")

    # =========================================================
    # EXPORT (Page 4 only) 
    # =========================================================
    st.divider()
    st.subheader("💾 Save Your Plan")

    if st.button("✅ Prepare Export Files"):
        res = st.session_state.get("res")
        alloc = st.session_state.get("saved_alloc", {})
        name_final = st.session_state.get("user_name", "ไม่ระบุชื่อ")
        ret_age_final = st.session_state.get("retire_age",65)
        life_exp_final = st.session_state.get("life_expectancy", 85)
        
        ret_age_final = int(st.session_state.get("retire_age", 60))
        life_exp_final = int(st.session_state.get("life_expectancy", 85))

        # 1. Basic Profile Data
        export_data = {
            "name": name_final,
            "retire_age": ret_age_final,
            "life_exp": life_exp_final,
            "inflation": st.session_state.get("inflation", 0.03),
            "inheritance_goal": st.session_state.get("inheritance_goal", 0.0),
            "sim_strat": strat_selection
        }

        # 2. Financial Calculations
        # Pull the totals we saved at the bottom of Page 1
        total_income = st.session_state.get("v_total_income", 0.0)
        total_expense = st.session_state.get("v_total_expense", 0.0)
        investable = st.session_state.get("start_port", 0.0)
        total_debt = st.session_state.get("money_debt", 0.0)
        net_saving = total_income - total_expense

        # 3. Update dictionary with ALL required keys
        export_data.update({
            "name": st.session_state.get("user_name", "ไม่ระบุชื่อ"),
            "retire_age": st.session_state.get("retire_age", 60),
            "life_exp": st.session_state.get("life_expectancy", 85),
            "total_income": total_income,
            "total_expense": total_expense,
            "yearly_savings": net_saving,
            "investable": investable,
            "total_debt": total_debt,
            "net_worth": investable - total_debt,         
            # Detailed Breakdown for CSV
            "inc_detail": {
                "Pension": get_num("inc_sal"),
                "Rental": get_num("inc_rent"),
                "Dividend": get_num("inc_div"),
                "Other": get_num("inc_other")
            },
            "exp_fixed_detail": {
                "Loan": get_num("exp_loan"),
                "Housing": get_num("exp_house"),
                "Insurance": get_num("exp_ins"),
                "Subscription": get_num("exp_sub"),
                "Other Fixed": get_num("exp_fix_oth")
            },
            "exp_var_detail": {
                "Transport": get_num("exp_trans"),
                "Food": get_num("exp_food"),
                "Entertain": get_num("exp_ent"),
                "Travel": get_num("exp_travel"),
                "Health": get_num("exp_health"),
                "Other Variable": get_num("exp_var_oth")
            },
            "asset_detail": {
                "Cash": get_num("cash_dep"),
                "Bond": get_num("bond"),
                "Thai Equity": get_num("stock"),
                "Global Equity": get_num("gl_stock"),
                "Other Invest": get_num("other_invest")
            },
            "debt_detail": {
                "Home Loan": get_num("debt_home"),
                "Car Loan": get_num("debt_car"),
                "Credit Card": get_num("debt_cc"),
                "Other Debt": get_num("debt_other")
            }
        })

        # 4. Generate the Files
        st.session_state["export_data"] = export_data
        st.session_state["export_csv_bytes"] = build_full_report_csv(export_data, res, alloc)
        st.session_state["export_pdf_bytes"] = build_pdf_bytes(export_data, res)

        st.success("เตรียมไฟล์สำเร็จ! Prepared Export Files ✅")
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "📄 Download Full Report CSV",
            data=st.session_state.get("export_csv_bytes", b""),
            file_name="full_retirement_report.csv",
            mime="text/csv",
            disabled=("export_csv_bytes" not in st.session_state),
        )
    with c2:
        st.download_button(
            "📕 Download PDF",
            data=st.session_state.get("export_pdf_bytes", b""),
            file_name="report.pdf",
            mime="application/pdf",
            disabled=("export_pdf_bytes" not in st.session_state),
        )
        
    st.button("⬅ Back", on_click=prev_step)
