
import streamlit as st

def contract_badge() -> None:
    """Compact summary of the current option contract in the sidebar."""
    opt = st.session_state.get("option_obj")
    if opt is None:
        return

    with st.sidebar.expander("📝 Current Contract", expanded=False):

        c1, c2 = st.columns([2, 3], gap="small")

        with c1:
            st.markdown("**Ticker**")
            st.markdown("**Type**")
            st.markdown("**Maturity**")
            st.markdown("**r**")
            st.markdown(r"**$\sigma$**")
            st.markdown("**Spot**")
            st.markdown("**Strike K**")

        with c2:
            st.markdown(opt.underlying_ticker)
            st.markdown(opt.option_type.capitalize())
            st.markdown(f"{opt.maturity} days")
            st.markdown(f"{opt.risk_free_rate:.2%}")
            st.markdown(f"{opt.volatility:.2%}")
            st.markdown(f"{opt.spot_price:,.2f}")
            st.markdown(f"{opt.strike_price:,.2f}")

        st.sidebar.page_link("0_Home.py", label="✏️ edit contract")


def pricing_badge() -> None:
    """Compact summary of the latest pricing run in the sidebar."""
    res = st.session_state.get("pricing_result")
    if res is None:
        return

    with st.sidebar.expander("💲 Latest Pricing", expanded=False):

        c1, c2 = st.columns([3, 2], gap="small")

        # left column – labels
        with c1:
            st.markdown("**Black-Scholes**")
            st.markdown(f"**{res['method']}**")
            st.markdown("**Jump Diffusion**")
            st.markdown("**Paths used**")

        # right column – values
        with c2:
            st.markdown(f"{res['bs_price']:,.4f}")
            st.markdown(f"{res['mc_price']:,.4f}")
            st.markdown(f"{res['jump_diff_price']:,.4f}")
            st.markdown(f"{res['n_paths']:,}")
