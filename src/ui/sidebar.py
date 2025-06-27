
import streamlit as st

def contract_badge() -> None:
    """Compact summary of the current option contract in the sidebar."""
    opt = st.session_state.get("option_obj")
    if opt is None:
        return

    with st.sidebar.expander("📝 Current Contract", expanded=True):

        c1, c2 = st.columns([2, 3], gap="small")

        with c1:
            st.markdown("**Ticker**")
            st.markdown("**Type**")
            st.markdown("**Maturity**")
            st.markdown("**Risk free rate r%**")
            st.markdown("**Volatility**")
            st.markdown("**Spot $**")
            st.markdown("**Strike K $**")

        with c2:
            st.markdown(opt.underlying_ticker)
            st.markdown(opt.option_type.capitalize())
            st.markdown(f"{opt.maturity} days")
            st.markdown(f"{opt.risk_free_rate:.2%}")
            st.markdown(f"{opt.volatility:.2%}")
            st.markdown(f"{opt.spot_price:,.2f}")
            st.markdown(f"{opt.strike_price:,.2f}")

        st.sidebar.page_link("0_Home.py", label="✏️ edit contract")
