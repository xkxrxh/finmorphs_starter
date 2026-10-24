
import streamlit as st

st.set_page_config(page_title="FinMorphs Diagnostic")

st.title("👓 FinMorphs")
st.success("The page is rendering correctly.")

st.write("If you can read this, Streamlit's basic page rendering works.")

if st.button("Test interaction"):
    st.balloons()
    st.write("Button interaction works!")

