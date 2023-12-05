import streamlit as st

st.set_page_config(page_title="DataFrame Demo", page_icon="📊")

st.markdown("# DataFrame Demo")
st.sidebar.header("DataFrame Demo")
st.write(
    """This demo shows how to use `st.write` to visualize Pandas DataFrames.
(Data courtesy of the [UN Data Explorer](http://data.un.org/Explorer.aspx).)"""
)

st.subheader('Check your own values!')
st.write('We trained our Machine Learning model so that you can check the biodiversity of your soil based on your own chemical measurements. Just insert your values below and we\'ll see what your soil is made of and how you could possibly improve it\'s conditions.')

col1, col2, col3, col4, col5, col6, col7 = st.columns([1, 1, 1, 1, 1, 1, 1])

columns = [(col1, 'pH_CaCl2'), (col2, 'EC'), (col3, 'OC'), (col4, 'CaCO3'), (col5, 'P'), (col6, 'N'), (col7, 'K')]

def create_input_field(column, label):
    with column:
        st.number_input(label, min_value=min(df[label]), max_value=max(df[label]), value=df[label].mean())

for col in columns:
    create_input_field(col[0], col[1])

if st.button('Check Your Soil Data!'):
    st.write('test')