## Installing the Library
`pip install streamlit`
## Use streamlit run
Once you've created your script, say your_script.py, the easiest way to run it is with streamlit run:
`streamlit run your_script.py`
### Note
We can't use same widget again on same page because the Internal Structure of Streamlit
assigned an internal key.Multiple widgets with an identical structure will result in the same internal key, which causes the error. To fix the error, we need to pass a unique key argument to widget.

## Streamlit Cheat Sheet
`https://cheat-sheet.streamlit.app/`