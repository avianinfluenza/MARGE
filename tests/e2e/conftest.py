"""E2E test configuration.

No stubs needed: BeeAI and Streamlit are installed in the venv, and
apps/streamlit_ui/app.py guards module-level Streamlit execution behind
a runtime context check, so importing app.py in tests is safe.
"""
