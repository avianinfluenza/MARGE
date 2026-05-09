from ._agent_factory import ingest_csv_and_build_agent

# Train and load into memory instantly
new_clinical_agent = ingest_csv_and_build_agent(
    csv_path="new_patients.csv",
    target_column="has_disease",
    agent_name="live_disease_tracker",
    artifact_filename="temp_tracker.joblib"
)

# Start predicting on the exact same line of code!
InputSchema = new_clinical_agent.input_schema
mock_data = InputSchema(feature1=1.2, feature2=3.4)
prediction = new_clinical_agent.predict(mock_data)