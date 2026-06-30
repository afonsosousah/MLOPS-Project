from mlops_project.pipeline_registry import register_pipelines


def test_registered_pipelines_are_green_taxi_only():
    pipelines = register_pipelines()

    expected = {
        "__default__",
        "data_drift",
        "data_quality",
        "data_cleaning",
        "data_unit_tests",
        "drift_monitoring",
        "feature_selection",
        "feature_store",
        "ingestion",
        "model_predict",
        "model_selection",
        "model_train",
        "preprocessing_batch",
        "preprocessing_train",
        "production_full_prediction_process",
        "production_full_train_process",
        "split_data",
        "split_train",
    }

    assert expected.issubset(pipelines)
    assert "feature_engineering" not in pipelines
    assert not any(name.startswith("example_") for name in pipelines)


def test_default_pipeline_contains_bank_example_style_nodes():
    pipeline = register_pipelines()["__default__"]
    node_names = {node.name for node in pipeline.nodes}

    assert {
        "ingestion_node",
        "modeling_data_unit_tests_node",
        "split_data_node",
        "split_train_node",
        "preprocessing_train_node",
        "select_features_node",
        "model_selection_node",
        "model_train_node",
        "preprocessing_batch_node",
        "model_predict_node",
        "detect_drift_node",
        "upload_features_to_store_node",
    }.issubset(node_names)


def test_data_quality_pipeline_includes_ingestion_inputs():
    pipeline = register_pipelines()["data_quality"]
    node_names = {node.name for node in pipeline.nodes}

    assert {
        "ingestion_node",
        "modeling_data_unit_tests_node",
        "drift_data_unit_tests_node",
    }.issubset(node_names)


def test_feature_store_pipeline_is_registered_and_present_in_default():
    pipelines = register_pipelines()

    assert {node.name for node in pipelines["feature_store"].nodes} == {
        "upload_features_to_store_node"
    }
    assert "upload_features_to_store_node" in {
        node.name for node in pipelines["__default__"].nodes
    }
