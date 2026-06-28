from mlops_project.pipeline_registry import register_pipelines


def test_registered_pipelines_are_green_taxi_only():
    pipelines = register_pipelines()

    expected = {
        "__default__",
        "data_cleaning",
        "data_prep",
        "data_unit_tests",
        "feature_engineering",
        "feature_selection",
        "ingestion",
        "model_train",
        "split_data",
        "training",
    }

    assert expected.issubset(pipelines)
    assert not any(name.startswith("example_") for name in pipelines)


def test_default_pipeline_contains_active_green_taxi_nodes():
    pipeline = register_pipelines()["__default__"]
    node_names = {node.name for node in pipeline.nodes}

    assert {
        "ingestion_node",
        "clean_modeling_data_node",
        "engineer_features_node",
        "select_features_node",
        "split_data_node",
        "model_train_node",
    }.issubset(node_names)
