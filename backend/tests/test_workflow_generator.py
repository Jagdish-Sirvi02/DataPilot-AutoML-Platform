"""
Tests for OrangeWorkflowGenerator
"""
import os
import pytest
import xml.etree.ElementTree as ET

from workflow_generator.orange_generator import OrangeWorkflowGenerator
from schemas.models import ProblemType, ModelMetrics


@pytest.fixture
def sample_metrics():
    return [
        ModelMetrics(
            model_name="Random Forest",
            accuracy=0.92, precision=0.91, recall=0.92, f1_score=0.91,
            roc_auc=0.97, cv_score_mean=0.90, cv_score_std=0.02,
            training_time_sec=1.5,
        ),
        ModelMetrics(
            model_name="Logistic Regression",
            accuracy=0.85, precision=0.84, recall=0.85, f1_score=0.84,
            roc_auc=0.91, cv_score_mean=0.83, cv_score_std=0.03,
            training_time_sec=0.3,
        ),
        ModelMetrics(
            model_name="Gradient Boosting",
            accuracy=0.89, precision=0.88, recall=0.89, f1_score=0.88,
            roc_auc=0.95, cv_score_mean=0.87, cv_score_std=0.02,
            training_time_sec=3.1,
        ),
    ]


@pytest.fixture
def generator(tmp_path, monkeypatch):
    gen = OrangeWorkflowGenerator("test-dataset-id")
    monkeypatch.setattr(gen, "workflow_dir", tmp_path)
    gen.workflow_dir = tmp_path
    return gen


class TestWorkflowFileCreation:
    def test_creates_ows_file(self, generator, sample_metrics, tmp_path):
        path = generator.generate(
            dataset_path="/tmp/test_data.csv",
            best_model_name="Random Forest",
            all_models=sample_metrics,
            problem_type=ProblemType.CLASSIFICATION,
            preprocessing_steps=["Imputed missing values", "StandardScaler applied"],
            target_column="target",
        )
        assert os.path.exists(path)
        assert path.endswith(".ows")

    def test_file_not_empty(self, generator, sample_metrics):
        path = generator.generate(
            dataset_path="/tmp/test_data.csv",
            best_model_name="Random Forest",
            all_models=sample_metrics,
            problem_type=ProblemType.CLASSIFICATION,
            preprocessing_steps=[],
            target_column="target",
        )
        assert os.path.getsize(path) > 500

    def test_regression_workflow_created(self, generator, tmp_path):
        reg_metrics = [
            ModelMetrics(
                model_name="Random Forest",
                rmse=2.1, mae=1.4, r2=0.88,
                cv_score_mean=0.85, cv_score_std=0.03,
                training_time_sec=1.2,
            ),
        ]
        path = generator.generate(
            dataset_path="/tmp/test_data.csv",
            best_model_name="Random Forest",
            all_models=reg_metrics,
            problem_type=ProblemType.REGRESSION,
            preprocessing_steps=["StandardScaler"],
            target_column="price",
        )
        assert os.path.exists(path)


class TestWorkflowXMLStructure:
    def test_valid_xml(self, generator, sample_metrics):
        path = generator.generate(
            dataset_path="/tmp/test.csv",
            best_model_name="Random Forest",
            all_models=sample_metrics,
            problem_type=ProblemType.CLASSIFICATION,
            preprocessing_steps=[],
            target_column="label",
        )
        tree = ET.parse(path)
        root = tree.getroot()
        assert root.tag == "scheme"

    def test_has_nodes_element(self, generator, sample_metrics):
        path = generator.generate(
            dataset_path="/tmp/test.csv",
            best_model_name="Random Forest",
            all_models=sample_metrics,
            problem_type=ProblemType.CLASSIFICATION,
            preprocessing_steps=[],
            target_column="label",
        )
        tree = ET.parse(path)
        root = tree.getroot()
        nodes_el = root.find("nodes")
        assert nodes_el is not None

    def test_has_links_element(self, generator, sample_metrics):
        path = generator.generate(
            dataset_path="/tmp/test.csv",
            best_model_name="Random Forest",
            all_models=sample_metrics,
            problem_type=ProblemType.CLASSIFICATION,
            preprocessing_steps=[],
            target_column="label",
        )
        tree = ET.parse(path)
        root = tree.getroot()
        links_el = root.find("links")
        assert links_el is not None

    def test_scheme_has_title(self, generator, sample_metrics):
        path = generator.generate(
            dataset_path="/tmp/test.csv",
            best_model_name="Random Forest",
            all_models=sample_metrics,
            problem_type=ProblemType.CLASSIFICATION,
            preprocessing_steps=[],
            target_column="label",
        )
        tree = ET.parse(path)
        root = tree.getroot()
        assert "Random Forest" in root.get("title", "")

    def test_node_count_reasonable(self, generator, sample_metrics):
        path = generator.generate(
            dataset_path="/tmp/test.csv",
            best_model_name="Random Forest",
            all_models=sample_metrics,
            problem_type=ProblemType.CLASSIFICATION,
            preprocessing_steps=[],
            target_column="label",
        )
        tree = ET.parse(path)
        root = tree.getroot()
        nodes = root.find("nodes").findall("node")
        # Expect at least: File, DataTable, Preprocess, Distributions, Scatter,
        # 3 Models, Test&Score, ConfusionMatrix, ROC, Predictions = 12+
        assert len(nodes) >= 8

    def test_links_connect_valid_nodes(self, generator, sample_metrics):
        path = generator.generate(
            dataset_path="/tmp/test.csv",
            best_model_name="Random Forest",
            all_models=sample_metrics,
            problem_type=ProblemType.CLASSIFICATION,
            preprocessing_steps=[],
            target_column="label",
        )
        tree = ET.parse(path)
        root = tree.getroot()
        nodes = {n.get("id") for n in root.find("nodes").findall("node")}
        links = root.find("links").findall("link")
        for link in links:
            assert link.get("source_node_id") in nodes
            assert link.get("sink_node_id")   in nodes
