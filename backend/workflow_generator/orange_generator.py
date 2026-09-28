"""
DataPilot – Orange3 Workflow Generator
Produces a valid Orange3 .ows (XML) file from the trained pipeline.

The generated workflow contains:
  File → DataTable, Preprocess, Distributions, ScatterPlot
  Preprocess → Test&Score (+ up-to-3 model learner widgets)
  Test&Score → ConfusionMatrix / ROC (classification) OR Predictions (regression)
"""
import uuid
import logging
from pathlib import Path
from typing import List
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom

from backend.schemas.models import ProblemType, ModelMetrics
from backend.utils.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Widget catalogue
# ---------------------------------------------------------------------------
_W = {
    "file": {
        "name": "File",
        "qualified_name": "Orange.widgets.data.owfile.OWFile",
        "category": "Data",
    },
    "data_table": {
        "name": "Data Table",
        "qualified_name": "Orange.widgets.data.owtable.OWDataTable",
        "category": "Data",
    },
    "preprocess": {
        "name": "Preprocess",
        "qualified_name": "Orange.widgets.data.owpreprocess.OWPreprocess",
        "category": "Data",
    },
    "distributions": {
        "name": "Distributions",
        "qualified_name": "Orange.widgets.visualize.owdistributions.OWDistributions",
        "category": "Visualize",
    },
    "scatter_plot": {
        "name": "Scatter Plot",
        "qualified_name": "Orange.widgets.visualize.owscatterplot.OWScatterPlot",
        "category": "Visualize",
    },
    "logistic_regression": {
        "name": "Logistic Regression",
        "qualified_name": "Orange.widgets.model.owlogisticregression.OWLogisticRegression",
        "category": "Model",
    },
    "random_forest": {
        "name": "Random Forest",
        "qualified_name": "Orange.widgets.model.owrandomforest.OWRandomForest",
        "category": "Model",
    },
    "gradient_boosting": {
        "name": "Gradient Boosting",
        "qualified_name": "Orange.widgets.model.owgradientboosting.OWGradientBoosting",
        "category": "Model",
    },
    "svm": {
        "name": "SVM",
        "qualified_name": "Orange.widgets.model.owsvm.OWSVM",
        "category": "Model",
    },
    "knn": {
        "name": "kNN",
        "qualified_name": "Orange.widgets.model.owknn.OWKNNLearner",
        "category": "Model",
    },
    "linear_regression": {
        "name": "Linear Regression",
        "qualified_name": "Orange.widgets.model.owlinearregression.OWLinearRegression",
        "category": "Model",
    },
    "test_and_score": {
        "name": "Test and Score",
        "qualified_name": "Orange.widgets.evaluate.owtestlearners.OWTestLearners",
        "category": "Evaluate",
    },
    "confusion_matrix": {
        "name": "Confusion Matrix",
        "qualified_name": "Orange.widgets.evaluate.owconfusionmatrix.OWConfusionMatrix",
        "category": "Evaluate",
    },
    "roc_analysis": {
        "name": "ROC Analysis",
        "qualified_name": "Orange.widgets.evaluate.owrocanalysis.OWROCAnalysis",
        "category": "Evaluate",
    },
    "predictions": {
        "name": "Predictions",
        "qualified_name": "Orange.widgets.evaluate.owpredictions.OWPredictions",
        "category": "Evaluate",
    },
}

_MODEL_KEY = {
    "Logistic Regression":        "logistic_regression",
    "Random Forest":               "random_forest",
    "Gradient Boosting":           "gradient_boosting",
    "SVM":                         "svm",
    "KNN":                         "knn",
    "Linear Regression":           "linear_regression",
    "Ridge":                       "linear_regression",
    "Random Forest Regressor":     "random_forest",
    "Gradient Boosting Regressor": "gradient_boosting",
    "SVR":                         "svm",
}


class OrangeWorkflowGenerator:

    def __init__(self, dataset_id: str):
        self.dataset_id  = dataset_id
        self.workflow_dir = Path(settings.WORKFLOW_DIR)
        self.workflow_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(
        self,
        dataset_path:        str,
        best_model_name:     str,
        all_models:          List[ModelMetrics],
        problem_type:        ProblemType,
        preprocessing_steps: List[str],
        target_column:       str,
    ) -> str:
        """Build the .ows XML, save it, and return the file path."""
        logger.info(f"Generating Orange3 workflow for {self.dataset_id}")

        nodes: list  = []
        links: list  = []
        nid = 0   # auto-incrementing node id

        def add_node(key: str, x: int, y: int) -> int:
            nonlocal nid
            w = _W[key]
            nodes.append({
                "id":             str(nid),
                "name":           w["name"],
                "qualified_name": w["qualified_name"],
                "project_name":   "Orange3",
                "version":        "",
                "title":          w["name"],
                "position":       f"({x}, {y})",
                "category":       w["category"],
            })
            cur = nid
            nid += 1
            return cur

        def add_link(src: int, src_ch: str, dst: int, dst_ch: str):
            links.append({
                "id":             str(uuid.uuid4()),
                "source_node_id": str(src),
                "sink_node_id":   str(dst),
                "source_channel": src_ch,
                "sink_channel":   dst_ch,
                "enabled":        "true",
            })

        # ── Layout ────────────────────────────────────────────────────────
        # Row 0 (y=60):   model learner widgets
        # Row 1 (y=220):  File  →  Preprocess  →  Test&Score  →  Eval widgets
        # Row 2 (y=400):  DataTable  Distributions  ScatterPlot

        file_id        = add_node("file",        80,  220)
        table_id       = add_node("data_table",  80,  400)
        preprocess_id  = add_node("preprocess",  300, 220)
        dist_id        = add_node("distributions",300, 400)
        scatter_id     = add_node("scatter_plot", 520, 400)

        add_link(file_id,       "Data",              table_id,      "Data")
        add_link(file_id,       "Data",              preprocess_id, "Data")
        add_link(file_id,       "Data",              dist_id,       "Data")
        add_link(file_id,       "Data",              scatter_id,    "Data")

        # Top 3 model learner widgets
        top_models   = all_models[:3]
        learner_ids  = []
        for i, m in enumerate(top_models):
            key = _MODEL_KEY.get(m.model_name, "random_forest")
            lid = add_node(key, 300 + i * 200, 60)
            learner_ids.append(lid)

        test_id = add_node("test_and_score", 700, 220)
        add_link(preprocess_id, "Preprocessed Data", test_id, "Data")
        for lid in learner_ids:
            add_link(lid, "Learner", test_id, "Learner")

        if problem_type == ProblemType.CLASSIFICATION:
            cm_id  = add_node("confusion_matrix", 920, 120)
            roc_id = add_node("roc_analysis",     920, 300)
            add_link(test_id, "Evaluation Results", cm_id,  "Evaluation Results")
            add_link(test_id, "Evaluation Results", roc_id, "Evaluation Results")
        else:
            pred_id = add_node("predictions", 920, 220)
            add_link(preprocess_id, "Preprocessed Data", pred_id, "Data")
            if learner_ids:
                add_link(learner_ids[0], "Learner", pred_id, "Predictors")

        xml_str = self._build_xml(nodes, links, best_model_name, problem_type)

        filename = f"datapilot_{self.dataset_id[:8]}.ows"
        filepath = self.workflow_dir / filename
        filepath.write_text(xml_str, encoding="utf-8")
        logger.info(f"Workflow saved → {filepath}")
        return str(filepath)

    # ── XML helpers ───────────────────────────────────────────────────────────

    def _build_xml(
        self,
        nodes:         list,
        links:         list,
        best_model:    str,
        problem_type:  ProblemType,
    ) -> str:
        scheme = Element("scheme")
        scheme.set("version",     "2.0")
        scheme.set("title",       f"DataPilot AutoML – {best_model}")
        scheme.set("description", (
            f"Auto-generated by DataPilot. "
            f"Best model: {best_model}. Task: {problem_type.value}."
        ))

        # Nodes
        nodes_el = SubElement(scheme, "nodes")
        for n in nodes:
            el = SubElement(nodes_el, "node")
            for k, v in n.items():
                el.set(k, v)

        # Links
        links_el = SubElement(scheme, "links")
        for lnk in links:
            el = SubElement(links_el, "link")
            for k, v in lnk.items():
                el.set(k, v)

        # Annotation
        ann_el  = SubElement(scheme, "annotations")
        txt_el  = SubElement(ann_el, "text")
        txt_el.set("id",   "0")
        txt_el.set("rect", "(20, 20, 480, 60)")
        txt_el.set("type", "text/plain")
        txt_el.set("font", "SourceSansPro,12,-1,5,50,0,0,0,0,0")
        txt_el.text = (
            f"DataPilot AutoML Pipeline\n"
            f"Best model: {best_model}  |  Task: {problem_type.value.title()}\n"
            f"Generated automatically – open in Orange3 to explore."
        )

        SubElement(scheme, "window_groups")

        raw    = tostring(scheme, encoding="unicode")
        parsed = minidom.parseString(raw)
        return parsed.toprettyxml(indent="    ", encoding=None)
