import shap
import torch
import numpy as np

class SHAPExplainer:
    def __init__(self, model, background_data):
        self.model = model
        self.background_data = background_data

    def explain(self, data_sample):
        self.model.eval()
        explainer = shap.DeepExplainer(self.model, self.background_data)
        shap_values = explainer.shap_values(data_sample)
        return shap_values

    def summary_plot(self, shap_values, data_sample, feature_names=None):
        shap.summary_plot(shap_values, data_sample, feature_names=feature_names)

    def waterfall_plot(self, shap_values, data_sample, index=0, feature_names=None):
        explanation = shap.Explanation(
            values=shap_values[index],
            data=data_sample[index],
            feature_names=feature_names
        )
        shap.plots.waterfall(explanation)

