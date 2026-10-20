from .compare_diagnostics_template import CompareDiagnosticsTemplate
from .diagnostics_template import DiagnosticsTemplate
from .feature_extraction_comparison_template import FeatureExtractionComparisonTemplate
from ....base.reporting.templates.feature_extraction_template import FeatureExtractionTemplate
from .feature_removal_combinations_template import FeatureRemovalCombinationsTemplate
from .feature_selection_template import FeatureSelectionTemplate
from .feature_sets_comparison_template import FeatureSetsComparisonTemplate
from .learning_curve_template import LearningCurveTemplate
from .model_comparison_template import ModelComparisonTemplate
from .permutation_importance_template import PermutationImportanceTemplate
from .refine_and_compare_template import RefineAndCompareTemplate
from .regularization_comparison_template import RegularizationComparisonTemplate
from .single_feature_removal_template import SingleFeatureRemovalTemplate
from .tune_model_template import TuneModelTemplate

__all__ = [
    "CompareDiagnosticsTemplate",
    "DiagnosticsTemplate",
    "FeatureExtractionComparisonTemplate",
    "FeatureExtractionTemplate",
    "FeatureRemovalCombinationsTemplate",
    "FeatureSelectionTemplate",
    "FeatureSetsComparisonTemplate",
    "LearningCurveTemplate",
    "ModelComparisonTemplate",
    "PermutationImportanceTemplate",
    "RefineAndCompareTemplate",
    "RegularizationComparisonTemplate",
    "SingleFeatureRemovalTemplate",
    "TuneModelTemplate",
]
