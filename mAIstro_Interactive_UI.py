import gradio as gr
import logging
from smolagents import LiteLLMModel, CodeAgent
from mAIstro_tools import (
    PyRadiomicsFeatureExtractionTool,
    EDAToolException,
    ExploratoryDataAnalysisTool,
    FeatureImportanceAnalysisTool,
    NNUNetTrainingTool,
    NNUNetInferenceTool,
    TotalSegmentatorTool,
    PyCaretClassificationTool,
    PyCaretInferenceTool,
    PyCaretRegressionInferenceTool,
    PyCaretRegressionTool,
    PyTorchResNetTrainingTool,
    PyTorchResNetInferenceTool,
    PyTorchVGG16InferenceTool,
    PyTorchVGG16TrainingTool,
    PyTorchInceptionV3InferenceTool,
    PyTorchInceptionV3TrainingTool
)
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# === Define LLM choices ===
llm_options = {
    "OpenAI GPT-4.1": "openai/gpt-4.1",
    "OpenAI GPT-4o": "openai/gpt-4o",
    "Anthropic Claude Sonnet": "anthropic/claude-3-7-sonnet-20250219",
    "DeepSeek V3": "deepseek/deepseek-chat",
    "DeepSeek Reasoner": "deepseek/deepseek-reasoner"
}

# === Define example prompt variables ===
rfe_ct_prompt_1 = """
Perform a comprehensive radiomic feature extraction for the CT scans in: "tests/test_radiomic_extractor_agent/ct/images".
The corresponding masks are here: "tests/test_radiomic_extractor_agent/ct/labels".
Save the results here: "tests/test_radiomic_extractor_agent/ct/results_generic_prompt".
"""

rfe_ct_prompt_2 = """
Extract shape and first order radiomic features for the CT scans in: "tests/test_radiomic_extractor_agent/ct/images".
The respective masks are here: "tests/test_radiomic_extractor_agent/ct/labels".
Save the results here: "tests/test_radiomic_extractor_agent/ct/results_specific_features_and_filters".
Use the following filters: Exponential, Gradient, LBP2D.
"""

rfe_mri_prompt_1 = """
Perform a comprehensive radiomic feature extraction for the MR scans in: "tests/test_radiomic_extractor_agent/mri/mama_mia/images_pre_contrast".
The respective masks are here: "tests/test_radiomic_extractor_agent/mri/mama_mia/labels".
Save the results here: "tests/test_radiomic_extractor_agent/mri/mama_mia/results_pre_contrast".
"""

rfe_mri_prompt_2 = """
Extract shape and glrlm and ngtdm radiomic features for the MR scans in: "tests/test_radiomic_extractor_agent/mri/brats21/images_0".
The corresponding masks are here: "tests/test_radiomic_extractor_agent/mri/brats21/labels".
Use the following filters: Exponential, Gradient, SquareRoot.
Save the results here: "tests/test_radiomic_extractor_agent/mri/brats21/results_extra_filters/results_0".
"""

eda_prompt_1 = """
Perform comprehensive exploratory data analysis for the file: "data/tabulated_datasets/classification/breast_cancer_wisconsin_diagnosis_dataset.csv".
Save the output here: "tests/test_eda_agent/breast_wisconsin_eda_results".
"""

eda_prompt_2 = """
Perform comprehensive EDA for the file: "data/tabulated_datasets/classification/predict_diabetes.csv".
Save the output here: "tests/test_eda_agent/predict_diabetes_eda_results".
"""

eda_prompt_3 = """
Perform comprehensive EDA for the file: "data/tabulated_datasets/classification/heart_disease_classification.csv".
Save the output here: "tests/test_eda_agent/heart_disease_eda_results".
"""

eda_prompt_4 = """
Perform comprehensive EDA for the file: "data/tabulated_datasets/classification/heart_failure_clinical_records_dataset.csv".
Save the output here: "tests/test_eda_agent/heart_failure_eda_results".
"""

# === Example prompts dictionary ===
example_prompts = {
    "CT Radiomics (generic)": rfe_ct_prompt_1,
    "CT Radiomics (specific features/filters)": rfe_ct_prompt_2,
    "MRI Radiomics (generic)": rfe_mri_prompt_1,
    "MRI Radiomics (filters/features)": rfe_mri_prompt_2,
    "EDA - Breast Cancer": eda_prompt_1,
    "EDA - Diabetes": eda_prompt_2,
    "EDA - Heart Disease": eda_prompt_3,
    "EDA - Heart Failure": eda_prompt_4
}  # Keep this block unchanged

# === Agent blueprints ===
def build_agents(model):
    radiomic_tool = PyRadiomicsFeatureExtractionTool()
    eda_tool = ExploratoryDataAnalysisTool()
    feature_selection_tool = FeatureImportanceAnalysisTool()
    nnunet_train = NNUNetTrainingTool()
    nnunet_infer = NNUNetInferenceTool()
    totalsegmentator_tool = TotalSegmentatorTool()
    pycaret_class_train = PyCaretClassificationTool()
    pycaret_class_infer = PyCaretInferenceTool()
    pycaret_regr_train = PyCaretRegressionTool()
    pycaret_regr_infer = PyCaretRegressionInferenceTool()
    resnet_train = PyTorchResNetTrainingTool()
    resnet_infer = PyTorchResNetInferenceTool()
    vgg_train = PyTorchVGG16TrainingTool()
    vgg_infer = PyTorchVGG16InferenceTool()
    inception_train = PyTorchInceptionV3TrainingTool()
    inception_infer = PyTorchInceptionV3InferenceTool()

    radiomic_agent = CodeAgent(
        tools=[radiomic_tool],
        model=model,
        add_base_tools=True,
        additional_authorized_imports=['os', 'pandas'],
        name="radiomic_extraction_agent",
        description="Extracts radiomic features from medical images and saves them as CSV files",
        max_steps=5
    )
    eda_agent = CodeAgent(
        tools=[eda_tool],
        model=model,
        add_base_tools=True,
        additional_authorized_imports=['os', 'pandas'],
        name="eda_agent",
        description="Performs comprehensive exploratory data analysis",
        max_steps=5
    )
    feature_selection_agent = CodeAgent(
        tools=[feature_selection_tool],
        model=model,
        add_base_tools=True,
        additional_authorized_imports=['os', 'pandas'],
        name="feature_importance_agent",
        description="Performs feature importance analysis and exports the most important features to CSV files",
        max_steps=5
    )
    nnunet_agent = CodeAgent(
        tools=[nnunet_train, nnunet_infer],
        model=model,
        add_base_tools=True,
        additional_authorized_imports=['re', 'subprocess', 'os'],
        name="nnunet_agent",
        description="Uses the nnUNet framework for training and inference of medical image segmentation models",
        max_steps=5
    )
    totalsegmentator_agent = CodeAgent(
        tools=[totalsegmentator_tool],
        model=model,
        add_base_tools=True,
        additional_authorized_imports=['os'],
        name="totalsegmentator_agent",
        description="Utilizes the TotalSegmentator framework to segment organs and tissues in medical imaging data",
        max_steps=5
    )
    pycaret_classification_agent = CodeAgent(
        tools=[pycaret_class_train, pycaret_class_infer],
        model=model,
        add_base_tools=True,
        additional_authorized_imports=['pycaret', 'setup', 'compare_models', 'tune_model', 'blend_models', 'pull', 'predict_model', 'save_model', 'plot_model', 'interpret_model', 'cuml', 'pandas'],
        name="pycaret_classification_agent",
        description="Builds and deploys classification models using the PyCaret framework on tabular data inputs",
        max_steps=5
    )
    pycaret_regression_agent = CodeAgent(
        tools=[pycaret_regr_train, pycaret_regr_infer],
        model=model,
        add_base_tools=True,
        additional_authorized_imports=['pycaret', 'setup', 'compare_models', 'tune_model', 'blend_models', 'pull', 'predict_model', 'save_model', 'plot_model', 'interpret_model', 'cuml', 'pandas'],
        name="pycaret_regression_agent",
        description="Builds and deploys regression models using the PyCaret framework on tabular data inputs",
        max_steps=5
    )
    image_classification_agent = CodeAgent(
        tools=[resnet_train, resnet_infer, inception_train, inception_infer, vgg_train, vgg_infer],
        model=model,
        add_base_tools=True,
        additional_authorized_imports=['os'],
        name="image_classification_agent",
        description="Builds and deploys ResNet, Inceptionv3 and VGG16 image classification models",
        max_steps=5
    )

    master_agent = CodeAgent(
        tools=[],
        managed_agents=[radiomic_agent, eda_agent, feature_selection_agent, nnunet_agent, totalsegmentator_agent, pycaret_classification_agent, pycaret_regression_agent, image_classification_agent],
        model=model,
        add_base_tools=True,
        additional_authorized_imports=['os'],
        name="mAIstro",
        max_steps=15
    )

    return {
        "Radiomic Extraction Agent": radiomic_agent,
        "EDA Agent": eda_agent,
        "Feature Selection Agent": feature_selection_agent,
        "nnUNet Agent": nnunet_agent,
        "TotalSegmentator Agent": totalsegmentator_agent,
        "PyCaret Classification Agent": pycaret_classification_agent,
        "PyCaret Regression Agent": pycaret_regression_agent,
        "Image Classification Agent": image_classification_agent,
        "mAIstro (Master Agent)": master_agent
    }

# === Execution function ===
def run_agent(agent_name, llm_name, api_key, example_prompt, custom_prompt):
    model_id = llm_options[llm_name]
    model = LiteLLMModel(model_id=model_id, api_key=api_key)
    agents = build_agents(model)

    prompt = custom_prompt if custom_prompt.strip() else example_prompts.get(example_prompt, "")
    if not prompt:
        return "No prompt provided."

    try:
        response = agents[agent_name].run(prompt)
        return response
    except Exception as e:
        return f"Error: {str(e)}"

# === Gradio UI ===
with gr.Blocks(title="mAIstro UI") as maistro_demo:
    gr.Markdown("""
                <div align="center">
                <img src="https://raw.githubusercontent.com/eltzanis/mAIstro/main/mAIstro_logo.png" alt="mAIstro Logo" width="400"/>
                
                <h1 style='font-size: 26px;'>A multi-agentic system for automated end-to-end development of radiomics and deep learning models for medical imaging.</h1>
                </div>

                """)


    with gr.Row():
        agent_dropdown = gr.Dropdown(label="Select Agent", choices=list(build_agents(LiteLLMModel(model_id='openai/gpt-4.1', api_key='temp')).keys()), value="mAIstro (Master Agent)")
        llm_dropdown = gr.Dropdown(label="Choose LLM", choices=list(llm_options.keys()), value="OpenAI GPT-4.1")

    api_key_input = gr.Textbox(label="Enter API Key", type="password")

    with gr.Row():
        example_prompt_dropdown = gr.Dropdown(label="Choose Example Prompt", choices=list(example_prompts.keys()))
        custom_prompt_input = gr.Textbox(label="Or Enter Custom Prompt")

    prompt_display = gr.Textbox(label="Active Prompt", lines=6, interactive=False)

    # Update preview when an example is selected
    example_prompt_dropdown.change(
        lambda name: example_prompts.get(name, "") if name else "",
        inputs=example_prompt_dropdown,
        outputs=prompt_display
    )

    # Override preview when user types custom prompt
    custom_prompt_input.change(
        lambda text: text.strip(),
        inputs=custom_prompt_input,
        outputs=prompt_display
    )

    run_button = gr.Button("Run Agent")
    output_box = gr.Textbox(label="Agent Response", lines=10)

    run_button.click(
        run_agent,
        inputs=[agent_dropdown, llm_dropdown, api_key_input, example_prompt_dropdown, custom_prompt_input],
        outputs=[output_box]
    )

# Launch the interface
maistro_demo.launch()
