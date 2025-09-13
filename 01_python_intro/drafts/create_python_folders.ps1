# PowerShell script to create Python course folder structure
# Each folder contains: live_session, drafts, tasks subfolders

# Define folder names based on lesson content
$folders = @(
    "01_python_intro",
    "02_decisions_conditionals",
    "03_loops_while_for",
    "04_collections_lists_dicts",
    "05_functions_basics",
    "06_oop_fundamentals",
    "07_oop_project_management",
    "08_design_patterns",
    "09_data_structures_algorithms",
    "10_advanced_algorithms",
    "11_hashing_optimized_structures",
    "12_caching_lru_memory",
    "13_numpy_beginners",
    "14_numpy_advanced_operations",
    "15_numpy_advanced_linear_algebra",
    "16_pandas_data_manipulation",
    "17_pandas_data_cleaning",
    "18_matplotlib_basics",
    "19_matplotlib_advanced",
    "20_seaborn_statistical_viz",
    "21_seaborn_advanced",
    "22_plotly_interactive_viz",
    "23_streamlit_dashboard_basics",
    "24_streamlit_ml_predictions",
    "25_streamlit_professional_dashboards",
    "26_streamlit_database_cloud_deploy",
    "27_leetcode_exercises",
    "28_opencv_image_processing",
    "29_dlib_face_recognition",
    "30_sklearn_introduction",
    "31_sklearn_advanced_classification",
    "32_sklearn_regression_project",
    "33_sklearn_iris_classification",
    "34_nlp_text_preprocessing",
    "35_nlp_text_classification",
    "36_tensorflow_fundamentals",
    "37_tensorflow_cnn",
    "38_tensorflow_transfer_learning",
    "39_yolo_object_detection",
    "40_yolo_realtime_project",
    "41_numpy_beginners_duplicate",
    "42_keras_introduction",
    "43_keras_advanced_customization",
    "44_keras_cnn_image_classification",
    "45_keras_transfer_learning",
    "46_keras_advanced_project",
    "47_keras_ultra_advanced_deploy",
    "48_keras_model_saving_deploy",
    "49_ai_deployment_platforms",
    "50_scaling_deep_learning_apps",
    "51_emotion_assistant_nlp_project",
    "52_object_vision_lab_project"
)

# Define subfolders to create in each main folder
$subfolders = @("live_session", "drafts", "tasks")

# Create main directory if it doesn't exist
$mainPath = "Python_Course"
if (-not (Test-Path $mainPath)) {
    New-Item -ItemType Directory -Path $mainPath
    Write-Host "Created main directory: $mainPath" -ForegroundColor Green
}

# Create each folder with its subfolders
foreach ($folder in $folders) {
    $folderPath = Join-Path $mainPath $folder

    # Create main folder
    if (-not (Test-Path $folderPath)) {
        New-Item -ItemType Directory -Path $folderPath
        Write-Host "Created folder: $folder" -ForegroundColor Cyan
    }

    # Create subfolders
    foreach ($subfolder in $subfolders) {
        $subfolderPath = Join-Path $folderPath $subfolder
        if (-not (Test-Path $subfolderPath)) {
            New-Item -ItemType Directory -Path $subfolderPath
            Write-Host "  Created subfolder: $subfolder" -ForegroundColor Yellow
        }
    }
}

Write-Host "`nFolder structure creation completed!" -ForegroundColor Green
Write-Host "Total folders created: $($folders.Count)" -ForegroundColor Green
Write-Host "Each folder contains: live_session, drafts, tasks subfolders" -ForegroundColor Green