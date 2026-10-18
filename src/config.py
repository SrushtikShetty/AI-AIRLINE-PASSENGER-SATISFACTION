from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "airline_passenger_satisfaction_cleaned.csv"
MODEL_DIR = ROOT_DIR / "models"
METRICS_DIR = ROOT_DIR / "reports" / "metrics"
FIGURES_DIR = ROOT_DIR / "reports" / "figures"
SHAP_DIR = ROOT_DIR / "reports" / "shap"
PASSENGER_REPORT_DIR = ROOT_DIR / "reports" / "passenger_reports"

TARGET_COLUMN = "Satisfaction"
ID_COLUMN = "ID"
TARGET_MAPPING = {"Neutral or Dissatisfied": 0, "Satisfied": 1}
TARGET_LABELS = {0: "Neutral or Dissatisfied", 1: "Satisfied"}
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 3
CV_SAMPLE_SIZE = 24000

CONTEXT_FEATURES = [
    "Age", "Gender", "Customer Type", "Type of Travel", "Class",
    "Flight Distance", "Departure Delay", "Arrival Delay",
]
SERVICE_FEATURES = [
    "Departure and Arrival Time Convenience", "Ease of Online Booking",
    "Check-in Service", "Online Boarding", "Gate Location", "On-board Service",
    "Seat Comfort", "Leg Room Service", "Cleanliness", "Food and Drink",
    "In-flight Service", "In-flight Wifi Service", "In-flight Entertainment",
    "Baggage Handling",
]
EXPECTED_FEATURES = CONTEXT_FEATURES + SERVICE_FEATURES
RATING_FEATURES = SERVICE_FEATURES
RATING_RANGES = {feature: (0, 5) for feature in SERVICE_FEATURES}
RATING_RANGES["Baggage Handling"] = (1, 5)

for directory in (MODEL_DIR, METRICS_DIR, FIGURES_DIR, SHAP_DIR, PASSENGER_REPORT_DIR):
    directory.mkdir(parents=True, exist_ok=True)