import pandas as pd

from src.config import CONTEXT_FEATURES, SERVICE_FEATURES


def sample_frame(rows=40):
    data = {}
    for column in CONTEXT_FEATURES:
        if column == "Age":
            data[column] = [20 + index % 35 for index in range(rows)]
        elif column == "Gender":
            data[column] = ["Female", "Male"] * (rows // 2) + (["Female"] if rows % 2 else [])
        elif column == "Customer Type":
            data[column] = ["Returning", "First-time"] * (rows // 2) + (["Returning"] if rows % 2 else [])
        elif column == "Type of Travel":
            data[column] = ["Business", "Personal"] * (rows // 2) + (["Business"] if rows % 2 else [])
        elif column == "Class":
            data[column] = ["Business", "Economy", "Economy Plus"] * (rows // 3) + ["Business", "Economy"][:rows % 3]
        elif column in {"Departure Delay", "Arrival Delay"}:
            data[column] = [index % 11 for index in range(rows)]
        else:
            data[column] = [200 + (index * 13) % 1000 for index in range(rows)]
    for position, column in enumerate(SERVICE_FEATURES):
        if column == "Baggage Handling":
            data[column] = [1 + index % 5 for index in range(rows)]
        else:
            data[column] = [(index + position) % 6 for index in range(rows)]
    data["ID"] = list(range(1, rows + 1))
    data["Satisfaction"] = ["Satisfied" if index % 2 else "Neutral or Dissatisfied" for index in range(rows)]
    return pd.DataFrame(data)