This project leverages the DeepSeek-V3 to recommend suitable Web APIs for Mashup applications based on their textual descriptions.

## 💡 Overview

The script reads Mashup descriptions from `shuffle_mashup_details.json`, queries the DeepSeek-V3 via API to obtain recommended APIs, and saves the results in `result.json`.

***

## 📂 File Descriptions

| File                          | Description                                             |
| :---------------------------- | :------------------------------------------------------ |
| `api_id_mapping.json`         | Maps each API URL to a unique API ID                    |
| `shuffle_mashup_details.json` | Contains Mashup descriptions and associated API usage   |
| `result.json`                 | Output file containing recommended APIs for each Mashup |
| `recommend.py`                | Main script that performs the recommendation            |

***

## 🚀 Usage

### 1. Install Dependencies

```
pip install requests

```

### 2. Run the Script

```
python recommend.py --key "Bearer sk-xxxxxxxx"

```

*   Replace `"Bearer sk-xxxxxxxx"` with your actual API authorization key.

***

## 📤 Output Format

Each line in `result.json` is a JSON object:

```
{
  "mashup_id": 1234,
  "recommend_api": [12, 45, 89, ..., 100],
  "api_target": [34, 78, 90]
}

```

*   `recommend_api`: A list of 20 recommended API IDs

*   `api_target`: Ground truth APIs originally used in the Mashup

***

## ⏳ Notes

*   **Rate Limiting**: The script sleeps for 60 seconds between API calls to avoid exceeding rate limits.

*   **Robust Parsing**: If the model's output cannot be parsed into a Python list, the script will automatically retry the request.



