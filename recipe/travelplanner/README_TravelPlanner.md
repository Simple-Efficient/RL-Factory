# TravelPlanner Environment Setup

## 1. Database Files Setup

Please place the TravelPlanner database files in the following locations:

```
envs/reward_score/database/
├── accommodations/
│   └── clean_accommodations_2022.csv
├── attractions/
│   └── attractions.csv
├── flights/
│   └── clean_Flights_2022.csv  # Note: Large file, download separately
├── restaurants/
│   └── clean_restaurant_2022.csv
└── googleDistanceMatrix/
    └── distance.csv
```

**Note**: The `clean_Flights_2022.csv` file exceeds GitHub's file size limit and needs to be downloaded separately from the original data source.

## 2. Data Preprocessing

Use the provided script to preprocess the TravelPlanner dataset:

```bash
python scripts/travelplanner_preprocess.py --local_dir /path/to/raw/data --save_dir data/Travelplanner
```

After processing, the data will be saved as `train.parquet` and `test.parquet` files, ready for training.

