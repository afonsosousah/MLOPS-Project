# Raw Data

Local raw data downloaded from the official NYC TLC trip record data page:

- Source page: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
- Monthly Green Taxi Parquet files: `https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_YYYY-MM.parquet`
- Taxi zone lookup: `https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv`

Downloaded coverage as of 2026-06-19:

- `green_taxi/2024/green_tripdata_2024-01.parquet` through `green_taxi/2024/green_tripdata_2024-12.parquet`
- `green_taxi/2025/green_tripdata_2025-01.parquet` through `green_taxi/2025/green_tripdata_2025-12.parquet`
- `green_taxi/2026/green_tripdata_2026-01.parquet` through `green_taxi/2026/green_tripdata_2026-04.parquet`
- `metadata/taxi_zone_lookup.csv`

The TLC page states that trip data is published monthly, typically with a two-month delay. Later 2026 files were not listed on the official page when this data was downloaded.

Raw files are small enough for this course repository and are not ignored by the project `.gitignore`. Temporary partial downloads ending in `.tmp` remain ignored.
